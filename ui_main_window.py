import os
import csv
import traceback
from datetime import timedelta
from scipy.signal import savgol_filter as sgolay

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import Qt, QSize, QThread
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QDoubleSpinBox,
                                QFileDialog, QFormLayout, QInputDialog,
                                QMainWindow, QMessageBox, QMenuBar,
                                QPushButton, QSpinBox, QTextEdit, QToolBar,
                                QVBoxLayout, QWidget, QProgressDialog)

from config import TRACE_COLORS, WINDOW_SIZE
from data_loader import load_mseed
from plot_canvas import MplCanvas

from worker import Worker
from ui_result_windows import VHResultWindow, PSDResultWindow, PolarResultWindow
from func_clean_3 import calculate_vh_data, calculate_psd_for_gui, calculate_polar_for_gui


class SpectrogramWindow(QDialog):
    def __init__(self, stream, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrogram")
        self.stream = stream

        layout = QVBoxLayout()
        self.canvas = MplCanvas(self, n_axes=len(self.stream), ncols=1)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.plot_spectrogram()

    def plot_spectrogram(self):
        try:
            for i, trace in enumerate(self.stream):
                ax = self.canvas.axes[i, 0]
                ax.specgram(trace.data, Fs=trace.stats.sampling_rate, NFFT=256, noverlap=128)
                ax.set_ylabel(trace.stats.channel, rotation=0, labelpad=25, ha='right')

            self.canvas.axes[0, 0].set_title("Spectrogram")
            midle_ax_index = len(self.stream) // 2
            self.canvas.axes[midle_ax_index, 0].set_ylabel("Frequency (Hz)")
            self.canvas.axes[-1, 0].set_xlabel("Time (s)")
            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self,"Error", f"Could not plot spectogram: {e}")

class BaseAnalysisDialog(QDialog):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.layout = QFormLayout(self)

        self.window_len = QSpinBox(self)
        self.window_len.setRange(1, 3600); self.window_len.setValue(60)
        self.layout.addRow("Window Length (s):", self.window_len)

        self.shift_len = QSpinBox()
        self.shift_len.setRange(1, 3600); self.shift_len.setValue(30)
        self.layout.addRow("Shift Length (s):", self.shift_len)

        self.cft_min = QDoubleSpinBox()
        self.cft_min.setRange(0.0, 10.0); self.cft_min.setValue(0.3); self.cft_min.setSingleStep(0.1)
        self.layout.addRow("STA/LTA Min", self.cft_min)

        self.cft_max = QDoubleSpinBox()
        self.cft_max.setRange(0.0, 10.0); self.cft_max.setValue(3.0); self.cft_max.setSingleStep(0.1)
        self.layout.addRow("STA/LTA Max", self.cft_max)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self) 
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def get_common_params(self):
        return {
            "windows": self.window_len.value(),
            "shift": self.shift_len.value(),
            "cft_max": self.cft_max.value(),
            "cft_min": self.cft_min.value()
        }  
    
class VHInputDialog(BaseAnalysisDialog):
    def __init__(self, parent=None):
        super().__init__("V/H Ratio Analysis", parent)
        self.layout.addWidget(self.button_box)
    def get_params(self): return self.get_common_params()

class PSDInputDialog(BaseAnalysisDialog):
    def __init__(self, parent=None):
        super().__init__("Power Spectral Density (PSD)", parent)
        self.dip_tres = QSpinBox()
        self.dip_tres.setRange(0, 90); self.dip_tres.setValue(45)
        self.layout.addRow("Dip Threshold (degrees):", self.dip_tres)
        self.layout.addWidget(self.button_box)
    def get_params(self):
        params = self.get_common_params()
        params["dip_tres"] = self.dip_tres.value()
        return params
    
class PolarInputDialog(BaseAnalysisDialog):
    def __init__(self, parent=None):
        super().__init__("Polarization Analysis", parent)
        self.dip_tres = QSpinBox()
        self.dip_tres.setRange(0, 90); self.dip_tres.setValue(45)
        self.layout.addRow("Dip Threshold (degrees):", self.dip_tres)
        self.layout.addWidget(self.button_box)
    def get_params(self):
        params = self.get_common_params()
        params["dip_tres"] = self.dip_tres.value()
        return params

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SRPY")
        self.stream = None
        self.current_filename = None
        self.spectrogram_window = None
        # Atribut untuk menyimpan referensi thread & worker
        self.thread = None
        self.worker = None

        self._create_actions()
        self._build_ui()
        self._connect_signals()

    def _create_actions(self):
        self.act_open = QAction(QIcon.fromTheme("document-open"), "Open...", self); self.act_open.setShortcut("Ctrl+O")
        self.act_save = QAction(QIcon.fromTheme("document-save"), "Save", self); self.act_save.setShortcut("Ctrl+S"); self.act_save.setEnabled(False)
        self.act_export = QAction(QIcon.fromTheme("document-export"), "Export as...", self); self.act_export.setEnabled(False)
        self.act_exit = QAction(QIcon.fromTheme("application-exit"), "Exit", self); self.act_exit.setShortcut("Ctrl+Q")
        self.act_filter = QAction("Band-pass...", self); self.act_filter.setEnabled(False)
        self.act_upsample = QAction("Upsample...", self); self.act_upsample.setEnabled(False)
        self.act_downsample = QAction("Downsample...", self); self.act_downsample.setEnabled(False)
        self.act_spectrogram = QAction("Spectrogram...", self); self.act_spectrogram.setEnabled(False)
        self.act_vh_analysis = QAction("Calculate V/H Ratio...", self); self.act_vh_analysis.setEnabled(False)
        self.act_psd_analysis = QAction("Calculate PSD...", self); self.act_psd_analysis.setEnabled(False)
        self.act_polar_analysis = QAction("Polarization Analysis...", self); self.act_polar_analysis.setEnabled(False)
        self.act_about = QAction("About SRPY", self)

    def _build_ui(self):
        menubar = QMenuBar(self)
        file_menu = menubar.addMenu("&File"); file_menu.addAction(self.act_open); file_menu.addAction(self.act_save); file_menu.addAction(self.act_export); file_menu.addSeparator(); file_menu.addAction(self.act_exit)
        edit_menu = menubar.addMenu("&Edit"); filter_menu = edit_menu.addMenu("Filter"); filter_menu.addAction(self.act_filter); resample_menu = edit_menu.addMenu("Resample"); resample_menu.addAction(self.act_upsample); resample_menu.addAction(self.act_downsample)
        analysis_menu = menubar.addMenu("&Analysis"); analysis_menu.addAction(self.act_spectrogram); analysis_menu.addAction(self.act_vh_analysis); analysis_menu.addAction(self.act_psd_analysis); analysis_menu.addAction(self.act_polar_analysis)
        help_menu = menubar.addMenu("&Help"); help_menu.addAction(self.act_about)
        self.setMenuBar(menubar)
        toolbar = QToolBar("Main Toolbar", self); toolbar.setIconSize(QSize(24, 24)); toolbar.addAction(self.act_open); toolbar.addAction(self.act_save); toolbar.addAction(self.act_export); toolbar.addSeparator(); toolbar.addAction(self.act_exit); self.addToolBar(toolbar)
        layout = QVBoxLayout(); self.button_open = QPushButton("Open"); self.text_properties = QTextEdit("Properties"); self.text_properties.setMaximumHeight(100); self.canvas = MplCanvas(self, width=8, height=6, dpi=100, nrows=3, ncols=1); layout.addWidget(self.button_open); layout.addWidget(self.text_properties); layout.addWidget(self.canvas)
        container = QWidget(); container.setLayout(layout); self.setCentralWidget(container)

    def _connect_signals(self):
        self.button_open.clicked.connect(self.open_file); self.act_open.triggered.connect(self.open_file); self.act_save.triggered.connect(self.save_file); self.act_export.triggered.connect(self.export_file); self.act_exit.triggered.connect(self.close); self.act_filter.triggered.connect(self.apply_bandpass); self.act_upsample.triggered.connect(self.upsample); self.act_downsample.triggered.connect(self.downsample); self.act_about.triggered.connect(self.show_about); self.act_spectrogram.triggered.connect(self.show_spectrogram)
        self.act_vh_analysis.triggered.connect(self.run_vh_analysis)
        self.act_psd_analysis.triggered.connect(self.run_psd_analysis) 
        self.act_polar_analysis.triggered.connect(self.run_polar_analysis)
        self.canvas.mpl_connect("scroll_event", self.on_scroll); self.canvas.mpl_connect("button_press_event", self.on_mouse_press); self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move); self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

    # --- Bagian yang Diperbarui: Fungsi Analisis ---

    def run_vh_analysis(self):
        if not self.stream: return QMessageBox.warning(self, "No Data", "Please open a waveform file first.")
        dialog = VHInputDialog(self)
        if dialog.exec():
            params = dialog.get_params()
            self.start_analysis(calculate_vh_data, self.on_vh_analysis_complete, "Calculating V/H Ratio...", **params)
            
    def run_psd_analysis(self):
        if not self.stream: return QMessageBox.warning(self, "No Data", "Please open a waveform file first.")
        dialog = PSDInputDialog(self)
        if dialog.exec():
            params = dialog.get_params()
            self.start_analysis(calculate_psd_for_gui, self.on_psd_analysis_complete, "Calculating PSD...", **params)
            
    def run_polar_analysis(self):
        if not self.stream: return QMessageBox.warning(self, "No Data", "Please open a waveform file first.")
        dialog = PolarInputDialog(self)
        if dialog.exec():
            params = dialog.get_params()
            self.start_analysis(calculate_polar_for_gui, self.on_polar_analysis_complete, "Calculating Polarization...", **params)
    
    def start_analysis(self, func, slot, progress_title, **kwargs):
        self.thread = QThread()
        self.worker = Worker(func, self.stream, **kwargs)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.result.connect(slot)
        self.worker.error.connect(self.on_analysis_error)
        
        self.thread.start()

        self.prog = QProgressDialog(progress_title, "Cancel", 0, 0, self)
        self.prog.setWindowModality(Qt.WindowModal)
        self.prog.show()

        QApplication.processEvents()
        
        self.worker.finished.connect(self.prog.close)

    def on_vh_analysis_complete(self, vh_data):
        if not vh_data or "VperH_array" not in vh_data or len(vh_data["VperH_array"]) == 0 :
            QMessageBox.information(self, "No Data", "No valid V/H ratio data could be calculated.")
            return

        self.plot_waveform()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(vh_data["valid_windows"])))
        self.draw_windows_on_waveform(vh_data["valid_windows"], colors)
        
        result_window = VHResultWindow(vh_data, self)
        result_window.resize(800, 600)
        result_window.exec()

    def on_psd_analysis_complete(self, psd_data):
        if not psd_data:
            QMessageBox.information(self, "No Data", "No valid PSD data could be calculated (check parameters).")
            return

        self.plot_waveform()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(psd_data["valid_windows"])))
        self.draw_windows_on_waveform(psd_data["valid_windows"], colors)

        result_window = PSDResultWindow(psd_data, self)
        result_window.resize(800, 600)
        result_window.exec()

    def on_polar_analysis_complete(self, polar_data):
        if not polar_data:
            QMessageBox.information(self, "No Data", "No valid polarization data could be calculated (check parameters).")
            return
        
        self.plot_waveform()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(polar_data["valid_windows"])))
        self.draw_windows_on_waveform(polar_data["valid_windows"], colors)
        
        result_window = PolarResultWindow(polar_data, self)
        result_window.resize(800, 600)
        result_window.exec()

    def on_analysis_error(self, err_tuple):
        e, tb = err_tuple
        print("="*50)
        print("An error occurred in the worker thread:")
        print(tb)
        print("="*50)
        QMessageBox.critical(self, "Analysis Error", f"An error occurred during analysis: {e}")

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Waveform file", "", "Mseed files (*.mseed)")
        if not file_path: 
            return

        def load_file_wrapper(stream_placeholder, path):
            return load_mseed(path)
        
        self.start_analysis(load_file_wrapper, self.on_file_load_complete, "Loading waveform data...", path=file_path)

    def on_file_load_complete(self, result):
        try:
            self.current_filename, self.stream = result
            self.text_properties.setPlainText(str(self.stream))
            self.plot_waveform()

            for act in [self.act_save, self.act_export, self.act_filter, self.act_upsample, self.act_downsample, self.act_spectrogram, self.act_vh_analysis, self.act_psd_analysis, self.act_polar_analysis]:
                act.setEnabled(True)
        except Exception as exc:
            self.on_analysis_error((exc, traceback.format_exc()))
            self.stream = None
            for act in [self.act_save, self.act_export, self.act_filter, self.act_upsample, self.act_downsample, self.act_spectrogram, self.act_vh_analysis, self.act_psd_analysis, self.act_polar_analysis]:
                act.setEnabled(False)

    def show_spectrogram(self):
        if not self.stream: return QMessageBox.warning(self, "No Data", "Please open a waveform file first.")
        self.spectrogram_window = SpectrogramWindow(self.stream, self)
        self.spectrogram_window.resize(600, 800)
        self.spectrogram_window.show()

    def draw_windows_on_waveform(self, windows, colors):
        if not windows: return
        for i, (start_time, end_time) in enumerate(windows):
            for ax_row in self.canvas.axes:
                ax = ax_row[0]
                ax.axvspan(start_time, end_time, color=colors[i], alpha=0.3)
        self.canvas.draw_idle()

    def save_file(self):
        if not self.stream: return
        out_path, _ = QFileDialog.getSaveFileName(self, "save waveform", self.current_filename, "Mseed files (.mseed)")
        if out_path: self.stream.write(out_path, format="MSEED"); QMessageBox.information(self, "Saved", f"file saved to {out_path}")

    def export_file(self):
        if not self.stream: return
        out_path, _ = QFileDialog.getSaveFileName(self, "Export As", self.current_filename.replace('.mseed', '.csv'), "CSV files (*.csv)")
        if not out_path: return
        trace = self.stream[0]; times = trace.times("utcdatetime"); data = trace.data
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f); writer.writerow(['timestamp', 'amplitude'])
            for t, d in zip (times,data): writer.writerow([t.isoformat(), d])
        QMessageBox.information(self, "Exported", f"CSV exported to {out_path}")

    def apply_bandpass(self):
        if not self.stream: return
        low, ok1 = QInputDialog.getDouble(self, "Band-pass filter", "Low cut-off (Hz):", 0.1, 0.0, 1000.0, 2)
        if not ok1: return
        high, ok2 = QInputDialog.getDouble(self, "Band-pass filter", "High cut-off (Hz)", 10.0, low, 1000.0, 2)
        if not ok2: return
        self.stream.filter("bandpass", freqmin=low, freqmax=high)
        QMessageBox.information(self, "Filter Applied", f"Band-pass ({low}-{high} Hz) has been applied.")
        self.plot_waveform()

    def upsample(self):
        if not self.stream: return
        factor, ok = QInputDialog.getInt(self, "Upsample", "Upsample factor (integer > 1):", 2, 2, 100, 1)
        if not ok: return
        self.stream.resample(self.stream[0].stats.sampling_rate * factor)
        QMessageBox.information(self, "Upsampled", f"The stream has been up-sampled by a factor of {factor}.")
        self.plot_waveform()

    def downsample(self):
        if not self.stream: return
        factor, ok = QInputDialog.getInt(self, "Downsample", "Downsample factor (integer > 1):", 2, 2, 100, 1)
        if not ok: return
        self.stream.resample(self.stream[0].stats.sampling_rate / factor)
        QMessageBox.information(self, "Downsampled", f"The stream has been down-sampled by a factor of {factor}.")
        self.plot_waveform()

    def plot_waveform(self):
        for ax_row in self.canvas.axes:
            ax_row[0].clear()
        if not self.stream: 
            return self.canvas.draw()
        for i, trace in enumerate(self.stream):
            ax = self.canvas.axes[i, 0]
            times_mpl = [t.matplotlib_date for t in trace.times("utcdatetime")]
            ax.plot(times_mpl, trace.data, color=TRACE_COLORS[i], linewidth=0.5)
            ax.set_ylabel(trace.id, rotation=0, labelpad=30, ha='right', va='center')
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        title = self.current_filename or "Seismic Waveform Components"
        self.canvas.axes[0, 0].set_title(title)
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def show_about(self): QMessageBox.about(self, "About SRPY", "Seismic Analysis Tool with Python")

    def on_scroll(self, event):
        if not self.stream: return
        ax = self.canvas.axes[0]
        xlim = self.canvas.axes[0].get_xlim()
        start_dt = mdates.num2date(xlim[0]); end_dt = mdates.num2date(xlim[1])
        delta = (end_dt - start_dt) * 0.1
        if event.button == 'up': new_start = start_dt + delta; new_end = end_dt - delta
        else: new_start = start_dt - delta; new_end = end_dt + delta
        for ax_ in self.canvas.axes: ax_.set_xlim(new_start, new_end)
        self.canvas.draw_idle()

    def on_mouse_press(self, event):
        if event.button != 1 or not self.stream or event.xdata is None: return
        self._pan_start_x = event.xdata
        self._orig_limits = [ax.get_xlim() for ax in self.canvas.axes]

    def on_mouse_move(self, event):
        if not hasattr(self, "_pan_start_x") or event.xdata is None: return
        dx = event.xdata - self._pan_start_x
        for ax, (orig_min, orig_max) in zip(self.canvas.axes, self._orig_limits):
            ax.set_xlim(orig_min - dx, orig_max - dx)
        self.canvas.draw_idle()

    def on_mouse_release(self, event):
        if hasattr(self, "_pan_start_x"):
            del self._pan_start_x
            del self._orig_limits