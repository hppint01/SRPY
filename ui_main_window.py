import os
import csv
from datetime import timedelta

import matplotlib.dates as mdates
import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox,
                                QFileDialog, QFormLayout, QInputDialog,
                                QMainWindow, QMessageBox, QMenuBar,
                                QPushButton, QSpinBox, QTextEdit, QToolBar,
                                QVBoxLayout, QWidget)

from config import TRACE_COLORS, WINDOW_SIZE
from data_loader import load_mseed
from plot_canvas import MplCanvas, trace_time_axis
from func_clean_3 import calculate_vh_data


class SpectrogramWindow(QDialog):
    def __init__(self, stream, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrogram")
        self.stream = stream

        layout = QVBoxLayout()
        self.canvas = MplCanvas(self, n_axes=len(self.stream))
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.plot_spectrogram()

    def plot_spectrogram(self):
        try:
            for i, trace in enumerate(self.stream):
                ax = self.canvas.axes[i]
                ax.specgram(trace.data, Fs=trace.stats.sampling_rate, NFFT=256, noverlap=128)

                ax.set_ylabel(trace.stats.channel, rotation=0, labelpad=25, ha='right')

            self.canvas.axes[0].set_title("Spectrogram")
            self.canvas.axes[1].set_ylabel("Frequency (Hz)")
            self.canvas.axes[-1].set_xlabel("Time (s)")
            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self,"Error", f"Could not plot spectogram: {e}")

class VHInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calculate V/H Ratio")

        self.layout = QFormLayout(self)

        self.window_len = QSpinBox(self)
        self.window_len.setRange(1, 3600)  # 1 second to 1 hour
        self.window_len.setValue(60)
        self.layout.addRow("Window Length (s):", self.window_len)

        self.shift_len = QSpinBox()
        self.shift_len.setRange(1, 3600)
        self.shift_len.setValue(30)
        self.layout.addRow("Shift Length (s):", self.shift_len) 

        self.cft_min = QDoubleSpinBox()
        self.cft_min.setRange(0.0, 10.0)
        self.cft_min.setValue(0.3)
        self.cft_min.setSingleStep(0.1)
        self.layout.addRow("STA/LTA Min", self.cft_min)

        self.cft_max = QDoubleSpinBox()
        self.cft_max.setRange(0.0, 10.0)
        self.cft_max.setValue(3.0)
        self.cft_max.setSingleStep(0.1)
        self.layout.addRow("STA/LTA Max", self.cft_max)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self) 
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def get_params(self):
        return {
            "windows": self.window_len.value(),
            "shift": self.shift_len.value(),
            "cft_min": self.cft_min.value(),
            "cft_max": self.cft_max.value()
        }
    
class VHResultWindow(QDialog):
    def __init__(self, vh_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("V/H Ratio Result")
        self.resize(800, 600)

        layout = QVBoxLayout()
        self.canvas = MplCanvas(self, n_axes=1)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.plot(vh_data)
    
    def plot(self, results):
        ax = self.canvas.axes[0]
        ax.clear()

        if not results:
            ax.text(0.5, 0.5, "No V/H ratio data available", ha='center', va='center')  
            self.canvas.draw()
            return
        
        for vh_curve in results["VperH_array"]:
            ax.plot(results["freq"], vh_curve, color='gray', alpha=0.5)

        ax.plot(results["freq"], results["median"], color='black', linewidth=2, label='Median')
        ax.plot(results["freq"], results["q1"], color='black', linewidth=1, linestyle='dashed')
        ax.plot(results["freq"], results["q3"], color='black', linewidth=1, linestyle='dashed')
        ax.plot(results["freq"], results["mean"], color='red', linewidth=2, label='Mean')

        ax.set_title(f"V/H Ratio (Window: {results['window_len']}s, Shift: {results['shift_len']}s)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("V/H Ratio")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.legend()
        self.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SRPY")

        self.stream = None
        self.current_filename = None
        self.spectrogram_window = None

        self._create_actions()
        self._build_ui()
        self._connect_signals()

    def _create_actions(self):
        
        self.act_open = QAction(QIcon.fromTheme("document open"), "Open...", self)
        self.act_open.setShortcut("Ctrl+O")

        self.act_save = QAction(QIcon.fromTheme("document-save"), "Save", self)
        self.act_save.setShortcut("Ctrl+S")
        self.act_save.setEnabled(False)

        self.act_export = QAction(QIcon.fromTheme("document-export"), "Export as...", self)
        self.act_export.setEnabled(False)

        self.act_exit = QAction(QIcon.fromTheme("application-exit"), "Exit", self)
        self.act_exit.setShortcut("Ctrl+Q")

        self.act_filter = QAction("Band-pass...", self)
        self.act_filter.setEnabled(False)

        self.act_upsample = QAction("Upsample...", self)
        self.act_upsample.setEnabled(False)

        self.act_downsample = QAction("Downsample...", self)
        self.act_downsample.setEnabled(False)

        self.act_spectrogram = QAction("Spectrogram...", self)
        self.act_spectrogram.setEnabled(False)

        self.act_vh_analysis = QAction("Calculate V/H Ratio...", self)
        self.act_vh_analysis.setEnabled(False)

        self.act_about = QAction("About SRPY", self)

    def _build_ui(self):

        menubar = QMenuBar(self)

        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.act_open)
        file_menu.addAction(self.act_save)
        file_menu.addAction(self.act_export)
        file_menu.addSeparator()
        file_menu.addAction(self.act_exit)

        edit_menu = menubar.addMenu("&Edit")

        filter_menu = edit_menu.addMenu("Filter")
        filter_menu.addAction(self.act_filter)

        resample_menu = edit_menu.addMenu("Resample")
        resample_menu.addAction(self.act_upsample)
        resample_menu.addAction(self.act_downsample)

        analysis_menu = menubar.addMenu("&Analysis")
        analysis_menu.addAction(self.act_spectrogram)
        analysis_menu.addAction(self.act_vh_analysis)

        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.act_about)

        self.setMenuBar(menubar)

        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setIconSize(QSize(24, 24))
        toolbar.addAction(self.act_open)
        toolbar.addAction(self.act_save)
        toolbar.addAction(self.act_export)
        toolbar.addSeparator()
        toolbar.addAction(self.act_exit)
        self.addToolBar(toolbar)

        layout = QVBoxLayout()

        self.button_open = QPushButton("Open")
        self.text_properties = QTextEdit("Properties")
        self.text_properties.setMaximumHeight(100)

        self.canvas = MplCanvas(self, width=8, height=6, dpi=100, n_axes=3)

        # self.slider = QSlider(Qt.Horizontal)
        # self.slider.setEnabled(False)

        layout.addWidget(self.button_open)
        layout.addWidget(self.text_properties)
        layout.addWidget(self.canvas)
        # layout.addWidget(self.slider)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _connect_signals(self):
        self.button_open.clicked.connect(self.open_file)
        # self.slider.valueChanged.connect(self.update_plot_view)
        self.act_open.triggered.connect(self.open_file)
        self.act_save.triggered.connect(self.save_file)
        self.act_export.triggered.connect(self.export_file)
        self.act_exit.triggered.connect(self.close)
        self.act_filter.triggered.connect(self.apply_bandpass)
        self.act_upsample.triggered.connect(self.upsample)
        self.act_downsample.triggered.connect(self.downsample)
        self.act_about.triggered.connect(self.show_about)
        self.act_spectrogram.triggered.connect(self.show_spectrogram)
        self.act_vh_analysis.triggered.connect(self.show_vh_analysis_dialog)


        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Waveform file", "", "Mseed files (*.mseed)"
        )

        if not file_path:
            return
        
        try:
            self.current_filename, self.stream = load_mseed(file_path)
            self.text_properties.setPlainText(str(self.stream))
            self.plot_waveform()
            # self.setup_slider()

            self.act_save.setEnabled(True)
            self.act_export.setEnabled(True)
            self.act_filter.setEnabled(True)
            self.act_upsample.setEnabled(True)
            self.act_downsample.setEnabled(True)
            self.act_spectrogram.setEnabled(True)
            self.act_vh_analysis.setEnabled(True)

        except Exception as exc:
            self.text_properties.setPlainText(f"Error: {exc}")
            self.stream = None
            # self.slider.setEnabled(False)

            self.act_save.setEnabled(False)
            self.act_export.setEnabled(False)
            self.act_filter.setEnabled(False)
            self.act_upsample.setEnabled(False)
            self.act_downsample.setEnabled(False)
            self.act_spectrogram.setEnabled(False)
            self.act_vh_analysis.setEnabled(False)

    def show_spectrogram(self):
        if not self.stream:
            QMessageBox.warning(self, "No Data", "Please open a waveform file first.")
            return
        
        self.spectrogram_window = SpectrogramWindow(self.stream, self)
        self.spectrogram_window.resize(600, 800)
        self.spectrogram_window.show()

    def show_vh_analysis_dialog(self):
        if not self.stream:
            QMessageBox.warning(self, "No Data", "Please open a waveform file first.")
            return
        
        dialog = VHInputDialog(self)
        if dialog.exec():
            params = dialog.get_params()
            try:
                vh_data = calculate_vh_data(self.stream, **params)
                result_window = VHResultWindow(vh_data, self)
                result_window.resize(800, 600)
                result_window.exec()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to calculate V/H ratio: {e}")

    def save_file(self):
        if not self.stream:
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "save waveform", self.current_filename,
            "Mseed files (.mseed)")
        if out_path:
            self.stream.write(out_path, format="MSEED")
            QMessageBox.information(self, "Saved", f"file saved to {out_path}")

    def export_file(self):
        if not self.stream:
            return
        
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export As", self.current_filename.replace(
                '.mseed', '.csv'
            ), "CSV files (*.csv)"
        )

        if not out_path:
            return
        
        trace = self.stream[0]
        times = trace_time_axis(trace)
        data = trace.data

        import csv
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'amplitude'])
            for t, d in zip (times,data):
                writer.writerow([t.isoformat(), d])
        QMessageBox.information(self, "Exported", f"CSV exported to {out_path}")

    def apply_bandpass(self):
        if not self.stream:
            return
        
        low, ok1 = QInputDialog.getDouble(self, "Band-pass filter",
                                          "Low cut-off (Hz):", 0.1, 0.0, 1000.0, 2)
        if not ok1:
            return
        
        high, ok2 = QInputDialog.getDouble(self, "Band-pass filter",
                                           "High cut-off (Hz)", 10.0, low, 1000.0, 2)
        if not ok2:
            return
        
        QMessageBox.information(self, "Filter", f"Band-pass ({low}-{high} Hz) would be applied here.")
        self.plot_waveform()

    def upsample(self):
        if not self.stream:
            return
        
        factor, ok = QInputDialog.getInt(self, "Upsample", "Upsample factor (interger > 1):",
                                         2, 2, 100, 1)
        if not ok:
            return
        QMessageBox.information(self, "Upsample", f"The stream woulb be up-sampled by a factor of {factor}.")
        self.plot_waveform()

    def downsample(self):
        if not self.stream:
            return
        
        factor, ok = QInputDialog.getInt(self, "Downsample", "Downsample factor (interger > 1):",
                                         2, 2, 100, 1)
        
        if not ok:
            return
        QMessageBox.information(self, "Downsample", f"The stream would be down-sampled by a factor of {factor}.")
        self.plot_waveform()


    # Define a constant for the initial window size in seconds


    def plot_waveform(self):
        if not self.stream:
            for ax in self.canvas.axes:
                ax.clear()
            self.canvas.draw()
            return

        for ax in self.canvas.axes:
            ax.clear()

        # Loop and plot the data
        for i, trace in enumerate(self.stream):
            ax = self.canvas.axes[i]
        
            utcdatetimes = trace.times("utcdatetime")
            times_np = np.array([t.datetime for t in utcdatetimes])
            times_mpl = mdates.date2num(times_np)
            data = trace.data
        
            ax.plot(times_mpl, data, color=TRACE_COLORS[i], linewidth=0.5)
            ax.set_ylabel(trace.id, rotation=0, labelpad=30, ha='right', va='center')
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter('%H:%M:%S')
            )
    
        # The conditional zoom logic has been removed.
        # Matplotlib will now automatically show the full duration.

        # Set title and redraw
        title = self.current_filename or "Seismic Waveform Components"
        self.canvas.axes[0].set_title(title)
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def show_about(self):
        QMessageBox.about(self, "About SRPY",
                          "On Progress")

    # def setup_slider(self):
    #     if not self.stream:
    #         self.slider.setEnabled(False)
    #         return
        
    #     total_samples = self.stream[0].stats.npts

    #     if total_samples > WINDOW_SIZE:
    #         self.slider.setEnabled(True)
    #         self.slider.setRange(0, 1000)
    #         self.slider.setValue(0)
    #         self.update_plot_view(0)

    #     else:
    #         self.slider.setEnabled(False)
    #         start_dt = self.stream[0].stats.starttime.datetime
    #         total_seconds = (total_samples - 1) / self.stream[0].stats.sampling_rate
    #         # Use the bounded helper
    #         end_dt = start_dt + _bounded_timedelta(total_seconds)

    #         for ax in self.canvas.axes:
    #             ax.set_xlim(start_dt, end_dt)
    #         self.canvas.draw()
    
    def update_plot_view(self, slider_pos: int):
        if not self.stream:
            return
        
        total_samples = self.stream[0].stats.npts
        max_start_index = total_samples - WINDOW_SIZE

        start_idx = int((slider_pos / 1000) * max_start_index)
        sr = self.stream[0].stats.sampling_rate

        start_offsec_sec = start_idx / sr
        window_offset_sec = WINDOW_SIZE / sr

        base_dt = self.stream[0].stats.starttime.datetime
        start_time = base_dt + timedelta(seconds=start_offsec_sec)
        end_time = start_time + timedelta(seconds=window_offset_sec)

        for ax in self.canvas.axes:
            ax.set_xlim(start_time, end_time)
        self.canvas.draw_idle()

    def on_scroll(self, event):
        if not self.stream:
            return
        ax = self.canvas.axes[0]
        xlim = ax.get_xlim()

        start_dt = mdates.num2date(xlim[0])
        end_dt = mdates.num2date(xlim[1])
        
        delta = (end_dt - start_dt) * 0.1

        if event.button == 'up':
            new_start = start_dt + delta
            new_end = end_dt - delta
        else:
            new_start = start_dt - delta
            new_end = end_dt + delta
        ax.set_xlim(new_start, new_end)
        self.canvas.draw_idle()

    def on_mouse_press(self, event):
        if event.button != 1 or not self.stream or event.xdata is None:
            return
        # PERBAIKAN: Ganti nama variabel agar konsisten
        self._pan_start_dt = mdates.num2date(event.xdata).replace(tzinfo=None)
        self._orig_limits = [ax.get_xlim() for ax in self.canvas.axes]

    def on_mouse_move(self, event):
        # PERBAIKAN: Sekarang pengecekan ini akan berhasil
        if not hasattr(self, "_pan_start_dt") or event.xdata is None:
            return
        
        current_dt = mdates.num2date(event.xdata).replace(tzinfo=None)
        # PERBAIKAN: Variabel ini sekarang ada
        time_delta = current_dt - self._pan_start_dt

        for ax, (orig_min_num, orig_max_num) in zip(self.canvas.axes, self._orig_limits):
            # Kita tidak perlu konversi lagi di sini karena batasnya sudah numerik
            new_min_num = orig_min_num - mdates.date2num(time_delta)
            new_max_num = orig_max_num - mdates.date2num(time_delta)

            ax.set_xlim(mdates.num2date(new_min_num), mdates.num2date(new_max_num))
        
        self.canvas.draw_idle()

    def on_mouse_release(self, event):
        # PERBAIKAN: Pengecekan ini juga sekarang benar
        if hasattr(self, "_pan_start_dt"):
            del self._pan_start_dt
            del self._orig_limits
        
