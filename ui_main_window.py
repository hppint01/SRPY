import os
import matplotlib.dates as mdates
import numpy as np

from PySide6.QtWidgets import (QMainWindow, QPushButton, QTextEdit,
                               QVBoxLayout, QWidget, QFileDialog,
                               QSlider, QToolBar, QMessageBox, QMenuBar, 
                               QInputDialog, QDialog)
from PySide6.QtGui import QAction, QIcon
from PySide6.QtCore import Qt, QSize

from config import WINDOW_SIZE, TRACE_COLORS
from data_loader import load_mseed
from plot_canvas import MplCanvas, trace_time_axis
from datetime import timedelta

MAX_TIMESPAN_SECONDS = 30 * 24 * 60 * 60   # 30 days

def _bounded_timedelta(seconds: float) -> timedelta:
    """Return a timedelta limited to MAX_TIMESPAN_SECONDS."""
    if seconds > MAX_TIMESPAN_SECONDS:
        # Show the full span in the title instead of trying to plot it.
        return timedelta(seconds=MAX_TIMESPAN_SECONDS)
    return timedelta(seconds=int(seconds))

SLIDER_STEP = 1
DEFAULT_ZOOM_FACTOR = 1.5

class SpectrogramWindow(QDialog):
    def __init__(self, stream, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrogram")
        self.stream = stream

        layout = QVBoxLayout()
        self.canvas = MplCanvas(self, n_axes=len(self.stream))
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.plot_spectogram()

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

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)

        layout.addWidget(self.button_open)
        layout.addWidget(self.text_properties)
        layout.addWidget(self.canvas)
        layout.addWidget(self.slider)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _connect_signals(self):
        self.button_open.clicked.connect(self.open_file)
        self.slider.valueChanged.connect(self.update_plot_view)
        self.act_open.triggered.connect(self.open_file)
        self.act_save.triggered.connect(self.save_file)
        self.act_export.triggered.connect(self.export_file)
        self.act_exit.triggered.connect(self.close)
        self.act_filter.triggered.connect(self.apply_bandpass)
        self.act_upsample.triggered.connect(self.upsample)
        self.act_downsample.triggered.connect(self.downsample)
        self.act_about.triggered.connect(self.show_about)
        self.act_spectrogram.triggered.connect(self.show_spectrogram)

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
            self.setup_slider()

            self.act_save.setEnabled(True)
            self.act_export.setEnabled(True)
            self.act_filter.setEnabled(True)
            self.act_upsample.setEnabled(True)
            self.act_downsample.setEnabled(True)
            self.act_spectrogram.setEnabled(True)

        except Exception as exc:
            self.text_properties.setPlainText(f"Error: {exc}")
            self.stream = None
            self.slider.setEnabled(False)

            self.act_save.setEnabled(False)
            self.act_export.setEnabled(False)
            self.act_filter.setEnabled(False)
            self.act_upsample.setEnabled(False)
            self.act_downsample.setEnabled(False)
            self.act_spectrogram.setEnabled(False)

    def show_spectrogram(self):
        if not self.stream:
            QMessageBox.warning(self, "No Data", "Please open a waveform file first.")
            return
        
        self.spectrogram_window = SpectrogramWindow(self.stream, self)
        self.spectrogram_window.resize(600, 800)
        self.spectrogram_window.show()

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

    def plot_waveform(self):
        if not self.stream:
            return
        
        for ax in self.canvas.axes:
            ax.clear()

        for i, trace in enumerate(self.stream):
            ax = self.canvas.axes[i]
            times = trace_time_axis(trace)
            ax.plot(times, trace.data, color=TRACE_COLORS[i], linewidth=0.5)
            #ax.plot(trace.data, color=TRACE_COLORS[i], linewidth=0.5)
            ax.set_ylabel(trace.id, rotation=0, labelpad=25)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(
                #mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
                mdates.DateFormatter('%H:%M:%S')
            )

        title = self.current_filename or "Seismic Waveform Components"
        self.canvas.axes[0].set_title(title)

        self.canvas.draw()

    def show_about(self):
        QMessageBox.about(self, "About SRPY",
                          "On Progress")

    def setup_slider(self):
        if not self.stream:
            self.slider.setEnabled(False)
            return
        
        total_samples = self.stream[0].stats.npts

        if total_samples > WINDOW_SIZE:
            self.slider.setEnabled(True)
            self.slider.setRange(0, 1000)
            self.slider.setValue(0)
            self.update_plot_view(0)

        else:
            self.slider.setEnabled(False)
            start_dt = self.stream[0].stats.starttime.datetime
            total_seconds = (total_samples - 1) / self.stream[0].stats.sampling_rate
            # Use the bounded helper
            end_dt = start_dt + _bounded_timedelta(total_seconds)

            for ax in self.canvas.axes:
                ax.set_xlim(start_dt, end_dt)
            self.canvas.draw()
    
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
        if event.button != 1 or not self.stream:
            return
        self._pan_start = (event.xdata, event.ydata)
        self._orig_limits = [ax.get_xlim() for ax in self.canvas.axes]

    def on_mouse_move(self, event):
        if not hasattr(self, "_pan_start") or event.xdata is None:
            return
        
        dx = event.xdata - self._pan_start[0]

        for ax, (orig_min, orig_max) in zip(self.canvas.axes, self._orig_limits):
            ax.set_xlim(orig_min - dx, orig_max - dx)
        self.canvas.draw_idle()

    def on_mouse_release(self, event):
        if hasattr(self, "_pan_start"):
            del self._pan_start
            del self._orig_limits
        
