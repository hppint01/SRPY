import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit
from plot_canvas import MplCanvas
from scipy.signal import savgol_filter as sgolay
from matplotlib.patches import Polygon

class VHResultWindow(QDialog):
    def __init__(self, vh_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("V/H Ratio Results")
        layout = QVBoxLayout(self)
        
        if not vh_data:
            self.info_text = QTextEdit("No valid data to display.")
            layout.addWidget(self.info_text)
            return

        self.canvas = MplCanvas(self, nrows=1, ncols=1)
        layout.addWidget(self.canvas)
        self.plot(vh_data)

    def plot(self, data):
        ax = self.canvas.axes[0, 0]
        freq = data["freq"]

        for vh_curve in data["VperH_array"]:
            ax.plot(freq, vh_curve, alpha=0.1, color='gray')

        median_vh = data["median"]
        ax.plot(freq, median_vh, color='black', linewidth=2, label='Median')
        ax.plot(freq, data["mean"], color='red', linewidth=2, label='Mean')
        
        ax.plot(freq, data["q1"], color='black', linewidth=1, linestyle='dashed', label='Interquartile Range')
        ax.plot(freq, data["q3"], color='black', linewidth=1, linestyle='dashed')

        try:
            fmin, fmax = 1.0, 8.0

            f_interp = np.arange(fmin, fmax, 0.01)
            vh_interp = np.interp(f_interp, freq, median_vh)

            peak_vh = np.max(vh_interp)
            peak_freq = f_interp[np.argmax(vh_interp)]

            ax.vlines(x=peak_freq, ymin=0, ymax=peak_vh, colors='red', linestyles='dashdot', label=f'Peak V/H at {peak_freq:.2f} Hz')
            
            ax.text(x=peak_freq * 1.05, y=peak_vh, 
                    s=f"f: {peak_freq:.2f} Hz\nV/H: {peak_vh:.2f}",
                    bbox=dict(facecolor='blue', alpha=0.2), wrap=True)
        except Exception as e:
            print(f"Could not calculate or plot V/H peak: {e}")

        ax.set_ylim(0, 2)
        ax.set_xlim(0.5, 10)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('V/H')
        ax.set_title(f'V/H Ratio - {data["window_count"]} Windows')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)
        self.canvas.draw()

class PSDResultWindow(QDialog):
    def __init__(self, psd_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Power Spectral Density (PSD) Results")
        layout = QVBoxLayout(self)
        self.canvas = MplCanvas(self, nrows=3, ncols=1, sharex=True)
        layout.addWidget(self.canvas)
        self.plot(psd_data)

    def plot(self, data):
        components = ["Z", "N", "E"]
        freq = data["freq"]

        for i, comp in enumerate(components):
            ax = self.canvas.axes[i, 0]
            psd_array = data[f"psd_{comp}_array"]
            
            for psd_curve in psd_array:
                ax.plot(freq, 10 * np.log10(psd_curve * 10**17), color='gray', alpha=0.1)
                
            mean_psd_scaled = data[f"mean_psd_{comp}"] * 10**17
            mean_psd_db = 10 * np.log10(mean_psd_scaled)
            mean_psd_smooth = sgolay(mean_psd_db, 101, 2)

            ax.plot(freq, mean_psd_smooth, color='black', linewidth=2, label=f'Mean PSD ({comp})')
            ax.set_ylabel("Power (dB)", rotation=90, labelpad=10)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlim(0.5, 10)
            ax.set_ylim(0, 150)

            if comp == 'Z':
                fmin, fmax = 1.0, 4.0
                finterp = np.arange(fmin, fmax, 0.05)
                Zinterp = np.interp(finterp, freq, mean_psd_smooth)
                
                peak_val = max(Zinterp)
                peak_freq = finterp[Zinterp.argmax()]
                
                psd_iz = np.trapz(Zinterp, x=finterp)
                
                verts = [(fmin, 20), *zip(finterp, Zinterp), (fmax, 20)]
                poly = Polygon(verts, facecolor='0.5', edgecolor='0.5', alpha=0.5)
                ax.add_patch(poly)

                ax.vlines(x=peak_freq, ymin=0, ymax=peak_val, colors='red', linestyles='dashdot', label='Maximum PSD-Z')
                ax.text(x=peak_freq * 1.1, y=peak_val,
                        s=f"f: {peak_freq:.2f} Hz\nZmax: {peak_val:.2f} dB\nIntegral: {psd_iz:.2f}",
                        bbox=dict(facecolor='blue', alpha=0.2), wrap=True)

        self.canvas.axes[-1, 0].set_xlabel("Frequency (Hz)")
        self.canvas.axes[0, 0].set_title(f'PSD Analysis - {data["window_count"]} Windows')
        self.canvas.figure.tight_layout()
        self.canvas.draw()

class PolarResultWindow(QDialog):
    def __init__(self, polar_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Polarization Analysis Results")
        layout = QVBoxLayout(self)
        self.canvas = MplCanvas(self, nrows=2, ncols=2, sharex=False)
        layout.addWidget(self.canvas)
        self.plot(polar_data)

    def plot(self, data):
        ax_L = self.canvas.axes[0, 0]
        ax_dip = self.canvas.axes[0, 1]
        ax_az = self.canvas.axes[1, 0]
        ax_eig = self.canvas.axes[1, 1]

        t_values = data['t_list'] 
        
        # Plot Rectilinearity
        ax_L.scatter(t_values, data['L_list'], color='black')
        ax_L.axhline(data['L_all'], color='black', label='All Data')
        ax_L.axhline(data['mean_L'], color='red', linestyle='dashed', label='Mean of Windows')
        ax_L.set_ylabel('Rectilinearity')
        ax_L.set_ylim(0, 1)

        # Plot Dip
        ax_dip.scatter(t_values, data['dip_list'], color='black')
        ax_dip.axhline(data['dip_all'], color='black')
        ax_dip.axhline(data['mean_dip'], color='red', linestyle='dashed')
        ax_dip.set_ylabel('Dip (°)')
        ax_dip.set_ylim(0, 90)

        # Plot Azimuth
        ax_az.scatter(t_values, data['azimuth_list'], color='black')
        ax_az.axhline(data['azimuth_all'], color='black')
        ax_az.axhline(data['mean_azimuth'], color='red', linestyle='dashed')
        ax_az.set_ylabel('Azimuth (°)')
        ax_az.set_ylim(-50, 50)

        # Plot Eigenvalue
        ax_eig.scatter(t_values, data['eig_list'], color='black')
        ax_eig.axhline(data['eig_all'], color='black')
        ax_eig.axhline(data['mean_eig'], color='red', linestyle='dashed')
        ax_eig.set_ylabel('Largest Eigenvalue')
        
        ax_az.set_xlabel("Time")
        ax_eig.set_xlabel("Time")

        for ax in self.canvas.axes.flat:
            ax.grid(True, linestyle='--', alpha=0.6)
            if ax.get_ylabel() != 'Largest Eigenvalue': 
                ax.legend()
            if ax.get_xticklabels():
                 ax.tick_params(axis='x', rotation=30)


        self.canvas.figure.suptitle(f'Polarization Analysis - {data["window_count"]} Windows')
        self.canvas.figure.tight_layout()
        self.canvas.draw()