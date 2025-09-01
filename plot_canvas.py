import matplotlib
import numpy as np
import matplotlib.dates as mdates

matplotlib.use("qtagg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from datetime import timedelta




class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=8, height=6, dpi=100, n_axes=1):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = fig.subplots(n_axes, 1, sharex=True)
        if n_axes == 1:
             self.axes = [self.axes]
        super().__init__(fig)
    
def trace_time_axis(trace):
    return trace.times("utcdatetime")


    
        