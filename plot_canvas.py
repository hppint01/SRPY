import matplotlib
import numpy as np
import matplotlib.dates as mdates

matplotlib.use("qtagg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from datetime import timedelta


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=8, height=6, dpi=100, nrows=1, ncols=1, sharex=True):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)

        self.axes = fig.subplots(nrows=nrows, ncols=ncols, sharex=sharex, squeeze=False)
        super().__init__(fig)
    
def trace_time_axis(trace):
    return trace.times("utcdatetime")


    
        