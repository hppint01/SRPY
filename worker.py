import traceback
from PySide6.QtCore import QObject, Signal, Slot

class Worker(QObject):
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self):
        try:
            output = self.fn(*self.args, **self.kwargs)
            self.result.emit(output)
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit((e, tb))
        finally:
            self.finished.emit()
