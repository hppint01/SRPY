import os
from obspy import read

def load_mseed(path: str, expected_traces: int = 3):
    basename = os.path.basename(path)
    stream = read(path)

    if expected_traces is not None and len(stream) != expected_traces:
        raise ValueError(
            f"File must contain {expected_traces} traces, but found {len(stream)}."
        )
    return basename, stream
