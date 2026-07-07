from __future__ import annotations

import csv
import os
import queue
import threading
from datetime import datetime

# Columns shared by every controller — hardware state always logged.
LOG_COMMON_COLUMNS = [
    'time_s',
    'q1_rad', 'q2_rad', 'q3_rad',
    'qdot1_rad_s', 'qdot2_rad_s', 'qdot3_rad_s',
    'tau_meas1_Nm', 'tau_meas2_Nm', 'tau_meas3_Nm',
]


class DataLogger:

    def __init__(self, prefix: str, extra_columns: list | tuple = (),
                 directory: str | None = None, path: str | None = None) -> None:
        self._columns = list(LOG_COMMON_COLUMNS) + list(extra_columns)
        if path is None:
            ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = directory or os.path.dirname(os.path.abspath(__file__))
            path     = os.path.join(save_dir, f'{prefix}_{ts}.csv')
        self.path    = path
        self._queue  = queue.Queue()
        self._thread = threading.Thread(target=self._writer, daemon=True, name='DataLogger')
        self._thread.start()
        print(f"[log] {self.path}")

    def log(self, **fields) -> None:
        self._queue.put_nowait(fields)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        print(f"[log] saved -> {self.path}")

    def _writer(self) -> None:
        with open(self.path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._columns, extrasaction='ignore')
            writer.writeheader()
            while True:
                row = self._queue.get()
                if row is None:
                    break
                writer.writerow(row)


def make_logger(prefix: str, extra_columns: list | tuple = (),
                directory: str | None = None) -> DataLogger:
    if directory is None:
        import inspect
        frame     = inspect.stack()[1]
        directory = os.path.dirname(os.path.abspath(frame.filename))
    return DataLogger(prefix, extra_columns, directory=directory)
