import csv
import time
from datetime import datetime
from pathlib import Path


class DataLogger:

    def __init__(self, columns: list[str],
                 filename: bool | str | None = None,
                 label: str = 'log',
                 output_dir: str | Path | None = 'data',
                 flush_interval_s: float = 1.0) -> None:
        self._columns          = columns
        self._label            = label
        self._output_dir       = Path(output_dir) if output_dir else None
        self._flush_interval_s = max(0.0, float(flush_interval_s))
        self._f                = None
        self._writer           = None
        self._last_flush       = 0.0
        if filename:
            self.open(filename)

    # Public API

    @property
    def active(self) -> bool:
        return self._f is not None

    def open(self, filename: bool | str = True) -> None:
        if self._f is not None:
            return  # already open
        if isinstance(filename, str):
            path = filename
        else:
            name = f'{self._label}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            if self._output_dir:
                self._output_dir.mkdir(parents=True, exist_ok=True)
                path = str(self._output_dir / name)
            else:
                path = name
        self._f = open(path, 'w', newline='')
        self._writer = csv.writer(self._f)
        self._writer.writerow(self._columns)
        self._last_flush = time.monotonic()
        print(f"[{self._label}] Logging to {path}")

    def write(self, row: list) -> None:
        if self._writer:
            self._writer.writerow(row)
            now = time.monotonic()
            if now - self._last_flush >= self._flush_interval_s:
                self._f.flush()
                self._last_flush = now

    def flush(self) -> None:
        if self._f:
            self._f.flush()
            self._last_flush = time.monotonic()

    def close(self) -> None:
        if self._f:
            self._f.close()  # close() already flushes
            self._f      = None
            self._writer = None

    def __enter__(self) -> 'DataLogger':
        return self

    def __exit__(self, *_) -> None:
        self.close()
