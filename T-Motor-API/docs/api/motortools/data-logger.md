# DataLogger

```python
from tmotorcan import DataLogger
```

Lightweight CSV logger for fixed-column telemetry. Used internally by
`MITMotor` for its CSV output, but also useful standalone — e.g. logging
sensor data alongside the motor's own log.

Every `write()` call flushes to disk so a crash leaves the last row intact.

---

## `DataLogger(columns, filename=None, label='log', output_dir='data')`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `columns` | `list[str]` | — | Ordered list of column header names. |
| `filename` | `bool \| str \| None` | `None` | `True` opens auto-named file. A string is the exact path. `None` / `False` leaves the logger closed. |
| `label` | `str` | `'log'` | Prefix for auto-named files and used in the `[label] Logging to …` print. |
| `output_dir` | `str \| Path \| None` | `'data'` | Folder for auto-named files. `None` writes to the current directory. Ignored when `filename` is an explicit path. |

---

## Methods and properties

### `logger.active` (property → `bool`)

`True` when the file is open and accepting writes.

### `open(filename=True)`

Open or reopen the file. No-op if already open.

```python
logger.open(True)              # auto: data/log_20260507_143022.csv
logger.open('runs/r1.csv')     # exact path; output_dir is ignored
```

### `write(row)`

Write one row. `row` must be a list of values matching the column count.
No-op when the logger is closed.

### `close()`

Flush and close. Safe to call multiple times.

---

## Example

```python
from tmotorcan import DataLogger

logger = DataLogger(
    columns=['time_s', 'pos_rad', 'vel_rad_s'],
    filename='run1.csv',
    label='exp',
)

for t, pos, vel in samples:
    logger.write([t, pos, vel])

logger.close()
```

### Context manager

```python
with DataLogger(['t', 'x'], filename=True, label='probe') as log:
    log.write([0.0, 1.23])
# auto-closes on exit
```

### Custom output directory

```python
DataLogger(['t', 'x'], output_dir='runs/2026-05-07')
DataLogger(['t', 'x'], output_dir=None)    # writes to CWD instead
```
