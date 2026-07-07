# DataLogger

```python
from tmotorcan import DataLogger
# also available directly:
from motortools import DataLogger
```

Lightweight CSV logger for fixed-column telemetry. Open / close / toggle at any
time. Rows are flushed to disk on a configurable timer (`flush_interval_s`); set it
to `0` to flush after every write and guarantee no rows are lost on a hard crash.

---

## `DataLogger`

### `DataLogger(columns, filename=None, label='log', output_dir='data', flush_interval_s=1.0)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `columns` | `list[str]` | — | Ordered list of column header names (e.g. `['time_s', 'pos_rad']`). |
| `filename` | `bool \| str \| None` | `None` | `True` → auto-name into `output_dir`. String → exact path. `None` / `False` → leave closed until `open()` is called. |
| `label` | `str` | `'log'` | Prefix for auto-named files and the `[label] Logging to …` print. |
| `output_dir` | `str \| Path \| None` | `'data'` | Folder for auto-named files. `None` writes to the current directory. Ignored when `filename` is an explicit path. |
| `flush_interval_s` | `float` | `1.0` | Maximum seconds between disk flushes. `0` flushes after every write (slower but no rows lost on a hard crash). |

### Members

| Member | Description |
|---|---|
| `active` (property → `bool`) | `True` when the CSV file is open and accepting writes. |
| `open(filename=True)` | Open or reopen the CSV file. `True` → auto-name `<output_dir>/<label>_YYYYMMDD_HHMMSS.csv`. String → exact path (`output_dir` ignored). No-op if already open. |
| `write(row)` | Write one row. Flushes to disk on the `flush_interval_s` timer, not every row. No-op when closed. |
| `flush()` | Force an immediate flush to disk. Safe to call when closed. |
| `close()` | Flush and close the file. Safe to call multiple times. |
| `__enter__` / `__exit__` | Context-manager support — closes on exit. |

### Example

```python
from tmotorcan import DataLogger, RealtimeLoop

logger = DataLogger(
    ['time_s', 'pos_rad', 'vel_rad_s'],
    filename=True,          # auto-name into ./data/log_YYYYMMDD_HHMMSS.csv
    label='run1',
)

with logger:
    for t in RealtimeLoop(dt=0.01):
        motor.update(t)
        logger.write([t, motor.state.position, motor.state.velocity])
        if t > 5.0:
            break

# explicit path — output_dir is ignored
logger2 = DataLogger(['t', 'x'], 'experiments/trial.csv')
logger2.write([0.0, 1.23])
logger2.close()
```
