# Filters

```python
from motortools import IIRFilter, FiniteDiff, PositionUnwrapper
# IIRFilter is also re-exported via tmotorcan:
from tmotorcan import IIRFilter
```

Digital signal filters for robotics sensor data: a first-order low-pass IIR,
a finite-difference derivative estimator, and a multi-turn position unwrapper.

---

## `IIRFilter`

First-order IIR (exponential moving average) low-pass filter.

```
y[n] = α × x[n] + (1 − α) × y[n−1]
```

| `alpha` | Effect |
|---|---|
| → `0.0` | heavy smoothing |
| `0.5` | moderate smoothing |
| `1.0` | passthrough |

Used inside `MITMotor.set_vel_filter()`, but also useful standalone for any
noisy signal (force sensor, derived torque, etc.).

### `IIRFilter(alpha, initial=0.0)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alpha` | `float` | — | Smoothing coefficient in `(0.0, 1.0]`. Out-of-range values raise `ValueError`. |
| `initial` | `float` | `0.0` | Starting value of the filter output. |

### Members

| Member | Description |
|---|---|
| `filt(x)` (call) | Feed one sample, return filtered output. |
| `filt.value` | Current output without feeding a new sample. |
| `filt.alpha` | Read or write `alpha`. Setter validates range. Re-tuning preserves internal state — no output glitch. |
| `filt.reset(value=0.0)` | Reset state to `value`. |

### Example

```python
from motortools import IIRFilter

filt = IIRFilter(alpha=0.1)
for raw in noisy_stream:
    smooth = filt(raw)

filt.alpha = 0.3      # retune; state preserved
filt.reset()          # back to 0.0
```

---

## `FiniteDiff`

Finite-difference derivative estimator. Returns `dy/dt` from consecutive
`(value, timestamp)` pairs. Returns `None` on the first call (no prior sample)
and any tick where `dt <= 0`.

### `FiniteDiff()`

No arguments.

### Members

| Member | Description |
|---|---|
| `diff(value, t)` (call) | Feed one `(value, time)` sample. Returns derivative or `None` on first call. |
| `diff.reset()` | Clear state — next call returns `None` again. |

### Example

```python
from motortools import FiniteDiff

diff = FiniteDiff()
for t, vel in samples:
    accel = diff(vel, t)
    if accel is not None:
        print(accel)
```

---

## `PositionUnwrapper`

Multi-turn position unwrapper. Removes a periodic encoder's range limit by
detecting wraparound crossings and accumulating an offset. Suitable for any
sensor with a fixed period (e.g. ±12.5 rad → period = 25.0 rad).

Used inside `MITMotor.set_multi_turn()`, but also useful standalone.

### `PositionUnwrapper(period)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `period` | `float` | — | Full range of the raw encoder value (e.g. `p_max - p_min = 25.0`). |

### Members

| Member | Description |
|---|---|
| `unwrap(raw)` (call) | Feed one raw sample, return unwrapped continuous position. |
| `unwrap.reset()` | Clear accumulated offset — call before a new motion segment. |

### Example

```python
from motortools import PositionUnwrapper

unwrap = PositionUnwrapper(period=25.0)   # ±12.5 rad encoder
for raw in encoder_stream:
    continuous = unwrap(raw)
```
