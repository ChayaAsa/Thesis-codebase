# IIRFilter

```python
from tmotorcan import IIRFilter
```

First-order IIR (exponential moving average) low-pass filter. Used inside
`MITMotor.set_vel_filter()`, but also useful standalone for any noisy signal
(force-sensor input, derived torque, etc.).

```
y[n] = α × x[n] + (1 − α) × y[n−1]
```

| `alpha` | Effect |
|---|---|
| → `0.0` | heavy smoothing |
| `0.5` | moderate smoothing |
| `1.0` | passthrough |

---

## `IIRFilter(alpha, initial=0.0)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alpha` | `float` | — | Smoothing coefficient in `(0.0, 1.0]`. Out-of-range values raise `ValueError`. |
| `initial` | `float` | `0.0` | Starting value of the filter output. |

---

## Methods and properties

| Member | Description |
|---|---|
| `filt(x)` (call) | Feed one sample, return filtered output. |
| `filt.value` | Current output (without feeding a new sample). |
| `filt.alpha` | Read or write `alpha`. Setter validates the new value. Re-tuning preserves filter state — no glitch. |
| `filt.reset(value=0.0)` | Reset state to `value`. |

---

## Example

```python
from tmotorcan import IIRFilter

filt = IIRFilter(alpha=0.1)            # heavy smoothing

for raw in noisy_stream:
    smooth = filt(raw)
    print(smooth)

# retune on the fly
filt.alpha = 0.3                       # state is preserved
filt.reset()                           # back to 0.0
```
