# Limits

```python
from motortools import SoftLimiter, ThermalDerate
```

Safety envelopes for motor commands: a per-field min/max clamp and a thermal
derating curve. Used internally by `MITMotor.set_limits()` and
`MITMotor.set_temp_derating()`, but also useful standalone.

---

## `SoftLimiter`

Per-field clamp for named values (e.g. `'position'`, `'velocity'`, `'torque'`).
Fields with no limit set pass through unchanged. Clamping is silent —
read `limiter.last_clamped` to see which fields were clipped on the most
recent tick.

### `SoftLimiter()`

No arguments.

### Members

| Member | Description |
|---|---|
| `set(field, (lo, hi))` | Set clamp range for one field. Raises `ValueError` if `lo >= hi`. |
| `set(field, None)` | Remove the limit on `field`. |
| `get(field)` | Return current `(lo, hi)`, or `None` if no limit is set. |
| `active(field=None)` | `True` if `field` has a limit. If `field` is `None`, `True` if any limit is set. |
| `clamp(field, value)` | Return `value` clipped into the field's range. Records the field in `last_clamped` if clipping occurred. |
| `last_clamped` (property → `set[str]`) | Fields clipped since the last `reset_tracking()`. |
| `reset_tracking()` | Clear `last_clamped`. Call once per tick, before `clamp()`. |

### Example

```python
from motortools import SoftLimiter

lim = SoftLimiter()
lim.set('position', (-1.0, 1.0))
lim.set('torque',   (-5.0, 5.0))

for t in loop:
    lim.reset_tracking()
    pos = lim.clamp('position', target_pos)
    tau = lim.clamp('torque',   target_tau)
    if lim.last_clamped:
        print(f"clipped: {lim.last_clamped}")
```

---

## `ThermalDerate`

Linear torque-command multiplier that ramps `1.0 → 0.0` as temperature rises:

| Temperature | Output |
|---|---|
| `temp ≤ start_C` | `1.0` (no derating) |
| `start_C < temp < end_C` | linear ramp |
| `temp ≥ end_C` | `0.0` |

Pairs with a hard `MotorFaultError` at `max_temp`: derating buys a soft ramp
before the hard cutoff.

### `ThermalDerate(start_C, end_C)`

| Parameter | Type | Description |
|---|---|---|
| `start_C` | `float` | Temperature at which derating begins. |
| `end_C` | `float` | Temperature at which output reaches zero. Must be `> start_C`. |

Raises `ValueError` if `end_C <= start_C`.

### Members

| Member | Description |
|---|---|
| `derate(temp_C)` (call) | Return scale factor `0.0`–`1.0` for the given temperature. |
| `derate.start_C` | Read-only start threshold. |
| `derate.end_C` | Read-only end threshold. |

### Example

```python
from motortools import ThermalDerate

derate = ThermalDerate(start_C=70, end_C=80)

for t in loop:
    scale = derate(motor.state.temp)      # 0.0 – 1.0
    motor.cmd.torque = target_tau * scale
    motor.update(t)
```
