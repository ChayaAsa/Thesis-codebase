# RealtimeLoop

```python
from tmotorcan import RealtimeLoop
```

Fixed-rate iterator that yields elapsed time on every tick. Sleeps for most of
each interval, then busy-spins the last 200 µs to hit the deadline accurately.
Ctrl+C triggers a soft-stop with optional fade-out.

---

## `RealtimeLoop(dt=0.001, report=False, fade=0.0, fifo_priority=0)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dt` | `float` | `0.001` | Loop period in seconds (e.g. `0.05` → 20 Hz). |
| `report` | `bool` | `False` | Print timing statistics (mean / std-dev error) when the loop ends. |
| `fade` | `float` | `0.0` | Seconds of soft-stop after Ctrl+C. `loop.fade` ramps `1.0 → 0.0` over this window. `0.0` = immediate stop. |
| `fifo_priority` | `int` | `0` | Linux only — FIFO real-time scheduler priority (1–99). Requires root. Ignored on Windows. |

---

## Iteration

`RealtimeLoop` is its own iterator. Each `__next__` returns `t`, the seconds
elapsed since the loop started.

```python
loop = RealtimeLoop(dt=0.02, report=True, fade=0.5)
for t in loop:
    motor.cmd.position = 0.5
    motor.update(t)
```

---

## Methods and properties

| Member | Description |
|---|---|
| `fade` (property → `float`) | Scale factor for ramping actuator output during soft-stop. `1.0` during normal run, decreases linearly to `0.0` over the fade window. Multiply torque/velocity commands by this. A second Ctrl+C during fade triggers a hard stop. |
| `stop()` | Stop the loop on the next iteration from inside the loop body. |
| `elapsed()` | Seconds since the loop started. Returns `0.0` before the first iteration. |
| `close()` | Restore prior signal handlers. Call if you `break` out of the loop manually instead of letting it exhaust. |

---

## Example

```python
from tmotorcan import RealtimeLoop

loop = RealtimeLoop(dt=0.02, report=True, fade=0.5)
try:
    for t in loop:
        motor.cmd.torque = target_torque * loop.fade
        motor.update(t)
        if abs(motor.state.position - target) < 0.01:
            loop.stop()
finally:
    loop.close()
```

---

## Timing report

With `report=True`, the loop prints when it ends:

```
[RealtimeLoop] 500 cycles @ 20.0 Hz
  mean timing error : +0.012 ms
  std dev           : 0.084 ms
```

Mean error is signed (positive = ran late on average). Standard deviation
captures jitter — large values usually indicate the host is too busy or the
selected `dt` is below what the platform can hit.
