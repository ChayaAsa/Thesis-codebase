# MotorGroup, update_all

```python
from tmotorcan import MotorGroup, update_all
# also available directly:
from motortools import MotorGroup, update_all
```

Multi-motor coordination helpers. `MotorGroup` is an iterable collection with
broadcast setters and a synchronised TX-all-then-RX-all update.
`update_all` is the standalone function form.

Anything motor-like that exposes `.id`, `.send_cmd()`, and
`.recv_state(timeout, t)` works — different motor types can share one group.

---

## `update_all(motors, t=0.0, timeout=1.0)`

Synchronised update: TX every motor, then RX every motor. Sequential
`motor.update()` staggers commands by a full CAN round-trip (~1–5 ms per motor);
`update_all` collapses that so all motors receive their setpoints within ~100 µs
of each other at 1 Mbit/s.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `motors` | iterable | — | Motor objects (duck-typed — `.send_cmd()` + `.recv_state(timeout, t)`). |
| `t` | `float` | `0.0` | Timestamp written to each motor's CSV log. |
| `timeout` | `float` | `1.0` | Seconds to wait for each motor's reply. |

Raises whatever the motor's `recv_state` raises — typically `MotorTimeoutError`
or `MotorFaultError` from `tmotorcan`.

```python
from tmotorcan import update_all

for t in loop:
    update_all([m1, m2, m3], t=t)
```

---

## `MotorGroup`

### `MotorGroup(motors)`

| Parameter | Type | Description |
|---|---|---|
| `motors` | iterable of motor objects | Stored as a list; order is preserved for indexing and iteration. |

### Synchronised I/O

| Member | Description |
|---|---|
| `update_all(t=0.0, timeout=1.0)` | TX every motor, then RX every reply. Same as `update_all(group, …)`. |
| `send_cmd()` | TX every motor's setpoint (no receive). |
| `recv_state(timeout=1.0, t=0.0)` | RX one reply per motor and process it. |

### Broadcast helpers

Each forwards `*args, **kwargs` to motors that implement the named method.
Motors without the method are silently skipped.

| Member | Forwards to |
|---|---|
| `enable()` / `disable()` / `zero()` / `coast()` / `close()` | Lifecycle commands. |
| `set_log(...)` | Toggle per-motor CSV logging. |
| `set_vel_filter(...)` | Apply velocity IIR. |
| `set_multi_turn(...)` | Enable multi-turn position unwrap. |
| `set_accel_est(...)` | Enable acceleration estimator. |
| `set_limits(...)` | Per-field soft limits. |
| `set_current_limit(...)` | Phase-current cap. |
| `set_temp_derating(...)` | Thermal derate curve. |

### Container protocol

| Member | Description |
|---|---|
| `for m in group` | Iterate over the underlying motors. |
| `len(group)` | Number of motors. |
| `group[i]` | Index access. |

### Example

```python
from tmotorcan import MotorBus, MITMotor, MotorGroup, RealtimeLoop

with MotorBus(raw) as bus:
    m1 = MITMotor(bus, motor_id=1, model='AK45-10')
    m2 = MITMotor(bus, motor_id=2, model='AK45-10')

    group = MotorGroup([m1, m2])
    group.enable()
    group.set_vel_filter(0.2)

    loop = RealtimeLoop(dt=0.02, fade=0.5)
    try:
        for t in loop:
            for m in group:
                m.cmd.position = 0.5 * loop.fade
            group.update_all(t)
    finally:
        group.disable()
        group.close()
```
