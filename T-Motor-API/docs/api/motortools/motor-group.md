# MotorGroup, update_all

```python
from tmotorcan import MotorGroup, update_all
```

Coordinate multiple motors with a single broadcast call and a synchronised
update. Sequential `motor.update()` calls stagger commands by a full CAN
round-trip per motor (~1–5 ms); `update_all` sends every TX frame first and
then collects replies, so all motors receive their setpoints within ~100 µs of
each other on a 1 Mbit/s bus.

---

## `update_all(motors, t=0.0, timeout=1.0)`

Standalone form: TX every motor, then RX every motor.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `motors` | iterable of motor-like objects | — | Each must implement `send_cmd()` and `recv_state(timeout, t)`. |
| `t` | `float` | `0.0` | Timestamp written to each motor's CSV log. |
| `timeout` | `float` | `1.0` | Per-motor reply timeout. |

```python
from tmotorcan import MotorBus, MITMotor, update_all

with MotorBus(raw) as bus:
    m1 = MITMotor(bus, motor_id=1)
    m2 = MITMotor(bus, motor_id=2)
    m1.enable(); m2.enable()
    for t in loop:
        update_all([m1, m2], t=t)
```

---

## MotorGroup

Iterable container with the same synced update plus broadcast helpers.

### `MotorGroup(motors)`

| Parameter | Type | Description |
|---|---|---|
| `motors` | iterable of motor-like objects | Stored in order; you can mix types as long as they share the duck-type contract. |

### Container protocol

```python
group = MotorGroup([m1, m2, m3])
len(group)                     # 3
group[0] is m1                 # True
for m in group:                # iterable
    print(m.id)
```

### Synchronised I/O

| Method | Description |
|---|---|
| `update_all(t=0.0, timeout=1.0)` | TX all → RX all. Same as `update_all(group, …)`. |
| `send_cmd()` | TX every motor's setpoint (no receive). |
| `recv_state(timeout=1.0, t=0.0)` | RX one reply per motor and process it. |

### Broadcast helpers

Every method below is forwarded to each motor that implements it; motors
without the method are silently skipped.

| Method | Forwards to |
|---|---|
| `enable()` / `disable()` | `motor.enable / disable` |
| `zero()` / `coast()` / `close()` | `motor.zero / coast / close` |
| `set_log(*a, **kw)` | `motor.set_log` |
| `set_vel_filter(*a, **kw)` | `motor.set_vel_filter` |
| `set_multi_turn(*a, **kw)` | `motor.set_multi_turn` |
| `set_accel_est(*a, **kw)` | `motor.set_accel_est` |
| `set_limits(*a, **kw)` | `motor.set_limits` |
| `set_current_limit(*a, **kw)` | `motor.set_current_limit` |
| `set_temp_derating(*a, **kw)` | `motor.set_temp_derating` |

```python
group = MotorGroup([m1, m2])
group.set_vel_filter(0.2)              # both motors
group.set_limits('torque', (-3, 3))    # both motors
group.zero()                            # both motors
for t in loop:
    group.update_all(t)
```

### Duck-type contract

Anything with the following attributes can live in a `MotorGroup`:

- `id` — used only in error messages
- `send_cmd()` — push the setpoint
- `recv_state(timeout, t)` — pull one reply and process it

Other methods (`set_vel_filter`, `zero`, …) are best-effort: if the object
defines them, they get called; if not, the broadcast skips them.
