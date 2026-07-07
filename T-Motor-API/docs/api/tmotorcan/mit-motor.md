# MITMotor

```python
from tmotorcan import MITMotor
```

Single AK-series motor driver in MIT CAN mode. Holds a `cmd: MotorSetpoint`
that you mutate freely, and a `state: MotorState` that is refreshed every
tick. All optional features (velocity filter, accel estimation, multi-turn
unwrap, soft limits, thermal derating, CSV logging) are **off by default**
and only add per-tick overhead while enabled — with nothing active, each
`update()` is a tight encode-send-recv-decode loop.

```python
with MITMotor(bus, motor_id=1, model='AK45-10') as m:
    m.zero()
    m.cmd.kp, m.cmd.kd = 2.0, 0.1
    for t in loop:
        m.cmd.position = 0.0
        m.update(t)
        print(m.state.position, m.state.velocity)
```

---

## Constructor

### `MITMotor(bus, motor_id, model='AK45-10', max_temp=None, log=False)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bus` | `MotorBus` | — | Open bus the motor lives on. Registered on construction; `ValueError` if the id is already taken. |
| `motor_id` | `int` | — | CAN arbitration ID (must match firmware). |
| `model` | `str` | `'AK45-10'` | Built-in model name **or** path to a custom `.yaml` file. |
| `max_temp` | `int \| None` | `None` | Override the YAML temperature limit (°C). `None` = use the YAML default. |
| `log` | `bool \| str` | `False` | `True` opens an auto-named CSV. A string is treated as the file path. `False` disables logging. |

`__init__` does not talk to the motor — it only registers an RX queue and
loads the model file. `__enter__` calls `enable()` and **raises
`MotorTimeoutError` if the motor doesn't reply** to the 0xFC enable frame.
Don't call `enable()` again inside the `with` block.

### Attributes

| Name | Type | Description |
|---|---|---|
| `id` | `int` | CAN id this motor was constructed with. |
| `cmd` | `MotorSetpoint` | Mutate freely; sent on each `update()`. |
| `state` | `MotorState` | Updated each tick; treat as read-only. |
| `params` | `MotorParams` | Loaded from the model YAML. |
| `kt_out` | `float` | Output-shaft torque constant (N·m/A) — `params.kt × params.gear_ratio`. Read-only. |
| `limited` | `set[str]` | Field names clipped on the most recent send. Empty when limits are off or all values are inside their ranges. |
| `log` | `bool` | `True` while CSV logging is active. |

---

## Core control

### `update(t=0.0, timeout=1.0)`

Send `motor.cmd` → wait for the reply → update `motor.state`. Call this every
tick at ≥20 Hz. Raises `MotorTimeoutError` on no reply, `MotorFaultError` on
hardware fault or overtemperature.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `t` | `float` | `0.0` | Timestamp written to the CSV log. |
| `timeout` | `float` | `1.0` | Seconds to wait for the reply. |

```python
for t in RealtimeLoop(dt=0.02):
    m.cmd.position = 0.5
    m.update(t)
```

### `send_cmd()` / `recv_state(timeout=1.0, t=0.0)`

The TX and RX halves of `update()`, exposed separately so `MotorGroup` /
`update_all` can broadcast every command before collecting any replies. You
rarely call these directly. `send_cmd()` runs the TX pipeline (soft limits,
thermal derating); `recv_state()` runs the RX pipeline (multi-turn, vel
filter, accel est), checks faults, writes the log row.

---

## Mode commands

### `enable()`

Enter MIT motor mode (sends `0xFC`). Called automatically by `__enter__`.
Returns `True` on reply, `False` on silence. The context-manager turns the
`False` case into a `MotorTimeoutError`.

### `disable()`

Exit MIT motor mode (sends `0xFD`), then `coast()`. Called automatically by
`__exit__`.

### `zero()`

Set the current shaft position as the zero reference (sends `0xFE`).

```python
m.zero()                # call once after homing
```

### `coast()`

Send a zero-torque, zero-gain command. Motor freewheels safely. Bypasses soft
limits and thermal derating, so an asymmetric torque limit cannot accidentally
command nonzero torque.

### `close()`

Disable the motor, close the CSV log, and unregister from the bus. Safe to call
multiple times. The context-manager form does this for you.

---

## Optional features

Each setter rebuilds the per-tick pipeline so disabled features add zero
overhead. All features below require the `[full]` extra
(`pip install tmotorcan[full]`); calling them on a core-only install raises
`ImportError` with a hint.

### `set_vel_filter(alpha)`

First-order IIR low-pass on the velocity feedback.

| `alpha` | Effect |
|---|---|
| near `0.0` | heavy smoothing |
| `0.5` | moderate smoothing |
| `1.0` | passthrough (no filtering) |
| `None` | disable; `state.velocity` is the raw encoder reading |

`alpha` must be in `(0.0, 1.0]`. Retuning `alpha` while enabled preserves the
filter's accumulated state — no glitch at the retune instant.

### `set_accel_est(enabled)`

Toggle finite-difference acceleration estimation (operates on the filtered
velocity). When disabled, `state.acceleration` stays at `0.0`.

### `set_multi_turn(enabled)`

Toggle multi-turn position unwrapping. Removes the encoder's ±12.5 rad limit by
detecting wraparound and accumulating an offset into `state.position`. Resets
the offset when called — call before motion starts.

```python
m.set_multi_turn(True)
m.zero()
# state.position is now continuous across full revolutions
```

---

## Safety envelopes

### `set_limits(field, lo_hi)`

Clamp a `cmd` field to `[lo, hi]` at send time. `field` is one of `'position'`,
`'velocity'`, `'torque'`. Pass `lo_hi=None` to remove that field's limit. The
TX pipeline step is lazy-created on the first call and dropped from the
pipeline when the last limit is removed.

```python
m.set_limits('position', (-1.0, 1.0))    # rad
m.set_limits('velocity', (-5.0, 5.0))    # rad/s
m.set_limits('torque',   (-3.0, 3.0))    # N·m
m.set_limits('position', None)           # remove
```

Clamping is silent — `motor.cmd` is not mutated, only the value sent on the
wire is capped. Inspect `motor.limited` (a `set[str]`) to see which fields were
clipped on the most recent send.

### `set_current_limit(amps)`

Cap phase current by converting to a symmetric torque limit via `kt_out`.
Equivalent to `set_limits('torque', (-amps*kt_out, amps*kt_out))`. Pass `None`
to remove the torque limit.

```python
m.set_current_limit(5.0)    # |cmd.torque| ≤ 5.0 A × kt_out
```

### `set_temp_derating(enabled, start_C=None, end_C=None)`

Soft thermal protection: scales `cmd.torque` linearly down to zero as
temperature rises from `start_C` to `end_C`. `cmd.kp` and `cmd.kd` are not
scaled (those are firmware-side gains, so impedance behavior continues
unchanged). The hard `MotorFaultError` cutoff at `max_temp` still fires.

| Parameter | Default if `None` |
|---|---|
| `start_C` | `max_temp - 10` |
| `end_C` | `max_temp` |

```python
m.set_temp_derating(True)              # default window: max_temp-10 → max_temp
m.set_temp_derating(True, 60, 75)      # custom window
m.set_temp_derating(False)             # disable
```

---

## Homing

### `home(direction=1.0, current_A=2.0, speed=0.5, timeout=10.0, dt=0.02, kd=0.5)`

Drive against a hard stop, detect by torque spike, then `zero()` the position.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `direction` | `float` | `1.0` | Sign sets drive direction; magnitude is ignored. |
| `current_A` | `float` | `2.0` | Stop detection threshold (A). Hit when `\|state.torque\| ≥ current_A × kt_out`. |
| `speed` | `float` | `0.5` | Drive velocity (rad/s). |
| `timeout` | `float` | `10.0` | Seconds before raising `TimeoutError` if no contact is detected. |
| `dt` | `float` | `0.02` | Inner update period. |
| `kd` | `float` | `0.5` | Velocity damping during the drive. |

Motor must already be enabled. A 300 ms settling window is skipped first so
startup torque noise does not trip the detector. The previous `cmd` is saved
and restored on success or failure — homing does not leak the drive setpoint
into your control loop.

```python
m.home(direction=-1, current_A=3.0)
```

---

## CSV logging

### `set_log(enabled, name=None)`

Toggle CSV logging at any time.

```python
m.set_log(True)               # auto: motor1_YYYYMMDD_HHMMSS.csv
m.set_log(True, name='r.csv') # exact path
m.set_log(False)              # flush and stop
```

Columns: `time_s, pos_rad, vel_rad_s, accel_rad_s2, torque_Nm, temp_C, error`.

Read `m.log` (bool property) to check whether logging is currently active.
Logging requires the `[full]` extra.
