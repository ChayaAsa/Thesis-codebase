# Dataclasses — `MotorState`, `MotorSetpoint`, `MotorParams`

```python
from tmotorcan import MotorState, MotorSetpoint, MotorParams
```

All three are `@dataclass`es. `MotorState` and `MotorSetpoint` are mutable;
`MotorParams` is `frozen=True`. All are picklable, so they pass through
`multiprocessing.Queue` without issue.

---

## MotorState

Last received motor telemetry. Treat as **read-only** from user code —
`MITMotor.update()` overwrites it every tick.

| Field | Type | Unit | Description |
|---|---|---|---|
| `position` | `float` | rad | Unwrapped multi-revolution position when `set_multi_turn(True)`, otherwise raw encoder value. |
| `velocity` | `float` | rad/s | Filtered when `set_vel_filter(alpha)` is active, otherwise raw. |
| `acceleration` | `float` | rad/s² | Estimated via finite difference. Stays at `0.0` unless `set_accel_est(True)` is called. |
| `torque` | `float` | N·m | Output-shaft torque reported by the motor. |
| `temp` | `int` | °C | Motor case temperature. |
| `error` | `int` | — | Fault code. `0` = healthy. See `FAULT_MESSAGES` in `tmotorcan.protocol`. |

### Methods

```python
state.as_tuple()              # → (pos, vel, accel, torque, temp, error)
MotorState.from_tuple(t)      # round-trip from a tuple
```

Useful for serialising telemetry across process boundaries.

---

## MotorSetpoint

The desired motor state, sent on each `MITMotor.update()`. Mutate fields
directly — the same instance is reused every tick.

| Field | Type | Unit | Description |
|---|---|---|---|
| `position` | `float` | rad | Position target (used by impedance / full-state control). |
| `velocity` | `float` | rad/s | Velocity target (used by velocity / full-state control). |
| `torque` | `float` | N·m | Feedforward output torque. |
| `kp` | `float` | N·m/rad | Position stiffness. |
| `kd` | `float` | N·m·s/rad | Velocity damping. |

### MIT control modes by gain combination

| Mode | `kp` | `kd` | Use |
|---|---|---|---|
| Pure torque | `0` | `0` | Set `torque` directly; motor outputs that torque. |
| Velocity | `0` | `>0` | Set `velocity`; damping holds the speed. |
| Impedance | `>0` | `>0` | Set `position`; spring-damper around it. |
| Full-state | `>0` | `>0` | All five fields — feedforward + impedance. |

```python
m.cmd.position = 0.5
m.cmd.kp       = 2.0
m.cmd.kd       = 0.1
m.cmd.torque   = 0.0    # no feedforward
```

---

## MotorParams

Immutable motor model constants loaded from a YAML file. You normally read
this off `motor.params` rather than constructing it directly. See
[`load_params`](models.md) to load by name or file path.

| Field | Type | Unit | Description |
|---|---|---|---|
| `p_min`, `p_max` | `float` | rad | Position encoding range (16-bit). |
| `v_min`, `v_max` | `float` | rad/s | Velocity encoding range (12-bit). |
| `t_min`, `t_max` | `float` | N·m | Torque encoding range (12-bit). |
| `kp_min`, `kp_max` | `float` | N·m/rad | `kp` encoding range (12-bit). |
| `kd_min`, `kd_max` | `float` | N·m·s/rad | `kd` encoding range (12-bit). |
| `gear_ratio` | `float` | — | Reduction ratio (output = motor / gear_ratio). |
| `kt` | `float` | N·m/A | Motor-side torque constant. Output-shaft Kt is `kt × gear_ratio`. |
| `pole_pairs` | `int` | — | Electrical pole pairs (used by ERPM conversions). |
| `max_temp` | `int` | °C | Default `MotorFaultError` temperature limit. |

```python
print(m.params.gear_ratio)                  # 10.0
print(m.params.kt * m.params.gear_ratio)    # output-shaft Kt
print(m.kt_out)                             # same value, precomputed
```
