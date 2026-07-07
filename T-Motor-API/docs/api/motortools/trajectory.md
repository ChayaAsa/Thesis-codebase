# TrapezoidalProfile

```python
from motortools import TrapezoidalProfile
```

Constant-acceleration trapezoidal velocity profile between two positions.
Accelerates at `a_max` up to `v_max`, cruises, then decelerates symmetrically.
If the move is too short to reach `v_max`, the profile collapses to a
triangle (peak velocity reduced, no cruise).

---

## `TrapezoidalProfile(v_max, a_max)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `v_max` | `float` | — | Maximum velocity in units/s. Must be `> 0`. |
| `a_max` | `float` | — | Maximum acceleration in units/s². Must be `> 0`. |

Out-of-range values raise `ValueError`. Units are arbitrary — pass rad/s
and rad/s² for joint-space, m/s and m/s² for cartesian, etc.

---

## Methods and properties

| Member | Description |
|---|---|
| `plan(q0, q1)` | Plan a move from `q0` to `q1`. Returns total `duration` (seconds). Safe to call mid-motion — replans immediately from the supplied `q0`. |
| `profile(t)` (call) | Evaluate at time `t` since plan start. Returns `(position, velocity, acceleration)`. Before `plan()` returns `(0.0, 0.0, 0.0)`. After `t >= duration`, returns `(q1, 0.0, 0.0)`. |
| `duration` (property → `float`) | Total move duration in seconds (= 2·t_accel + t_cruise). |

---

## Example

```python
from tmotorcan import RealtimeLoop, MotorBus, MITMotor
from motortools import TrapezoidalProfile
from motortools.virtual import serve_in_thread
import can

sim = serve_in_thread(motor_id=1, channel='virt')

profile = TrapezoidalProfile(v_max=2.0, a_max=5.0)
T = profile.plan(q0=0.0, q1=3.14)        # plan a π-rad move
print(f"duration = {T:.3f} s")

raw = can.interface.Bus(interface='virtual', channel='virt',
                        bitrate=1_000_000, receive_own_messages=False)
with MotorBus(raw) as bus, MITMotor(bus, motor_id=1) as m:
    m.cmd.kp, m.cmd.kd = 20.0, 0.5
    for t in RealtimeLoop(dt=0.01):
        pos, vel, _ = profile(t)
        m.cmd.position = pos
        m.cmd.velocity = vel
        m.update(t)
        if t >= profile.duration:
            break
```
