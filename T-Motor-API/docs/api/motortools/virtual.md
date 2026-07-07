# Virtual motor

Software-only AK-series motor that responds to MIT CAN commands on a
python-can `virtual` bus. Use it to develop and test against the
`tmotorcan` API without hardware. Python equivalent of the
`tool/Motor-Analyzer-MCU/AK45_Simulator.ino` firmware.

```python
from motortools.virtual import VirtualMotor, serve_in_thread, SimHandle
```

> python-can's `virtual` interface is **same-process only**. Two separate
> `python …` terminals get isolated buses and never see each other — run
> the simulator in-process from your control script via `serve_in_thread()`.

The standalone CLI is only useful for harnesses that import
`VirtualMotor` directly from the same Python process:

```bash
python -m motortools.virtual --motor-id 1 --verbose
```

Dynamics (matches the Arduino simulator):

```
torque = kp × (p_des − pos) + kd × (v_des − vel) + torque_ff
accel  = (torque − damping × vel) / inertia
vel   += accel × dt
pos   += vel × dt
```

---

## `VirtualMotor(params, motor_id, inertia=0.08, damping=0.15)`

Standalone simulator (no bus). Fed one frame at a time via
`handle_frame()`. Most users want `serve_in_thread()` instead.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `params` | `MotorParams` | — | Motor model constants (use `load_params('AK45-10')`). |
| `motor_id` | `int` | — | CAN arbitration ID this sim should answer to. |
| `inertia` | `float` | `0.08` | Reflected output-shaft inertia in kg·m². |
| `damping` | `float` | `0.15` | Viscous damping in N·m·s/rad. |

### Live state attributes

| Attribute | Type | Description |
|---|---|---|
| `pos` | `float` | Current position (rad). |
| `vel` | `float` | Current velocity (rad/s). |
| `torque` | `float` | Last applied torque (N·m). |
| `temp` | `int` | Reported temperature (°C, default 25). |
| `error` | `int` | Reported error code. |
| `enabled` | `bool` | Mirrors host enable/disable. |

### Methods

| Member | Description |
|---|---|
| `handle_frame(data: bytes) -> bytes \| None` | Process one 8-byte MIT command frame. Returns reply bytes, or `None` if `len(data) != 8`. Recognises the `FC` enable, `FD` disable, and `FE` zero special frames. |

---

## `serve_in_thread(...)`

```python
serve_in_thread(motor_id=1, channel='virt', model='AK45-10',
                inertia=0.08, damping=0.15, verbose=False) -> SimHandle
```

Start a `VirtualMotor` running on a daemon thread that listens on a
python-can `virtual` bus. The sim-side bus is created inside this call;
your control script must open its own bus on the **same channel name in
the same Python process**.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `motor_id` | `int` | `1` | Arbitration ID the sim answers to. |
| `channel` | `str` | `'virt'` | Virtual channel name. Match in your control bus. |
| `model` | `str` | `'AK45-10'` | Built-in model name or path to a YAML. |
| `inertia` | `float` | `0.08` | Reflected inertia (kg·m²). |
| `damping` | `float` | `0.15` | Viscous damping (N·m·s/rad). |
| `verbose` | `bool` | `False` | Print sim state on every command. |

Returns a `SimHandle`. Daemon thread exits automatically on program exit;
explicit shutdown is optional.

---

## `SimHandle`

Object returned by `serve_in_thread()`. Backward-compatible with the old
`threading.Event` return — `.set()` / `.is_set()` still stop the sim.

| Attribute / method | Description |
|---|---|
| `.motor` | The live `VirtualMotor`. Read `.pos` / `.vel` / `.torque` / `.enabled` from any thread (float reads are atomic in CPython). |
| `.params` | The `MotorParams` used by the sim. |
| `.thread` | The background `threading.Thread`. |
| `.stop` | The underlying `threading.Event`. |
| `.set()` | Signal the sim to stop. |
| `.is_set()` | `True` once stop has been signalled. |

---

## Example

```python
import can
from tmotorcan import MotorBus, MITMotor, RealtimeLoop
from motortools.virtual import serve_in_thread

sim = serve_in_thread(motor_id=1, channel='virt')

raw = can.interface.Bus(interface='virtual', channel='virt',
                        bitrate=1_000_000, receive_own_messages=False)

with MotorBus(raw) as bus, MITMotor(bus, motor_id=1) as m:
    m.cmd.kp, m.cmd.kd = 20.0, 0.5
    for t in RealtimeLoop(dt=0.01):
        m.cmd.position = 0.5
        m.update(t)
        if t > 1.0:
            break
    print(f"sim pos = {sim.motor.pos:+.3f} rad")

sim.set()    # optional — daemon thread exits with the program
```
