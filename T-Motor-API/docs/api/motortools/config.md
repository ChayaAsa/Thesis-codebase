# MotorConfig

```python
from tmotorcan import MotorConfig
```

YAML-backed rig configuration: bus settings, motors with their gains and
limits, and loop parameters. Save once, reload from disk on every run.

---

## `MotorConfig(...)`

Dataclass with these fields (all keyword-only, all optional):

| Field | Type | Default | Description |
|---|---|---|---|
| `interface` | `str` | `'seeedstudio'` | python-can interface name. |
| `channel` | `str` | `'COM4'` | CAN channel string. |
| `bitrate` | `int` | `1_000_000` | CAN bit rate in bps. |
| `motors` | `list[dict]` | `[]` | One dict per motor — see schema below. |
| `dt` | `float` | `0.05` | `RealtimeLoop` period in seconds. |
| `fade` | `float` | `0.5` | Ctrl+C fade-out duration. |
| `report` | `bool` | `True` | Print loop timing stats on exit. |

### `motors` entry schema

| Key | Type | Default | Description |
|---|---|---|---|
| `id` | `int` | required | CAN arbitration id. |
| `model` | `str` | `'AK45-10'` | Built-in name or path to YAML. |
| `max_temp` | `int` | — | °C override. If omitted, falls back to the model YAML's `max_temp`. |
| `kp` | `float` | `0.0` | Initial `motor.cmd.kp`. |
| `kd` | `float` | `0.0` | Initial `motor.cmd.kd`. |
| `limits` | `dict` | `{}` | `{field: [lo, hi], …}`, applied via `motor.set_limits`. |

```python
cfg = MotorConfig(
    channel='COM4',
    motors=[
        {'id': 1, 'model': 'AK45-10', 'kp': 2.0, 'kd': 0.1,
         'limits': {'torque': [-5.0, 5.0]}},
    ],
)
```

---

## Methods

| Member | Description |
|---|---|
| `MotorConfig.load(path)` | Classmethod. Load from a YAML file. Missing keys fall back to dataclass defaults. |
| `cfg.save(path)` | Write to a YAML file. Creates parent directories. |
| `cfg.open_bus()` | Open and return the raw `python-can` `Bus` using the stored `interface`, `channel`, `bitrate`. Wrap in a `MotorBus` yourself. |
| `cfg.open_motors(bus)` | Instantiate one `MITMotor` per `motors` entry. Per-motor `kp`, `kd`, and `limits` are applied. Returns motors in YAML order. |
| `cfg.make_loop()` | Return a fresh `RealtimeLoop` with the stored `dt`, `fade`, `report`. |

---

## YAML schema

```yaml
bus:
  interface: seeedstudio
  channel:   COM4
  bitrate:   1000000
motors:
  - id: 1
    model: AK45-10
    # max_temp omitted → AK45-10.yaml's max_temp is used
    kp: 2.0
    kd: 0.1
    limits:
      torque: [-5.0, 5.0]
loop:
  dt:     0.05
  fade:   0.5
  report: true
```

---

## Example

```python
from tmotorcan import MotorBus, MotorConfig

cfg = MotorConfig.load('config/my_rig.yaml')

with MotorBus(cfg.open_bus()) as bus:
    motors = cfg.open_motors(bus)         # list[MITMotor]
    for m in motors:
        m.enable()

    loop = cfg.make_loop()
    m1   = motors[0]
    for t in loop:
        m1.cmd.position = 0.0
        m1.update(t)

    for m in motors:
        m.close()
```
