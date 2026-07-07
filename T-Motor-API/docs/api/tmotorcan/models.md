# load_params

```python
from tmotorcan import load_params
```

Load a `MotorParams` from a built-in model name or a custom YAML file. Used
internally by `MITMotor.__init__`; useful directly when you need the encoding
ranges or torque constant without instantiating a motor.

---

## `load_params(model)`

| Parameter | Type | Description |
|---|---|---|
| `model` | `str` | Built-in model name (e.g. `'AK45-10'`) or a path ending in `.yaml`/`.yml`. |

Returns a frozen `MotorParams` (see [dataclasses.md](dataclasses.md#motorparams)).
Built-in models ship inside the `tmotorcan.models` package and are read via
`importlib.resources`, so they work whether you installed the package or run
it from `src/`.

| Raises | When |
|---|---|
| `FileNotFoundError` | Built-in name not found in the package. |
| `KeyError` | YAML is missing a required field. |
| `ValueError` | Field is nonsensical (e.g. `p_min >= p_max`, `kt <= 0`, `gear_ratio <= 0`). |

```python
from tmotorcan import load_params

p = load_params('AK45-10')
print(p.p_min, p.p_max)       # -12.5, 12.5
print(p.kt * p.gear_ratio)    # output-shaft Kt

# custom motor file
p = load_params('/path/to/my-motor.yaml')
```

---

## Built-in models

Located in `src/tmotorcan/models/`:

| Name | File |
|---|---|
| `AK45-10` | `ak45-10.yaml` |
| `AK45-36` | `ak45-36.yaml` |

Names are case-insensitive and spaces become hyphens (`'ak45 10'` resolves to
`ak45-10.yaml`).

---

## Custom YAML format

A custom model needs every field of `MotorParams`. Validation runs on load,
so any nonsensical combination is caught up-front rather than at the first
`update()`. Copy `ak45-10.yaml` as a starting point:

```yaml
# CAN frame encoding ranges (must match firmware)
p_min: -12.5
p_max:  12.5
v_min: -20.0
v_max:  20.0
t_min:  -8.0
t_max:   8.0
kp_min:  0.0
kp_max: 500.0
kd_min:  0.0
kd_max:  5.0

# Motor physical constants
gear_ratio: 10.0
kt:          0.127
pole_pairs: 14

# Safety
max_temp: 80
```

Reference it by stem name afterwards:
`MITMotor(bus, motor_id=1, model='MyMotor')`.
