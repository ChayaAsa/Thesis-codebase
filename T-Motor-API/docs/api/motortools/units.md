# Unit conversions

Standalone helpers for converting between motor units. `rad/s` is the
canonical velocity unit in this library; everything else is provided for
interfacing with datasheets, GUIs, and ESC firmware.

```python
# torque/current — re-exported via tmotorcan
from tmotorcan import torque_to_current, current_to_torque

# all other conversions — motortools only
from motortools import (
    rads_to_rpm, rpm_to_rads,
    rads_to_dps, dps_to_rads,
    rpm_to_dps,  dps_to_rpm,
    rads_to_erpm, erpm_to_rads,
    rpm_to_erpm,  erpm_to_rpm,
)
```

---

## Torque ↔ current

Either pass `kt_out` (output-shaft torque constant in N·m/A) directly, or
pass `kt` (rotor-side) plus `gear_ratio`. `kt_out = kt × gear_ratio`. The
motor object exposes `motor.kt_out` for convenience.

| Function | Signature | Formula |
|---|---|---|
| `torque_to_current` | `(torque_Nm, kt=None, gear_ratio=1.0, *, kt_out=None) -> float` | `I = τ / (kt × ratio)` or `I = τ / kt_out` |
| `current_to_torque` | `(current_A, kt=None, gear_ratio=1.0, *, kt_out=None) -> float` | `τ = I × kt × ratio` or `τ = I × kt_out` |

```python
torque_to_current(5.0, kt_out=1.27)             # 3.94 A
torque_to_current(5.0, kt=0.127, gear_ratio=10) # 3.94 A
current_to_torque(3.0, kt_out=motor.kt_out)     # using a live motor
```

Raises `ValueError` if neither `kt_out` nor `kt` is supplied.

---

## Mechanical velocity conversions

All take a single `float` and return a `float`. No motor parameters needed.

| Function | Conversion |
|---|---|
| `rads_to_rpm(rads)` | rad/s → mechanical RPM |
| `rpm_to_rads(rpm)` | mechanical RPM → rad/s |
| `rads_to_dps(rads)` | rad/s → degrees/second |
| `dps_to_rads(dps)` | degrees/second → rad/s |
| `rpm_to_dps(rpm)` | mechanical RPM → degrees/second |
| `dps_to_rpm(dps)` | degrees/second → mechanical RPM |

---

## Electrical velocity conversions (require `pole_pairs`)

ERPM = mechanical RPM × pole pairs. Used by VESC-style ESCs and some
firmware status frames. `pole_pairs` lives in the motor model YAML
(`MotorParams.pole_pairs`).

| Function | Conversion |
|---|---|
| `rads_to_erpm(rads, pole_pairs)` | rad/s → electrical RPM |
| `erpm_to_rads(erpm, pole_pairs)` | electrical RPM → rad/s |
| `rpm_to_erpm(rpm, pole_pairs)` | mechanical RPM → electrical RPM |
| `erpm_to_rpm(erpm, pole_pairs)` | electrical RPM → mechanical RPM |

---

## Example

```python
from tmotorcan import load_params, current_to_torque
from motortools import rads_to_rpm, rads_to_erpm

p = load_params('AK45-10')

vel_rads = 12.5
print(f"{rads_to_rpm(vel_rads):.1f} RPM")
print(f"{rads_to_erpm(vel_rads, p.pole_pairs):.0f} ERPM")

# convert phase current reading to N·m
tau = current_to_torque(2.5, kt=p.kt, gear_ratio=p.gear_ratio)
```
