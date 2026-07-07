# Exceptions

```python
from tmotorcan import MotorFaultError, MotorTimeoutError
from tmotorcan.protocol import FAULT_MESSAGES
```

Both exceptions subclass `RuntimeError`. They surface from `MITMotor.update()`,
`recv_state()`, and `__enter__()` — you generally catch them around your
realtime loop.

---

## MotorFaultError

The motor reported a hardware fault code, or its temperature exceeded the
configured `max_temp`.

| Attribute | Type | Description |
|---|---|---|
| `motor_id` | `int` | CAN id of the offending motor. |
| `code` | `int` | Fault code from the motor firmware. `0` is reserved for the synthetic over-temperature path. |
| `args[0]` | `str` | Human-readable message including motor id, code, and detail. |

Raised inside `MITMotor.update()` whenever:

- `state.error != 0` — firmware-reported fault. Detail comes from `FAULT_MESSAGES`.
- `state.temp > max_temp` — soft cutoff. `code` is `0`; detail reads e.g. `"82°C > 80°C"`.

### Fault codes

From `tmotorcan.protocol.FAULT_MESSAGES`:

| Code | Meaning |
|---|---|
| 0 | No error |
| 1 | Over temperature |
| 2 | Over current |
| 3 | Over voltage |
| 4 | Under voltage |
| 5 | Encoder fault |
| 6 | Phase current unbalance |

```python
from tmotorcan import MITMotor, MotorFaultError

try:
    m.update()
except MotorFaultError as e:
    print(f"motor {e.motor_id} fault {e.code}: {e}")
    m.coast()
```

---

## MotorTimeoutError

No CAN reply was received within the timeout.

| Attribute | Type | Description |
|---|---|---|
| `motor_id` | `int` | CAN id that didn't reply. |
| `timeout` | `float` | The timeout (seconds) that elapsed. |

Raised in two places:

- `MITMotor.update()` / `recv_state()` — bus dropped a reply or the motor
  went silent mid-loop.
- `MITMotor.__enter__()` — the initial `enable()` (0xFC) frame got no
  answer. Raising here means the real cause surfaces immediately, instead
  of a confusing timeout one second later on the first `update()`.

Common causes: motor not powered, wrong CAN id, wrong bitrate, broken cable.

```python
from tmotorcan import MITMotor, MotorTimeoutError

try:
    m.update(timeout=0.1)
except MotorTimeoutError as e:
    print(f"motor {e.motor_id} silent for {e.timeout}s")
```

---

## FAULT_MESSAGES

The fault-code dictionary lives in `tmotorcan.protocol` and is the single
source of truth used by `MotorFaultError`'s detail string. Import it
directly if you want to render fault codes in a UI:

```python
from tmotorcan.protocol import FAULT_MESSAGES

print(FAULT_MESSAGES.get(m.state.error, 'Unknown'))
```
