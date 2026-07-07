# CAN bus tools

Frame inspection, live sniffing, and per-motor TX/RX capture. All three
are re-exported at the top level of `motortools`.

```python
from motortools import inspect, sniff, peek
```

---

## `inspect(msg, params=None, direction='auto')`

Format a single CAN frame as a human-readable multi-line string. Returns
the string — does not print.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `msg` | `can.Message \| bytes \| bytearray` | — | The frame. Raw bytes are treated as ID `0`. |
| `params` | `MotorParams \| None` | `None` | When provided, the frame is also decoded as MIT CAN (position / velocity / torque / kp / kd). |
| `direction` | `str` | `'auto'` | `'rx'` decodes only as a motor response; `'tx'` only as a host command; `'auto'` shows both, each tagged `(if RX response)` / `(if TX command)`. |

Recognises the special `FF FF FF FF FF FF FF FC/FD/FE` enable / disable /
zero frames and labels them `MIT: ENABLE` etc. instead of trying to
decode them as impedance commands.

```python
from tmotorcan import load_params
from motortools import inspect

p = load_params('AK45-10')
print(inspect(b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFC', p))
```

---

## `sniff(bus, duration=None, count=None, params=None, direction='rx')`

Print frames from a python-can bus to stdout in real time. Blocks until
the first stop condition is met; Ctrl+C is caught silently.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bus` | `can.BusABC` | — | Any python-can bus instance. |
| `duration` | `float \| None` | `None` | Stop after this many seconds. `None` = no time limit. |
| `count` | `int \| None` | `None` | Stop after this many frames. `None` = unlimited. |
| `params` | `MotorParams \| None` | `None` | Forwarded to `inspect()` for MIT decode. |
| `direction` | `str` | `'rx'` | Forwarded to `inspect()`. Set `'auto'` if your driver loops back TX frames. |

```python
import can
from motortools import sniff

with can.Bus(interface='socketcan', channel='can0') as bus:
    sniff(bus, duration=10.0)
```

---

## `peek(motor, direction='tx', hex_only=False)`

Return the last TX or RX frame for a `MITMotor` as a formatted string.
First call attaches a transparent capture hook to `motor._bus`; subsequent
calls (for any motor on the same bus) reuse it. Multiple motors are
captured independently by id.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `motor` | `MITMotor` | — | Live motor instance (needs `._bus`, `.id`, `.params`). |
| `direction` | `str` | `'tx'` | `'tx'` = last command sent; `'rx'` = last reply received. |
| `hex_only` | `bool` | `False` | `True` returns bare hex like `"FF FF FF FF FF FF FF FC"`; `False` returns full `inspect()` output. |

Returns `'(no frame yet)'` if nothing has been captured for that motor /
direction. Raises `ValueError` for any other `direction` string.

```python
from motortools import peek

motor.update(t)
print(peek(motor, 'tx'))                    # full decode
print(peek(motor, 'rx', hex_only=True))     # raw 8-byte hex
```

---

## End-to-end example

```python
import can
from tmotorcan import MotorBus, MITMotor
from motortools import peek
from motortools.virtual import serve_in_thread

serve_in_thread(motor_id=1, channel='virt')
raw = can.interface.Bus(interface='virtual', channel='virt',
                        bitrate=1_000_000, receive_own_messages=False)

with MotorBus(raw) as bus, MITMotor(bus, motor_id=1) as m:
    m.cmd.position = 0.5
    m.update(0.0)
    print(peek(m, 'tx'))
    print(peek(m, 'rx'))
```
