# MotorBus

```python
from tmotorcan import MotorBus
```

Wraps a `python-can` `BusABC` and routes incoming frames into per-motor receive
queues via a daemon RX thread. Thread-safe (one bus shared across many threads
is fine), but **not** process-safe — create a fresh `MotorBus` inside each
subprocess.

---

## `MotorBus(bus)`

Construct from any `python-can` interface (seeedstudio, socketcan, virtual,
pcan, …). The RX thread starts immediately and prints the bus channel info.

| Parameter | Type | Description |
|---|---|---|
| `bus` | `can.BusABC` | The raw python-can bus. `MotorBus` takes ownership and shuts it down on `close()`. |

```python
import can
from tmotorcan import MotorBus

raw = can.interface.Bus(interface='virtual', channel='virt', bitrate=1_000_000)
with MotorBus(raw) as bus:
    ...   # bus is closed automatically on exit
```

---

## `register(motor_id)`

Reserve an RX queue for a motor. Called automatically by `MITMotor.__init__`,
so you usually do not need to call this yourself.

| Parameter | Type | Description |
|---|---|---|
| `motor_id` | `int` | CAN arbitration ID. |

Raises `ValueError` if the id is already registered (catches the "two `MITMotor`
instances on the same id" mistake). Close the existing motor or call
`unregister()` first to recycle the id.

---

## `unregister(motor_id)`

Drop a motor's RX queue. Safe to call on an unknown id.

---

## `send(motor_id, data)`

Send a raw 8-byte CAN frame to the given arbitration id. TX is serialised by
an internal lock, so concurrent sends from multiple threads are safe. Used
internally by `MITMotor`; you rarely call this directly.

| Parameter | Type |
|---|---|
| `motor_id` | `int` |
| `data` | `bytes \| bytearray` |

---

## `recv(motor_id, timeout=1.0)`

Block until a frame for `motor_id` arrives, or until `timeout` seconds elapse.
Returns the `can.Message`, or `None` on timeout. Each motor queue holds at
most one frame (drop-oldest), so `recv()` always returns the newest reply.

| Parameter | Type | Default |
|---|---|---|
| `motor_id` | `int` | — |
| `timeout` | `float` | `1.0` |

---

## `drain()`

Discard any stale reply frames sitting in motor queues. Useful before a fresh
command sequence to make sure you read this round's reply, not the previous
round's.

```python
bus.drain()
m.update()
```

---

## `close()` / `shutdown()`

Stop the RX thread and shut down the underlying CAN bus. Safe to call multiple
times. `shutdown` is an alias for `close`.

Raises `RuntimeError` if the RX thread doesn't stop within 2 seconds — in that
case the underlying bus is **not** shut down (avoids a use-after-close race
inside `bus.recv()`). If you wrap `close()` in `try/except`, surface this
error rather than swallowing it.

The context-manager form (`with MotorBus(raw) as bus:`) calls this for you.
