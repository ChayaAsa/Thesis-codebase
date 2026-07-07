from __future__ import annotations
from tmotorcan import MotorBus, MITMotor

try:
    import msvcrt
except ImportError:
    msvcrt = None


# Motor command helpers

def cmd_free(motors: list, free_kd: dict) -> None:
    for m in motors:
        m.cmd.kp       = 0.0
        m.cmd.kd       = free_kd[m.id]
        m.cmd.position = 0.0
        m.cmd.velocity = 0.0
        m.cmd.torque   = 0.0
        m.update()


def cmd_lock(motors: list, lock_kp: dict, lock_kd: dict) -> None:
    for m in motors:
        m.cmd.kp       = lock_kp[m.id]
        m.cmd.kd       = lock_kd[m.id]
        m.cmd.position = 0.0
        m.cmd.velocity = 0.0
        m.cmd.torque   = 0.0
        m.update()


# Hardware constructors

def make_bus(port: str, bitrate: int):
    import can
    from tmotorcan import MotorBus
    raw = can.interface.Bus(
        interface='slcan',
        channel=port,
        bitrate=bitrate,
        frame_type='STD',
    )
    return MotorBus(raw)


def build_motors(bus, motor_ids: list, model: str, max_temp: int) -> list[MITMotor]:
    from tmotorcan import MITMotor
    return [MITMotor(bus, motor_id=mid, model=model, max_temp=max_temp)
            for mid in motor_ids]


# Math utilities

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def motor_limits(sign: dict, id: int, lo: float, hi: float) -> tuple[float, float]:
    a, b = sign[id] * lo, sign[id] * hi
    return (a, b) if a <= b else (b, a)


# Non-blocking keyboard input (Windows)

class KeyboardLine:

    def __init__(self):
        self._buf = []

    def poll(self) -> str | None:
        if msvcrt is None:
            return None
        line = None
        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ('\x00', '\xe0'):        # arrow / function key: swallow 2nd code
                msvcrt.getwch()
                continue
            if ch in ('\r', '\n'):
                line = ''.join(self._buf)
                self._buf.clear()
                print()
            elif ch == '\x08':                # Backspace
                if self._buf:
                    self._buf.pop()
            elif ch == '\x03':                # Ctrl+C
                raise KeyboardInterrupt
            else:
                self._buf.append(ch)
        return line
