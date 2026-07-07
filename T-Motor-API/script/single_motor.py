import msvcrt
import sys
import time
from pathlib import Path

import can

# make imports work regardless of which directory we're launched from
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))                          # tmotorcan / motortools

from tmotorcan import (
    MotorBus, MITMotor,
    MotorFaultError, MotorTimeoutError,
    RealtimeLoop, torque_to_current
)

# Bus selection
# Real hardware:
# raw_bus = can.interface.Bus(
#     interface='seeedstudio',
#     channel='COM10',
#     bitrate=1_000_000,
# )

# Virtual (no hardware): start a VirtualMotor in-thread, then open the bus
# on the SAME channel. python-can's 'virtual' interface only bridges buses
# inside one Python process, so the sim MUST run in this same process.

# from motortools.virtual import serve_in_thread
# _sim_stop = serve_in_thread(motor_id=1, channel='virt')

# vir_bus = can.interface.Bus(
#     interface='virtual',
#     channel='virt',
#     bitrate=1_000_000,
# )

# raw_bus = can.interface.Bus(
#     interface='seeedstudio',
#     baudrate=2_000_000,
#     channel='COM17',
#     bitrate=1_000_000,
#     frame_type='STD'
# )

raw_bus = can.interface.Bus(
    interface='slcan',
    # baudrate=2_000_000,
    channel='COM18',
    bitrate=1_000_000,
    frame_type='STD'
)


def _hex(data: bytes) -> str:
    return ' '.join(f'{b:02X}' for b in data) if data else '--'


class KeyboardLine:

    def __init__(self):
        self._buf = []

    def poll(self):
        line = None
        while msvcrt.kbhit():                 # is a key waiting? -> read it, else return
            ch = msvcrt.getwch()
            if ch in ('\x00', '\xe0'):        # arrow / function key: swallow its 2nd code
                msvcrt.getwch()
                continue
            if ch in ('\r', '\n'):            # Enter -> line complete
                line = ''.join(self._buf)
                self._buf.clear()
                print()                       # move to a fresh line after the input
            elif ch == '\x08':                # Backspace
                if self._buf:
                    self._buf.pop()
            elif ch == '\x03':                # Ctrl+C
                raise KeyboardInterrupt
            else:
                self._buf.append(ch)
        return line


def main():
    bus    = MotorBus(raw_bus)
    motor1 = MITMotor(bus, motor_id=3, model='AK45-10', max_temp=80)
    # torque_to_current(tor, motor1.kt_out) →  A    (import from motortools)
    # current_to_torque(cur, motor1.kt_out) →  N·m

    # ── Frame capture: wrap bus.send / bus.recv to record last TX/RX bytes ──
    last_tx: list[bytes] = [b'']
    last_rx: list[bytes] = [b'']

    _orig_send = bus.send
    _orig_recv = bus.recv

    def _traced_send(motor_id, data):
        last_tx[0] = bytes(data)
        return _orig_send(motor_id, data)

    def _traced_recv(motor_id, timeout=1.0):
        msg = _orig_recv(motor_id, timeout=timeout)
        if msg is not None:
            last_rx[0] = bytes(msg.data)
        return msg

    bus.send = _traced_send
    bus.recv = _traced_recv

    try:
        bus.drain()
        time.sleep(1.5)   # wait for hardware to finish booting

        motor1.enable()
        print(motor1.enable())
        print(motor1.state)
        motor1.coast()
        motor1.zero()
        input("Press Enter to start the control loop...")  # wait for user before ramping up
        print("Ready. Ctrl+C to stop.\n")

        motor1.cmd.kp = 0
        motor1.cmd.kd = 2.1

        target = 0.0                 # commanded position (rad); stays put until you type a new one
        kb     = KeyboardLine()      # non-blocking keyboard reader
        print("Type a target position (rad) then Enter to move. Ctrl+C to stop.\n")

        loop = RealtimeLoop(dt=0.05, report=True)
        try:
            for t in loop:
                # --- non-blocking input: returns a line only when Enter is pressed ---
                line = kb.poll()
                if line is not None:
                    line = line.strip()
                    if line:
                        try:
                            target = float(line)
                            print(f"[set] target = {target:.3f} rad")
                        except ValueError:
                            print(f"[ignored] not a number: {line!r}")

                motor1.cmd.torque = target
                motor1.update(t)     # runs every tick regardless of input -> bus never starves

                print(f"{t:.3f}  "
                      f"target={target:.3f}  "
                      f"pos={motor1.state.position:.3f}  "
                      f"vel={motor1.state.velocity:.3f}  "
                      f"tor={motor1.state.torque:.3f}  "
                      f"cur={torque_to_current(motor1.state.torque, motor1.kt_out):.3f}  "
                      f"temp={motor1.state.temp:.1f}°C  "
                      f"err={motor1.state.error}  "
                      f"TX={_hex(last_tx[0])}  RX={_hex(last_rx[0])}")

            motor1.coast()

        except (MotorFaultError, MotorTimeoutError) as e:
            print(f"\n[STOP] {e}")

    finally:
        motor1.close()
        bus.close()


if __name__ == '__main__':
    main()


# MULTI-MOTOR (same loop, same bus)
#
#   from tmotorcan import MotorBus, MITMotor, update_all, RealtimeLoop
#
#   with MotorBus(raw) as bus:
#       bus.drain()
#       time.sleep(1.5)
#       with MITMotor(bus, motor_id=1) as m1, MITMotor(bus, motor_id=2) as m2:
#           m1.zero(); m2.zero()
#           loop = RealtimeLoop(dt=0.05, report=True, fade=0.5)
#           for t in loop:
#               m1.cmd.position = 0.0
#               m2.cmd.position = m1.state.position   # mirror motor 1
#               update_all([m1, m2], t)               # TX both → RX both
#               print(m1.state.position, m2.state.position)
#
# update_all sends all command frames first then collects all replies,
# keeping both motors' setpoints within ~100 µs of each other.
# Sequential update() calls would stagger by a full round-trip (~1–5 ms).

# MULTITHREADING (one thread per motor)
#
# import threading
#
#   def run_motor(bus, motor_id, results, stop_event):
#       with MITMotor(bus, motor_id=motor_id) as m:
#           m.zero()
#           while not stop_event.is_set():
#               m.cmd.kp = 2.0; m.cmd.kd = 0.1
#               m.update()
#               results[motor_id] = m.state.as_tuple()
#
#   with MotorBus(raw) as bus:
#       bus.drain(); time.sleep(1.5)
#       results    = {}
#       stop_event = threading.Event()
#       threads = [
#           threading.Thread(target=run_motor, args=(bus, mid, results, stop_event))
#           for mid in [1, 2]
#       ]
#       for th in threads: th.start()
#       time.sleep(5)
#       stop_event.set()
#       for th in threads: th.join()
#
# MotorBus is thread-safe: the TX lock prevents frame interleaving and the
# RX thread routes replies to per-motor queues.

# MULTIPROCESSING
#
#   from multiprocessing import Process, Queue
#   from tmotorcan import MotorBus, MITMotor, MotorState
#
#   def motor_proc(motor_id, out_q, com_port='COM10'):
#       # Each process must open its own MotorBus — it cannot be shared/pickled.
#       with MotorBus(raw) as bus:
#           bus.drain(); time.sleep(1.5)
#           with MITMotor(bus, motor_id=motor_id) as m:
#               m.zero()
#               for _ in range(100):
#                   m.cmd.kp = 2.0; m.cmd.kd = 0.1
#                   m.update()
#                   out_q.put((motor_id, m.state.as_tuple()))
#
#   if __name__ == '__main__':
#       q = Queue()
#       procs = [Process(target=motor_proc, args=(mid, q)) for mid in [1, 2]]
#       for p in procs: p.start()
#       for p in procs: p.join()
#       while not q.empty():
#           mid, tup = q.get()
#           print(mid, MotorState.from_tuple(tup))
#
# Use MotorState.as_tuple() / from_tuple() to move state across process
# boundaries — MotorState itself is not picklable (dataclass with no special
# pickle support would work, but as_tuple keeps the pattern explicit).
