import msvcrt
import sys
import time

import can

from tmotorcan import (
    MotorBus, MITMotor,
    MotorFaultError, MotorTimeoutError,
    RealtimeLoop, update_all, torque_to_current,
)

# ── Bus selection (match your single_motor.py working config) ─────────────────
raw_bus = can.interface.Bus(
    interface='slcan',
    channel='COM18',
    bitrate=1_000_000,
    frame_type='STD',
)

MOTOR_IDS = [1, 2, 3]
HOLD_KP   = 200.0   # N·m/rad position stiffness while holding
HOLD_KD   = 1.0     # N·m·s/rad velocity damping while holding


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
    motors = [MITMotor(bus, motor_id=mid, model='AK45-10', max_temp=80)
              for mid in MOTOR_IDS]

    try:
        bus.drain()
        time.sleep(1.5)   # wait for hardware to finish booting

        for m in motors:
            m.enable()
            m.coast()
            m.zero()       # set the current shaft angle as each motor's zero reference

        input("Pose the arm by hand, then press Enter to START (Ctrl+C to quit)...")
        print("Started FREE. Enter = toggle FREE<->HOLD.  Ctrl+C = stop.\n")

        holding   = False         # start free so you can pose by hand
        kb        = KeyboardLine()
        next_print = 0.0          # throttle printing to ~10 Hz

        loop = RealtimeLoop(dt=0.02, report=True, fade=0.4)
        try:
            for t in loop:
                # --- Enter toggles FREE <-> HOLD (any Enter press, text ignored) ---
                if kb.poll() is not None:
                    holding = not holding
                    if holding:
                        # latch each motor's current position as its hold target
                        for m in motors:
                            m.cmd.position = m.state.position
                            m.cmd.velocity = 0.0
                            m.cmd.torque   = 0.0
                        print(">>> HOLD  (latched current pose, gravity comp active)")
                    else:
                        print(">>> FREE  (re-pose by hand)")

                # gains applied every tick; fade ramps them down on Ctrl+C for a soft release
                f = loop.fade
                for m in motors:
                    if holding:
                        m.cmd.kp = HOLD_KP * f
                        m.cmd.kd = HOLD_KD * f
                    else:
                        m.cmd.kp = 0.0
                        m.cmd.kd = 0.0
                    m.cmd.torque = 0.0

                update_all(motors, t)         # TX all 3, then RX all 3 (synced)

                if t >= next_print:
                    next_print = t + 0.1
                    mode = "HOLD" if holding else "FREE"
                    cols = "  ".join(
                        f"M{m.id}: pos={m.state.position:+.3f} "
                        f"cur={torque_to_current(m.state.torque, m.kt_out):+.3f}A "
                        f"tor={m.state.torque:+.3f}Nm "
                        f"{m.state.temp:.0f}C"
                        for m in motors
                    )
                    print(f"[{mode}] t={t:6.2f}  {cols}")

            for m in motors:
                m.coast()

        except (MotorFaultError, MotorTimeoutError) as e:
            print(f"\n[STOP] {e}")

    finally:
        for m in motors:
            m.close()
        bus.close()


if __name__ == '__main__':
    main()
