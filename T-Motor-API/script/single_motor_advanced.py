import sys
import time
import can

sys.path.insert(0, '../src')

from tmotorcan import (
    MotorBus, MITMotor,
    MotorFaultError, MotorTimeoutError,
    RealtimeLoop,
)
from motortools import (
    TrapezoidalProfile,
    rads_to_rpm, rads_to_dps, rads_to_erpm,
    torque_to_current,
)

raw_bus = can.interface.Bus(
    interface='seeedstudio',
    channel='COM10',
    bitrate=1_000_000,
)


def main():
    bus = MotorBus(raw_bus)

    # Wait for hardware to settle BEFORE enabling MIT mode so the motor's
    # watchdog doesn't fire during the sleep and drop it out of MIT mode.
    bus.drain()
    time.sleep(1.5)

    with MITMotor(bus, motor_id=1, model='AK45-10') as motor:

        # Optional features
        motor.set_vel_filter(0.2)    # IIR smoother on velocity (α=0.2, heavier = lower α)
        motor.set_accel_est(True)    # finite-difference acceleration from filtered velocity
        motor.set_multi_turn(True)   # unwrap position past the ±12.5 rad encoder limit

        # Logging
        motor.set_log(True, name='run1.csv')   # start CSV log immediately

        # Hardware init
        motor.zero()
        print("Ready. Running trapezoidal move 0 → π rad. Ctrl+C to stop.\n")

        # Trajectory
        profile = TrapezoidalProfile(v_max=1.0, a_max=2.0)
        profile.plan(q0=0.0, q1=3.14159)
        print(f"Move duration: {profile.duration:.2f} s\n")

        motor.cmd.kp = 5.0
        motor.cmd.kd = 0.1

        # Control loop
        loop = RealtimeLoop(dt=0.05, report=True, fade=0.5)
        try:
            for t in loop:
                pos_cmd, vel_cmd, _ = profile(t)
                motor.cmd.position = pos_cmd
                motor.cmd.velocity = vel_cmd
                motor.cmd.torque   = 0.0 * loop.fade   # taper to zero on Ctrl+C

                motor.update(t)

                vel = motor.state.velocity
                tor = motor.state.torque
                current_A = torque_to_current(tor, motor.kt_out)

                print(
                    f"t={t:.2f}  "
                    f"pos={motor.state.position:+.3f} rad  "
                    f"vel={vel:+.2f} rad/s "
                    f"({rads_to_rpm(vel):.1f} RPM  "
                    f"{rads_to_dps(vel):.0f} DPS  "
                    f"{rads_to_erpm(vel, motor.params.pole_pairs):.0f} ERPM)  "
                    f"accel={motor.state.acceleration:+.2f} rad/s²  "
                    f"tor={tor:+.3f} N·m  "
                    f"I={current_A:+.2f} A  "
                    f"temp={motor.state.temp}°C"
                )

        except MotorFaultError as e:
            print(f"\n[FAULT]   motor_id={e.motor_id}  code={e.code}  {e}")
        except MotorTimeoutError as e:
            print(f"\n[TIMEOUT] motor_id={e.motor_id}  timeout={e.timeout}s")

    bus.close()


if __name__ == '__main__':
    main()


# CAN FRAME INSPECTION
#
# Decode live frames with MIT field values printed alongside hex/binary:
#
# from motortools import sniff
#   from tmotorcan import MotorParams, load_params
#
#   params = load_params('AK45-10')
#       sniff(bus, duration=10.0, params=params)
#
# Or inspect a single captured frame:
#
# from motortools import inspect
#   print(inspect(msg, params=params))
#
# DISABLE FEATURES AT RUNTIME
#
#   motor.set_vel_filter(None)   # back to raw encoder velocity
#   motor.set_accel_est(False)   # stop estimating acceleration
#   motor.set_multi_turn(False)  # back to raw ±12.5 rad range
#   motor.set_log(False)         # flush and close CSV
#
# MULTI-MOTOR (see single_motor.py for full patterns)
#
# from tmotorcan import update_all
#   update_all([m1, m2], t)   # TX all → RX all, keeps motors within ~100 µs
