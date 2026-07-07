import time
from datetime import datetime
from pathlib import Path

from tmotorcan import (
    DataLogger, RealtimeLoop, update_all,
    MotorFaultError, MotorTimeoutError,
)

import sysid_config as cfg


def main():
    bus    = cfg.make_bus()
    motors = cfg.build_motors(bus)
    by_id  = {m.id: m for m in motors}

    # One multisine driving all three joints (channels are in MOTOR_IDS order).
    # It oscillates around each joint's band CENTRE (POS_CENTER), not the zeroed
    # pose, so an off-centre zero doesn't shrink the motion.
    amps    = [cfg.POS_AMP[mid]    for mid in cfg.MOTOR_IDS]
    centers = [cfg.POS_CENTER[mid] for mid in cfg.MOTOR_IDS]
    traj = cfg.MultiSine(amps, cfg.BASE_FREQ_HZ, cfg.N_HARMONICS,
                         cfg.DURATION_S, cfg.RAMP_S, seed=cfg.SEED, center=centers)

    # Combined CSV logger (one tidy, time-aligned file for identify_sysid.py).
    out_dir = Path(__file__).parent / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)
    fname   = out_dir / f'sysid_impedance_{datetime.now():%Y%m%d_%H%M%S}.csv'
    logger  = DataLogger(cfg.LOG_COLUMNS, filename=str(fname),
                         label='sysid', output_dir=str(out_dir))

    try:
        bus.drain()
        time.sleep(1.5)            # let the hardware finish booting

        for m in motors:
            m.enable()
            m.coast()
            # Hard position clamp as a safety net (reference already stays inside).
            # Limits are in MOTOR space, so map the custom-frame band through SIGN.
            m.set_limits('position', cfg.motor_limits(m.id, *cfg.JOINT_LIMITS[m.id]))
            m.set_current_limit(cfg.CURRENT_LIMIT_A)

        print(f"Excitation: {cfg.N_HARMONICS} harmonics of {cfg.BASE_FREQ_HZ} Hz, "
              f"{cfg.DURATION_S:.0f} s, motion (rad) = "
              + ", ".join(f"J{mid}:{c:+.2f}±{a:.2f}"
                          for mid, a, c in zip(cfg.MOTOR_IDS, amps, centers)))
        # ── Enter #1: calibrate the initial pose as zero ─────────────────────
        input("\n[1/2] Hand-pose the arm to the CENTRE of its safe range, then press "
              "Enter to CALIBRATE this pose as zero (Ctrl+C to abort)...")
        for m in motors:
            m.zero()               # current shaft angle becomes (0,0,0)

        # ── Hold at zero while you verify, until Enter #2 ────────────────────
        print("Calibrated. Motors are now HOLDING the zero pose.")
        print("[2/2] Verify the arm sits at the 0 pose you want, then press Enter "
              "to START data collection (Ctrl+C to abort)...")
        cfg.wait_for_enter_holding(motors, cfg.TRACK_KP, cfg.TRACK_KD, cfg.DT)

        print(f"\nLogging to {fname.name}.  Ctrl+C = soft stop.\n")

        loop       = RealtimeLoop(dt=cfg.DT)
        next_print = 0.0
        completed  = False              # True only on a clean finish (not Ctrl+C)
        for t in loop:
            q_ref  = traj.value(t)
            qd_ref = traj.deriv(t)

            for ch, mid in enumerate(cfg.MOTOR_IDS):
                m  = by_id[mid]
                s  = cfg.SIGN[mid]
                lo, hi = cfg.JOINT_LIMITS[mid]
                q_cmd = cfg.clamp(q_ref[ch], lo, hi)   # clamp in the custom frame
                m.cmd.position = s * q_cmd             # then map to the motor frame
                m.cmd.velocity = s * qd_ref[ch]
                m.cmd.torque   = 0.0
                m.cmd.kp       = cfg.TRACK_KP[mid]
                m.cmd.kd       = cfg.TRACK_KD[mid]

            update_all(motors, t)          # synced TX-all then RX-all

            # one combined row: t, (q,qd,tau)×3, ref×3, temp×3.
            # State is read back in MOTOR space -> map through SIGN to the custom
            # frame so the logged q,qd,tau match the kinematics the identifier uses.
            row = [f"{t:.6f}"]
            for mid in cfg.MOTOR_IDS:
                m = by_id[mid]
                s = cfg.SIGN[mid]
                row += [f"{s*m.state.position:.6f}",
                        f"{s*m.state.velocity:.6f}",
                        f"{s*m.state.torque:.6f}"]
            row += [f"{q_ref[ch]:.6f}" for ch in range(3)]   # ref already in-frame
            row += [f"{by_id[mid].state.temp:.1f}" for mid in cfg.MOTOR_IDS]
            logger.write(row)

            if t >= next_print:
                next_print = t + 0.2
                cols = "  ".join(
                    f"J{mid}: q={cfg.SIGN[mid]*by_id[mid].state.position:+.3f} "
                    f"tau={cfg.SIGN[mid]*by_id[mid].state.torque:+.2f}"
                    for mid in cfg.MOTOR_IDS)
                print(f"t={t:6.2f}/{traj.total_time:.0f}  {cols}")

            # Stop cleanly once the windowed trajectory has returned to rest.
            # f < 1.0 means a Ctrl+C fade is in progress -> let the loop finish it.
            if t >= traj.total_time:
                completed = True
                break

        if completed:
            # Trajectory finished at the zero pose -> keep HOLDING (0,0,0) with the
            # tracking impedance so you can set up the next attempt, then press Enter
            # to end. No rows are written during the hold (it is not ID data); the
            # logger is closed in the finally block.
            print("\nDone — trajectory complete. HOLDING the zero pose (0,0,0).")
            print("Set up for the next attempt, then press Enter to END "
                  "(motors will coast). Ctrl+C also ends.")
            try:
                cfg.wait_for_enter_holding(motors, cfg.TRACK_KP, cfg.TRACK_KD, cfg.DT)
            except KeyboardInterrupt:
                pass
            for m in motors:
                m.coast()
            print("Motors coasting.")
        else:
            # Ctrl+C / soft-stop during excitation -> just coast (no snap to zero).
            for m in motors:
                m.coast()
            print("\nStopped — motors coasting.")

    except (MotorFaultError, MotorTimeoutError) as e:
        print(f"\n[STOP] {e}")
    finally:
        logger.close()
        for m in motors:
            m.close()
        bus.close()


if __name__ == '__main__':
    main()
