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

    amps = [cfg.TORQUE_AMP[mid] for mid in cfg.MOTOR_IDS]
    traj = cfg.MultiSine(amps, cfg.BASE_FREQ_HZ, cfg.N_HARMONICS,
                         cfg.DURATION_S, cfg.RAMP_S, seed=cfg.SEED)

    out_dir = Path(__file__).parent / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)
    fname   = out_dir / f'sysid_torque_{datetime.now():%Y%m%d_%H%M%S}.csv'
    logger  = DataLogger(cfg.LOG_COLUMNS, filename=str(fname),
                         label='sysid', output_dir=str(out_dir))

    aborted = None
    try:
        bus.drain()
        time.sleep(1.5)

        for m in motors:
            m.enable()
            m.coast()
            m.set_current_limit(cfg.CURRENT_LIMIT_A)   # phase-current cap

        print(f"OPEN-LOOP TORQUE excitation, peak amps (N·m) = "
              + ", ".join(f"J{mid}:{a:.2f}" for mid, a in zip(cfg.MOTOR_IDS, amps)))
        print("⚠  The arm will move freely under gravity. Keep clear; Ctrl+C fades out.")
        # ── Enter #1: calibrate the initial pose as zero ─────────────────────
        input("\n[1/2] Hand-pose the arm to the CENTRE of its safe range, then press "
              "Enter to CALIBRATE this pose as zero (Ctrl+C to abort)...")
        for m in motors:
            m.zero()

        # ── Hold at zero (impedance) while you verify, until Enter #2 ────────
        print("Calibrated. Motors are now HOLDING the zero pose (impedance).")
        print("[2/2] Verify the arm sits at the 0 pose you want, then press Enter "
              "to START open-loop torque excitation (Ctrl+C to abort)...")
        cfg.wait_for_enter_holding(motors, cfg.TRACK_KP, cfg.TRACK_KD, cfg.DT)

        print(f"\nLogging to {fname.name}.  Ctrl+C = soft stop.\n")

        loop       = RealtimeLoop(dt=cfg.DT, report=True, fade=0.4)
        next_print = 0.0
        completed  = False              # True only on a clean finish (not Ctrl+C/abort)
        for t in loop:
            f      = loop.fade
            tau_ff = traj.value(t)

            # During the ramp-in window [0, RAMP_S] the multisine torque is windowed
            # to zero, so without stiffness the arm would fall under gravity immediately.
            # Blend kp/kd from full impedance hold down to pure damping in sync with
            # the torque window, so the arm is always supported during the transition.
            alpha = min(1.0, t / cfg.RAMP_S) if cfg.RAMP_S > 0.0 else 1.0

            for ch, mid in enumerate(cfg.MOTOR_IDS):
                m = by_id[mid]
                m.cmd.position = 0.0
                m.cmd.velocity = 0.0
                m.cmd.kp       = cfg.TRACK_KP[mid] * (1.0 - alpha) * f
                m.cmd.kd       = (cfg.TRACK_KD[mid] * (1.0 - alpha)
                                  + cfg.CONTAIN_KD[mid] * alpha) * f
                m.cmd.torque   = cfg.SIGN[mid] * tau_ff[ch] * f

            update_all(motors, t)

            # ── hard safety-band check: abort the instant a joint leaves its band ──
            # state.position is MOTOR space -> map through SIGN to the custom frame.
            for mid in cfg.MOTOR_IDS:
                lo, hi = cfg.JOINT_LIMITS[mid]
                q = cfg.SIGN[mid] * by_id[mid].state.position
                if q < lo or q > hi:
                    aborted = (mid, q, lo, hi)
                    break
            if aborted is not None:
                break

            # State read in MOTOR space -> map through SIGN to the custom frame.
            row = [f"{t:.6f}"]
            for mid in cfg.MOTOR_IDS:
                m = by_id[mid]
                s = cfg.SIGN[mid]
                row += [f"{s*m.state.position:.6f}",
                        f"{s*m.state.velocity:.6f}",
                        f"{s*m.state.torque:.6f}"]
            row += [f"{tau_ff[ch]:.6f}" for ch in range(3)]      # commanded torque (in-frame)
            row += [f"{by_id[mid].state.temp:.1f}" for mid in cfg.MOTOR_IDS]
            logger.write(row)

            if t >= next_print:
                next_print = t + 0.2
                cols = "  ".join(
                    f"J{mid}: q={cfg.SIGN[mid]*by_id[mid].state.position:+.3f} "
                    f"tau={cfg.SIGN[mid]*by_id[mid].state.torque:+.2f}"
                    for mid in cfg.MOTOR_IDS)
                print(f"t={t:6.2f}/{traj.total_time:.0f}  {cols}")

            if t >= traj.total_time and f >= 1.0:
                completed = True
                break

        if completed:
            # Clean finish: the arm drifted under gravity, so don't snap to zero.
            # Latch the current pose firmly, settle, then ramp the target down to 0
            # in small steps; finally HOLD (0,0,0) until Enter ends the run. No rows
            # are logged during recovery (it is not ID data); logger closes in finally.
            print("\nDone — recovering to the zero pose (firm hold, then slow ramp)...")
            try:
                cfg.recover_to_zero(motors, cfg.HOLD_KP, cfg.HOLD_KD, cfg.DT)
                print("At zero. HOLDING (0,0,0). Set up for the next attempt, then "
                      "press Enter to END (motors will coast). Ctrl+C also ends.")
                cfg.wait_for_enter_holding(motors, cfg.HOLD_KP, cfg.HOLD_KD, cfg.DT)
            except KeyboardInterrupt:
                pass
            for m in motors:
                m.coast()
            print("Motors coasting.")
        else:
            # Band abort or Ctrl+C -> coast immediately; never drive a runaway arm.
            for m in motors:
                m.coast()
            if aborted:
                mid, q, lo, hi = aborted
                print(f"\n[ABORT] J{mid} left its safe band: q={q:+.3f} not in "
                      f"[{lo:+.3f}, {hi:+.3f}]. Motors coasting.")
            else:
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
