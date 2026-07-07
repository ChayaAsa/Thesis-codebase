from __future__ import annotations

import os
import statistics
import sys
import time

# Resolve control_config path
from easy_path import WS_ROOT
_CTRL_ROOT = os.path.join(WS_ROOT, '3dof', '3_control')
if _CTRL_ROOT not in sys.path:
    sys.path.insert(0, _CTRL_ROOT)
from control_config import PORT, BITRATE, MOTOR_IDS, MODEL

PHASE1_TIMEOUT = 2.0     # seconds to wait for each raw reply
PHASE2_CYCLES  = 200     # update_all() iterations
PHASE2_HZ      = 100.0   # target rate during phase 2
PHASE2_TIMEOUT = 2.0     # timeout passed to update_all / recv_state

SEP = "=" * 62


# Phase 1 — raw bus test

def _encode_enable() -> bytes:
    return bytes([0xFF] * 7 + [0xFC])


def _encode_disable() -> bytes:
    return bytes([0xFF] * 7 + [0xFD])


def phase1_raw_test() -> dict[int, dict]:
    import can

    print(f"\n{SEP}")
    print("PHASE 1 — RAW BUS TEST  (bypasses MotorBus entirely)")
    print(SEP)

    try:
        raw_bus = can.interface.Bus(
            interface='slcan',
            channel=PORT,
            bitrate=BITRATE,
            frame_type='STD',
        )
    except Exception as exc:
        print(f"  [FATAL] Cannot open CAN bus on {PORT}: {exc}")
        print("  Check: USB cable, COM port number, adapter driver.")
        sys.exit(1)

    print(f"  Bus open: {raw_bus.channel_info}")

    # Drain any stale frames before we start pinging
    _drain_raw(raw_bus, window_s=0.3)

    results: dict[int, dict] = {}

    try:
        for mid in MOTOR_IDS:
            print(f"\n  Motor {mid}: sending enable (0xFF×7+0xFC) …", end='', flush=True)
            msg = can.Message(arbitration_id=mid, data=_encode_enable(),
                              is_extended_id=False)
            t_send = time.perf_counter()
            raw_bus.send(msg)

            r = {'replied': False, 'reply_ms': None, 'arb_id': None,
                 'data0': None, 'raw_data': None}

            deadline = t_send + PHASE1_TIMEOUT
            while time.perf_counter() < deadline:
                reply = raw_bus.recv(timeout=min(0.1, deadline - time.perf_counter()))
                if reply is None:
                    continue
                # Accept the first frame with ≥ 6 bytes that matches this motor
                # (by arbitration ID *or* data[0]).  On MIT CAN both should agree.
                if len(reply.data) < 6:
                    continue
                r['replied']   = True
                r['reply_ms']  = (time.perf_counter() - t_send) * 1e3
                r['arb_id']    = reply.arbitration_id
                r['data0']     = int(reply.data[0])
                r['raw_data']  = bytes(reply.data)
                break

            if r['replied']:
                ms   = r['reply_ms']
                arb  = r['arb_id']
                d0   = r['data0']
                if d0 == mid:
                    print(f"  replied in {ms:.1f} ms  arb_id={arb}  data[0]={d0}  OK")
                else:
                    print(f"  replied in {ms:.1f} ms  arb_id={arb}  data[0]={d0}"
                          f"  *** MISMATCH: expected {mid} ***")
            else:
                print(f"  NO REPLY in {PHASE1_TIMEOUT:.0f} s  -> HARDWARE FAIL")

            results[mid] = r

            # Send disable so motor doesn't stay in MIT mode between tests
            raw_bus.send(can.Message(arbitration_id=mid, data=_encode_disable(),
                                     is_extended_id=False))
            _drain_raw(raw_bus, window_s=0.15)

    finally:
        raw_bus.shutdown()

    # Verdict
    no_reply   = [mid for mid, r in results.items() if not r['replied']]
    mismatched = [mid for mid, r in results.items()
                  if r['replied'] and r['data0'] != mid]

    print(f"\n  {'─'*40}")
    print("  PHASE 1 VERDICT")
    if no_reply:
        print(f"  HARDWARE PROBLEM — motors {no_reply} did not reply.")
        print("  Check: power supply, CAN wiring, termination resistors,")
        print("         motor ID DIP-switch settings, correct COM port.")
    elif mismatched:
        for mid in mismatched:
            print(f"  ID MISMATCH — motor {mid} replied with data[0]="
                  f"{results[mid]['data0']} (expected {mid}).")
        print("  MotorBus routes replies by data[0].  A mismatch means")
        print("  replies are routed to the WRONG motor's queue → timeout.")
        print("  Fix: correct the motor ID in firmware or MOTOR_IDS list.")
    else:
        print("  PASS — all motors replied; data[0] matches expected IDs.")

    return results


def _drain_raw(bus, window_s: float = 0.2) -> int:
    n = 0
    deadline = time.perf_counter() + window_s
    while time.perf_counter() < deadline:
        m = bus.recv(timeout=min(0.05, deadline - time.perf_counter()))
        if m is not None:
            n += 1
    return n


# Phase 2 — software stack test

def phase2_stack_test() -> None:
    import can
    from tmotorcan import MotorBus, MITMotor, update_all
    from tmotorcan.protocol import MotorFaultError, MotorTimeoutError

    print(f"\n{SEP}")
    print(f"PHASE 2 — SOFTWARE STACK TEST  "
          f"({PHASE2_CYCLES} cycles @ {PHASE2_HZ:.0f} Hz)")
    print(SEP)

    raw  = can.interface.Bus(interface='slcan', channel=PORT,
                              bitrate=BITRATE, frame_type='STD')
    bus  = MotorBus(raw)
    motors = [MITMotor(bus, motor_id=mid, model=MODEL) for mid in MOTOR_IDS]

    # Enable all motors and do a first update to populate state
    for m in motors:
        m.enable()
    try:
        update_all(motors, timeout=3.0)
    except (MotorTimeoutError, MotorFaultError) as exc:
        print(f"  [ABORT] Cannot initialise motors: {exc}")
        print("  Phase 1 passed but Phase 2 init failed — check enable sequence.")
        _shutdown(motors, bus)
        return

    dt          = 1.0 / PHASE2_HZ
    rtt_ms:  dict[int, list[float]] = {mid: [] for mid in MOTOR_IDS}
    loop_ms: list[float] = []
    timeouts: list[dict] = []

    print(f"  Running …  (Ctrl+C to stop early)\n")

    for cycle in range(PHASE2_CYCLES):
        t_loop = time.perf_counter()

        # Safe zero-torque command every tick
        for m in motors:
            m.cmd.kp = m.cmd.kd = m.cmd.torque = m.cmd.position = m.cmd.velocity = 0.0

        # TX all motors together (same as your control loop)
        for m in motors:
            m.send_cmd()

        # RX each motor, timing individually
        for m in motors:
            t_recv = time.perf_counter()
            try:
                m.recv_state(timeout=PHASE2_TIMEOUT)
                rtt_ms[m.id].append((time.perf_counter() - t_recv) * 1e3)
            except MotorTimeoutError as exc:
                waited = (time.perf_counter() - t_recv) * 1e3
                rx_alive = bus._rx_thread.is_alive()
                timeouts.append({
                    'cycle':         cycle,
                    'motor_id':      exc.motor_id,
                    'waited_ms':     waited,
                    'rx_alive':      rx_alive,
                })
                print(f"  [TIMEOUT] cycle={cycle:4d}  motor={exc.motor_id}"
                      f"  waited={waited:.0f} ms"
                      f"  RX_thread={'alive' if rx_alive else '*** DEAD ***'}")
            except MotorFaultError as exc:
                print(f"  [FAULT]   cycle={cycle:4d}  motor={exc.motor_id}  code={exc.code}")

        loop_ms.append((time.perf_counter() - t_loop) * 1e3)

        if cycle % 50 == 0 and cycle > 0:
            print(f"  cycle {cycle:4d}/{PHASE2_CYCLES}"
                  f"  loop={loop_ms[-1]:.1f} ms"
                  f"  timeouts={len(timeouts)}")

        # Sleep the remaining slice
        remaining = dt - (time.perf_counter() - t_loop)
        if remaining > 0:
            time.sleep(remaining)

    _shutdown(motors, bus)

    # Report
    print(f"\n  {'─'*40}")
    print("  PHASE 2 RESULTS\n")

    for mid in MOTOR_IDS:
        s = rtt_ms[mid]
        if s:
            srt = sorted(s)
            p95 = srt[int(len(srt) * 0.95)]
            print(f"  Motor {mid}:  mean={statistics.mean(s):.1f} ms"
                  f"  max={max(s):.1f} ms"
                  f"  p95={p95:.1f} ms"
                  f"  n={len(s)}")
        else:
            print(f"  Motor {mid}:  *** NO SUCCESSFUL REPLIES ***")

    budget_ms   = dt * 1e3
    overrun_thr = budget_ms * 1.2          # >20% over budget counts as overrun
    overruns    = [t for t in loop_ms if t > overrun_thr]
    print(f"\n  Loop:  mean={statistics.mean(loop_ms):.1f} ms"
          f"  max={max(loop_ms):.1f} ms"
          f"  budget={budget_ms:.0f} ms"
          f"  overruns(>{overrun_thr:.0f} ms)={len(overruns)}/{len(loop_ms)}")

    # Verdict
    print(f"\n  {'─'*40}")
    print("  PHASE 2 VERDICT\n")

    if not timeouts:
        print("  PASS — 0 timeouts in 200 cycles at 100 Hz.")
        print("  Your timeouts in the real controller are likely caused by")
        print("  heavier computation (DOB/RFOB math + Jacobian) making the")
        print("  loop overrun its 10 ms budget.  Try dropping to 50 Hz:")
        print("    LOOP_HZ = 50  in RFOB_force.py")
        return

    n = len(timeouts)
    rx_died  = any(not t['rx_alive'] for t in timeouts)
    over_pct = len(overruns) / len(loop_ms) * 100

    if rx_died:
        bad = [t for t in timeouts if not t['rx_alive']]
        print(f"  *** SOFTWARE — RX THREAD DIED ***")
        print(f"  First occurrence: cycle {bad[0]['cycle']}  motor {bad[0]['motor_id']}")
        print("  The MotorBus background thread crashed.  Likely causes:")
        print("    • USB-CAN adapter disconnected (check cable)")
        print("    • USB driver error on Windows (try powered hub or USB 2.0 port)")
        print("    • slcan serial buffer overflow at 100 Hz with 3 motors")
        return

    if over_pct > 10:
        print(f"  SOFTWARE — LOOP OVERRUN ({over_pct:.0f}% of cycles exceed budget)")
        print("  The control computation takes too long → motor hasn't had time")
        print("  to reply before recv_state() is called with the short deadline.")
        print("  Fix: reduce LOOP_HZ to 50, or check for blocking calls in the loop.")
        return

    # Identify whether one motor dominates
    counts = {mid: sum(1 for t in timeouts if t['motor_id'] == mid)
              for mid in MOTOR_IDS}
    worst  = max(counts, key=counts.get)
    if counts[worst] >= n * 0.70:
        print(f"  LIKELY HARDWARE — motor {worst} causes {counts[worst]}/{n} timeouts.")
        print(f"  All other motors replied fine.  Focus on motor {worst}:")
        print("    • Check CAN wiring specifically for that motor")
        print("    • Check 120 Ω termination at that end of the bus")
        print("    • Try replacing the CAN cable on that joint")
    else:
        distr = "  ".join(f"M{mid}:{counts[mid]}" for mid in MOTOR_IDS)
        print(f"  INTERMITTENT / BUS NOISE — {n} timeouts spread: {distr}")
        print("  All motors affected roughly equally.  Likely causes:")
        print("    • Missing or wrong CAN termination (need 120 Ω at BOTH ends)")
        print("    • Long or unshielded CAN cables picking up motor PWM noise")
        print("    • USB-CAN adapter dropping frames under Windows scheduling jitter")
        print("  Quick check: what is the max round-trip above?")
        mx = max(max(v) for v in rtt_ms.values() if v)
        print(f"    max RTT = {mx:.1f} ms  (> 8 ms → OS jitter, not motor)")


def _shutdown(motors, bus) -> None:
    for m in motors:
        try: m.coast(); m.disable()
        except Exception: pass
    try: bus.close()
    except Exception: pass


# Entry point

if __name__ == '__main__':
    print(f"\n{'='*62}")
    print("  CAN Motor Timeout Diagnostic")
    print(f"  Port={PORT}   Bitrate={BITRATE//1000} kbit/s   Motors={MOTOR_IDS}")
    print(f"{'='*62}")

    p1 = phase1_raw_test()

    all_ok = all(r['replied'] and r['data0'] == mid for mid, r in p1.items())
    if all_ok:
        print("\nPhase 1 passed — starting Phase 2 …")
        try:
            phase2_stack_test()
        except KeyboardInterrupt:
            print("\n  [Interrupted by user]")
    else:
        print("\nPhase 1 failed — fix the hardware issue before running Phase 2.")

    print(f"\n{'='*62}")
    print("  Diagnostic complete.")
    print(f"{'='*62}\n")
