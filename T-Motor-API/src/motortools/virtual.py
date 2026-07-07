import argparse
import sys
import threading
import time
from pathlib import Path

import can

from tmotorcan.models   import load_params, MotorParams
from tmotorcan.protocol import _encode, _decode


# ── Default dynamics (same values as the .ino) ───────────────────────────────

SIM_INERTIA = 0.08   # kg·m² reflected to output shaft
SIM_DAMPING = 0.15   # N·m·s/rad


class VirtualMotor:

    def __init__(self, params: MotorParams, motor_id: int,
                 inertia: float = SIM_INERTIA,
                 damping: float = SIM_DAMPING) -> None:
        self.params   = params
        self.motor_id = motor_id
        self.inertia  = inertia
        self.damping  = damping

        self.pos      = 0.0    # rad
        self.vel      = 0.0    # rad/s
        self.torque   = 0.0    # N·m (last applied torque
        self.temp     = 25     # °C
        self.error    = 0
        self.enabled  = False
        self._last_t  = time.perf_counter()

    # Frame I/O

    def handle_frame(self, data: bytes) -> bytes | None:
        if len(data) != 8:
            return None

        # Special commands — seven 0xFF bytes + action byte
        if data[:7] == b'\xff' * 7:
            action = data[7]
            if action == 0xFC:
                self.enabled = True
                self.pos = self.vel = self.torque = 0.0
                self._last_t = time.perf_counter()
            elif action == 0xFD:
                self.enabled = False
            elif action == 0xFE:
                self.pos = 0.0
                self.vel = 0.0
            return self._build_reply()

        # Regular impedance command — decode, step dynamics, reply
        p_des, v_des, kp, kd, t_ff = self._decode_command(data)
        self._step(p_des, v_des, kp, kd, t_ff)
        return self._build_reply()

    def _decode_command(self, data: bytes) -> tuple[float, float, float, float, float]:
        p = self.params
        p_int  = (data[0] << 8) | data[1]
        v_int  = (data[2] << 4) | (data[3] >> 4)
        kp_int = ((data[3] & 0x0F) << 8) | data[4]
        kd_int = (data[5] << 4) | (data[6] >> 4)
        t_int  = ((data[6] & 0x0F) << 8) | data[7]
        return (
            _decode(p_int,  p.p_min,  p.p_max,  16),
            _decode(v_int,  p.v_min,  p.v_max,  12),
            _decode(kp_int, p.kp_min, p.kp_max, 12),
            _decode(kd_int, p.kd_min, p.kd_max, 12),
            _decode(t_int,  p.t_min,  p.t_max,  12),
        )

    def _build_reply(self) -> bytes:
        p = self.params
        pos_i = _encode(self.pos,    p.p_min, p.p_max, 16)
        vel_i = _encode(self.vel,    p.v_min, p.v_max, 12)
        tor_i = _encode(self.torque, p.t_min, p.t_max, 12)

        buf = bytearray(8)
        buf[0] = self.motor_id
        buf[1] =  pos_i >> 8
        buf[2] =  pos_i & 0xFF
        buf[3] =  vel_i >> 4
        buf[4] = ((vel_i & 0x0F) << 4) | (tor_i >> 8)
        buf[5] =  tor_i & 0xFF
        buf[6] = int(self.temp)
        buf[7] = int(self.error)
        return bytes(buf)

    # Dynamics

    def _step(self, p_des: float, v_des: float,
              kp: float, kd: float, t_ff: float) -> None:
        p = self.params
        now = time.perf_counter()
        dt  = now - self._last_t
        self._last_t = now
        if not (0.0 < dt < 0.1):
            dt = 0.005     # guard startup / long stalls

        if self.enabled:
            torque = kp * (p_des - self.pos) + kd * (v_des - self.vel) + t_ff
            torque = max(p.t_min, min(p.t_max, torque))
        else:
            torque = 0.0

        accel = (torque - self.damping * self.vel) / self.inertia
        self.vel += accel * dt
        self.vel  = max(p.v_min, min(p.v_max, self.vel))
        self.pos += self.vel * dt
        self.pos  = max(p.p_min, min(p.p_max, self.pos))
        self.torque = torque


# In-process helper

class SimHandle:
    def __init__(self, motor: 'VirtualMotor', stop: threading.Event,
                 thread: threading.Thread) -> None:
        self.motor  = motor
        self.stop   = stop
        self.thread = thread
        self.params = motor.params

    def set(self)    -> None: self.stop.set()
    def is_set(self) -> bool: return self.stop.is_set()


def serve_in_thread(motor_id: int = 1,
                    channel:  str = 'virt',
                    model:    str = 'AK45-10',
                    inertia:  float = SIM_INERTIA,
                    damping:  float = SIM_DAMPING,
                    verbose:  bool = False) -> SimHandle:
    params = load_params(model)
    motor  = VirtualMotor(params, motor_id, inertia=inertia, damping=damping)
    stop   = threading.Event()

    sim_bus = can.interface.Bus(
        interface='virtual', channel=channel,
        bitrate=1_000_000, receive_own_messages=False,
    )

    def _loop() -> None:
        try:
            while not stop.is_set():
                msg = sim_bus.recv(timeout=0.1)
                if msg is None or msg.arbitration_id != motor_id or len(msg.data) != 8:
                    continue
                reply = motor.handle_frame(bytes(msg.data))
                if reply is None:
                    continue
                sim_bus.send(can.Message(arbitration_id=motor_id,
                                         data=reply, is_extended_id=False))
                if verbose:
                    print(f"[virtual_motor] pos={motor.pos:+.3f}  vel={motor.vel:+.3f}  "
                          f"tor={motor.torque:+.3f}  enabled={motor.enabled}")
        finally:
            sim_bus.shutdown()

    th = threading.Thread(target=_loop, name=f'VirtualMotor-{motor_id}', daemon=True)
    th.start()
    print(f"[virtual_motor] {model} motor_id={motor_id} serving in-thread on channel='{channel}'")
    return SimHandle(motor, stop, th)


# CLI

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Virtual AK-series motor on a python-can virtual bus.')
    ap.add_argument('--channel',  default='virt',    help='virtual channel name (default: %(default)s)')
    ap.add_argument('--motor-id', type=int, default=1, help='CAN arbitration ID (default: %(default)s)')
    ap.add_argument('--model',    default='AK45-10', help='motor model YAML (default: %(default)s)')
    ap.add_argument('--inertia',  type=float, default=SIM_INERTIA, help='J in kg·m² (default: %(default)s)')
    ap.add_argument('--damping',  type=float, default=SIM_DAMPING, help='B in N·m·s/rad (default: %(default)s)')
    ap.add_argument('--verbose', '-v', action='store_true', help='print state on every command')
    args = ap.parse_args()

    params = load_params(args.model)
    motor  = VirtualMotor(params, args.motor_id,
                          inertia=args.inertia, damping=args.damping)

    bus = can.interface.Bus(
        interface='virtual',
        channel=args.channel,
        bitrate=1_000_000,
        receive_own_messages=False,   # don't loop back our own replies
    )

    print(f"[virtual_motor] {args.model}  motor_id={args.motor_id}  "
          f"channel='{args.channel}'  J={args.inertia}  B={args.damping}")
    print(f"[virtual_motor] Waiting for commands. Ctrl+C to stop.\n")

    try:
        for msg in bus:
            if msg is None or msg.arbitration_id != args.motor_id or len(msg.data) != 8:
                continue
            reply = motor.handle_frame(bytes(msg.data))
            if reply is None:
                continue
            bus.send(can.Message(arbitration_id=args.motor_id,
                                 data=reply, is_extended_id=False))
            if args.verbose:
                print(f"[virtual_motor] pos={motor.pos:+.3f} rad  vel={motor.vel:+.3f} rad/s  "
                      f"tor={motor.torque:+.3f} N·m  temp={motor.temp}°C  "
                      f"err={motor.error}  enabled={motor.enabled}")
    except KeyboardInterrupt:
        pass
    finally:
        bus.shutdown()
        print("\n[virtual_motor] Shutdown.")


if __name__ == '__main__':
    main()
