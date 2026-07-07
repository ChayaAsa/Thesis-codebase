from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .bus import MotorBus
from .protocol import MotorFaultError, MotorTimeoutError, FAULT_MESSAGES, _encode, _decode
from .models import load_params, MotorParams

if TYPE_CHECKING:
    # Only evaluated by type-checkers (mypy/pylance), never at runtime.
    from motortools.filters import IIRFilter, FiniteDiff, PositionUnwrapper
    from motortools.limits  import SoftLimiter, ThermalDerate
    from motortools.logger  import DataLogger


@dataclass
class MotorState:
    position:     float = 0.0   # rad
    velocity:     float = 0.0   # rad/s
    acceleration: float = 0.0   # rad/s²
    torque:       float = 0.0   # N·m
    temp:         int   = 0     # °C
    error:        int   = 0     # 0 = healthy

    def as_tuple(self) -> tuple:
        return (self.position, self.velocity, self.acceleration,
                self.torque, self.temp, self.error)

    @staticmethod
    def from_tuple(t: tuple) -> 'MotorState':
        return MotorState(*t)


@dataclass
class MotorSetpoint:
    position: float = 0.0   # rad
    velocity: float = 0.0   # rad/s
    torque:   float = 0.0   # N·m
    kp:       float = 0.0   # position stiffness (N·m/rad)
    kd:       float = 0.0   # velocity damping (N·m·s/rad)


class MITMotor:

    def __init__(self, bus: MotorBus, motor_id: int,
                 model: str = 'AK45-10',
                 max_temp: int | None = None,
                 log: bool | str = False) -> None:
        self.id     = motor_id
        self.state  = MotorState()
        self.cmd    = MotorSetpoint()
        self.params: MotorParams = load_params(model)

        self._bus      = bus
        self._max_temp = max_temp if max_temp is not None else self.params.max_temp
        self._kt_out   = self.params.kt * self.params.gear_ratio  # N·m/A at output shaft
        self._closed   = False
        bus.register(motor_id)

        self._pos_range: float = self.params.p_max - self.params.p_min  # 25.0 rad

        # all optional features off by default (None = disabled)
        self._vel_filter: IIRFilter        | None = None
        self._accel_diff: FiniteDiff       | None = None
        self._unwrapper:  PositionUnwrapper| None = None
        self._limiter:    SoftLimiter | None = None   # created on first set_limits() call
        self._derate:     ThermalDerate | None = None

        # empty pipelines = zero per-tick overhead when no features are active
        self._rx_pipeline: list = []
        self._tx_pipeline: list = []

        # per-tick scratch — RX side (written by _parse_response, read by rx steps)
        self._cur_raw_pos:   float        = 0.0
        self._cur_raw_vel:   float        = 0.0
        self._cur_now:       float        = 0.0

        # per-tick scratch — TX side (written by send_cmd, mutated by tx steps)
        self._tx_pos: float = 0.0
        self._tx_vel: float = 0.0
        self._tx_tor: float = 0.0

        self._logger: DataLogger | None = None
        if log:
            self._open_log(log if isinstance(log, str) else True)

    # Core

    def update(self, t: float = 0.0, timeout: float = 1.0) -> None:
        self.send_cmd()
        self.recv_state(timeout=timeout, t=t)

    def send_cmd(self) -> None:
        c = self.cmd
        self._tx_pos = c.position
        self._tx_vel = c.velocity
        self._tx_tor = c.torque
        for step in self._tx_pipeline:
            step()
        self._send_raw_cmd(position=self._tx_pos, velocity=self._tx_vel,
                           torque=self._tx_tor, kp=c.kp, kd=c.kd)

    def recv_state(self, timeout: float = 1.0, t: float = 0.0) -> None:
        msg = self._bus.recv(self.id, timeout=timeout)
        if msg is None:
            raise MotorTimeoutError(self.id, timeout)
        self._parse_response(msg.data)
        for step in self._rx_pipeline:
            step()
        self._check_safety()
        self._log_state(t)

    # Mode commands

    def enable(self) -> bool:
        res = False
        self.coast()
        self._bus.send(self.id, bytes([0xFF] * 7 + [0xFC]))
        msg = self._bus.recv(self.id)
        if msg:
            self._parse_response(msg.data)
            res = True
        return res

    def disable(self) -> bool:
        res = False
        self._bus.send(self.id, bytes([0xFF] * 7 + [0xFD]))
        msg = self._bus.recv(self.id)
        if msg:
            self._parse_response(msg.data)
            res = True
        self.coast()
        return res


    def zero(self) -> bool:
        res = False
        self._bus.send(self.id, bytes([0xFF] * 7 + [0xFE]))
        msg = self._bus.recv(self.id)
        if msg:
            self._parse_response(msg.data)
            res = True
        return res

    def coast(self) -> bool:
        res = False
        self._send_raw_cmd(position=0.0, velocity=0.0, torque=0.0, kp=0.0, kd=0.0)
        msg = self._bus.recv(self.id, timeout=0.5)
        if msg:
            self._parse_response(msg.data)
            res = True
        return res

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.disable()            # uses bus.recv
        self._close_log()
        self._bus.unregister(self.id)

    @property
    def kt_out(self) -> float:
        return self._kt_out

    # Optional feature toggles

    def set_vel_filter(self, alpha: float | None) -> None:
        if alpha is None:
            if self._vel_filter is None:
                return
            self._vel_filter = None
            self._rebuild_pipeline()
        elif self._vel_filter is None:
            try:
                from motortools.filters import IIRFilter
            except ImportError:
                raise ImportError(
                    "velocity filtering requires motortools — "
                    "pip install tmotorcan[full]"
                ) from None
            self._vel_filter = IIRFilter(alpha, initial=self.state.velocity)
            self._rebuild_pipeline()
        else:
            self._vel_filter.alpha = alpha   # keep accumulated state

    def set_accel_est(self, enabled: bool) -> None:
        if enabled:
            try:
                from motortools.filters import FiniteDiff
            except ImportError:
                raise ImportError(
                    "acceleration estimation requires motortools — "
                    "pip install tmotorcan[full]"
                ) from None
            self._accel_diff = FiniteDiff()
        else:
            self._accel_diff = None
            self.state.acceleration = 0.0
        self._rebuild_pipeline()

    def set_multi_turn(self, enabled: bool) -> None:
        if enabled:
            try:
                from motortools.filters import PositionUnwrapper
            except ImportError:
                raise ImportError(
                    "multi-turn unwrapping requires motortools — "
                    "pip install tmotorcan[full]"
                ) from None
            self._unwrapper = PositionUnwrapper(self._pos_range)
        else:
            self._unwrapper = None
        self._rebuild_pipeline()

    # Safety envelopes

    def set_limits(self, field: str, lo_hi: tuple[float, float] | None) -> None:
        if lo_hi is None and self._limiter is None:
            return  # nothing to remove
        if self._limiter is None:
            try:
                from motortools.limits import SoftLimiter
            except ImportError:
                raise ImportError(
                    "soft limits require motortools — "
                    "pip install tmotorcan[full]"
                ) from None
            self._limiter = SoftLimiter()
        self._limiter.set(field, lo_hi)
        self._rebuild_pipeline()

    def set_current_limit(self, amps: float | None) -> None:
        if amps is None:
            if self._limiter is None:
                return  # no limiter exists, nothing to remove
            self.set_limits('torque', None)
        else:
            t_max = abs(amps) * self._kt_out
            self.set_limits('torque', (-t_max, t_max))

    @property
    def limited(self) -> set[str]:
        if self._limiter is None or not self._limiter.active():
            return set()
        return self._limiter.last_clamped

    def set_temp_derating(self, enabled: bool,
                          start_C: float | None = None,
                          end_C:   float | None = None) -> None:
        if not enabled:
            self._derate = None
        else:
            try:
                from motortools.limits import ThermalDerate
            except ImportError:
                raise ImportError(
                    "thermal derating requires motortools — "
                    "pip install tmotorcan[full]"
                ) from None
            if start_C is None:
                start_C = self._max_temp - 10
            if end_C is None:
                end_C = self._max_temp
            self._derate = ThermalDerate(start_C=start_C, end_C=end_C)
        self._rebuild_pipeline()

    # Homing

    def home(self, direction: float = 1.0,
             current_A: float = 2.0,
             speed:     float = 0.5,
             timeout:   float = 10.0,
             dt:        float = 0.02,
             kd:        float = 0.5) -> None:
        threshold_Nm = abs(current_A) * self._kt_out
        drive_vel    = (1.0 if direction >= 0 else -1.0) * abs(speed)

        saved = (self.cmd.position, self.cmd.velocity, self.cmd.torque,
                 self.cmd.kp, self.cmd.kd)
        self.cmd.position = 0.0
        self.cmd.velocity = drive_vel
        self.cmd.torque   = 0.0
        self.cmd.kp       = 0.0
        self.cmd.kd       = kd

        t_start         = time.perf_counter()
        settle_deadline = t_start + 0.3
        hit             = False

        try:
            while True:
                now = time.perf_counter()
                if now - t_start > timeout:
                    raise TimeoutError(
                        f"Motor {self.id}: no hard stop detected within {timeout}s "
                        f"(|torque| never exceeded {threshold_Nm:.3f} N·m)")
                self.update()
                if now >= settle_deadline and abs(self.state.torque) >= threshold_Nm:
                    hit = True
                    break
                time.sleep(dt)
        finally:
            (self.cmd.position, self.cmd.velocity, self.cmd.torque,
             self.cmd.kp, self.cmd.kd) = saved

        if hit:
            self.coast()
            self.zero()

    # Internal pipeline steps

    def _rebuild_pipeline(self) -> None:
        self._rx_pipeline = [fn for fn, active in (
            (self._step_multi_turn, self._unwrapper  is not None),
            (self._step_vel_filter, self._vel_filter is not None),
            (self._step_accel_est,  self._accel_diff is not None),
        ) if active]
        self._tx_pipeline = [fn for fn, active in (
            (self._step_limits, self._limiter is not None and self._limiter.active()),
            (self._step_derate, self._derate is not None),
        ) if active]

    # RX steps ­— called from _parse_response, read/write self.state + scratch
    def _step_multi_turn(self) -> None:
        self.state.position = self._unwrapper(self._cur_raw_pos)

    def _step_vel_filter(self) -> None:
        self.state.velocity = self._vel_filter(self._cur_raw_vel)

    def _step_accel_est(self) -> None:
        accel = self._accel_diff(self.state.velocity, self._cur_now)
        if accel is not None:
            self.state.acceleration = accel

    # TX steps — called from send_cmd, mutate _tx_pos / _tx_vel / _tx_tor
    def _step_limits(self) -> None:
        self._limiter.reset_tracking()
        self._tx_pos = self._limiter.clamp('position', self._tx_pos)
        self._tx_vel = self._limiter.clamp('velocity', self._tx_vel)
        self._tx_tor = self._limiter.clamp('torque',   self._tx_tor)

    def _step_derate(self) -> None:
        self._tx_tor *= self._derate(self.state.temp)

    # Context manager

    def __enter__(self) -> 'MITMotor':
        if not self.enable():
            # No reply to the 0xFC enable frame — bus alive but motor silent.
            # Raise now so the user sees the real cause instead of a confusing
            # MotorTimeoutError on the first update() one second later.
            raise MotorTimeoutError(self.id, 1.0)
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # Log

    @property
    def log(self) -> bool:
        return self._logger is not None and self._logger.active

    def set_log(self, enabled: bool, name: str | None = None) -> None:
        if enabled:
            if not self.log:
                self._open_log(name or True)
        else:
            self._close_log()

    # Internal

    def _open_log(self, filename: bool | str) -> None:
        try:
            from motortools.logger import DataLogger
        except ImportError:
            raise ImportError(
                "CSV logging requires motortools — "
                "pip install tmotorcan[full]"
            ) from None
        self._logger = DataLogger(
            columns=['time_s', 'pos_rad', 'vel_rad_s', 'accel_rad_s2', 'torque_Nm', 'temp_C', 'error'],
            filename=filename,
            label=f'motor{self.id}',
        )

    def _close_log(self) -> None:
        if self._logger:
            self._logger.close()
            self._logger = None

    def _send_raw_cmd(self, *, position: float, velocity: float, torque: float,
                      kp: float, kd: float) -> None:
        p = self.params
        pos = _encode(position, p.p_min, p.p_max, 16)
        vel = _encode(velocity, p.v_min, p.v_max, 12)
        tor = _encode(torque,   p.t_min, p.t_max, 12)
        kp_ = _encode(kp,       p.kp_min, p.kp_max, 12)
        kd_ = _encode(kd,       p.kd_min, p.kd_max, 12)

        # MIT CAN frame layout: pos[15:0] | vel[11:0] | kp[11:0] | kd[11:0] | tor[11:0]
        buf = bytearray(8)
        buf[0] =  pos >> 8
        buf[1] =  pos & 0xFF
        buf[2] =  vel >> 4
        buf[3] = ((vel & 0xF) << 4) | (kp_ >> 8)
        buf[4] =  kp_ & 0xFF
        buf[5] =  kd_ >> 4
        buf[6] = ((kd_ & 0xF) << 4) | (tor >> 8)
        buf[7] =  tor & 0xFF
        self._bus.send(self.id, buf)

    def _parse_response(self, d: bytes | bytearray) -> None:
        p   = self.params
        now = time.perf_counter()

        p_int = (d[1] << 8) | d[2]
        v_int = (d[3] << 4) | (d[4] >> 4)
        t_int = ((d[4] & 0xF) << 8) | d[5]

        raw_pos = _decode(p_int, p.p_min, p.p_max, 16)
        raw_vel = _decode(v_int, p.v_min, p.v_max, 12)

        # write scratch space then run active steps
        self._cur_raw_pos = raw_pos
        self._cur_raw_vel = raw_vel
        self._cur_now     = now

        # defaults — overridden by pipeline steps when features are enabled
        self.state.position   = raw_pos
        self.state.velocity   = raw_vel      
        self.state.torque = _decode(t_int, p.t_min, p.t_max, 12)
        self.state.temp   = float(d[6]) - 40.0 if len(d) > 6 else 0
        self.state.error  = int(d[7]) if len(d) > 7 else 0


    def _check_safety(self) -> None:
        if self.state.error:
            raise MotorFaultError(self.id, self.state.error,
                                  FAULT_MESSAGES.get(self.state.error, 'Unknown'))
        if self.state.temp > self._max_temp:
            raise MotorFaultError(self.id, 0,
                                  f"{self.state.temp}°C > {self._max_temp}°C")

    def _log_state(self, t: float) -> None:
        # Snapshot the reference: another thread may set self._logger to None
        # via set_log(False) between the truthiness check and the .write() call.
        logger = self._logger
        if logger:
            logger.write([
                f"{t:.6f}",
                f"{self.state.position:.6f}",
                f"{self.state.velocity:.6f}",
                f"{self.state.acceleration:.6f}",
                f"{self.state.torque:.6f}",
                f"{self.state.temp:.1f}",
                self.state.error,
            ])
