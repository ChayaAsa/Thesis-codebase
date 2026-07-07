def update_all(motors, t: float = 0.0, timeout: float = 1.0) -> None:
    motors = list(motors)
    for m in motors:
        m.send_cmd()
    for m in motors:
        m.recv_state(timeout=timeout, t=t)


class MotorGroup:

    def __init__(self, motors) -> None:
        self._motors = list(motors)

    # Synchronised I/O

    def update_all(self, t: float = 0.0, timeout: float = 1.0) -> None:
        update_all(self._motors, t=t, timeout=timeout)

    def send_cmd(self) -> None:
        for m in self._motors:
            m.send_cmd()

    def recv_state(self, timeout: float = 1.0, t: float = 0.0) -> None:
        for m in self._motors:
            m.recv_state(timeout=timeout, t=t)

    # Broadcast helpers

    def enable(self) -> None:    self._broadcast('enable')
    def disable(self) -> None:   self._broadcast('disable')
    def zero(self) -> None:      self._broadcast('zero')
    def coast(self) -> None:     self._broadcast('coast')
    def close(self) -> None:     self._broadcast('close')

    def set_log(self, *args, **kwargs) -> None:
        self._broadcast('set_log', *args, **kwargs)

    def set_vel_filter(self, *args, **kwargs) -> None:
        self._broadcast('set_vel_filter', *args, **kwargs)

    def set_multi_turn(self, *args, **kwargs) -> None:
        self._broadcast('set_multi_turn', *args, **kwargs)

    def set_accel_est(self, *args, **kwargs) -> None:
        self._broadcast('set_accel_est', *args, **kwargs)

    def set_limits(self, *args, **kwargs) -> None:
        self._broadcast('set_limits', *args, **kwargs)

    def set_current_limit(self, *args, **kwargs) -> None:
        self._broadcast('set_current_limit', *args, **kwargs)

    def set_temp_derating(self, *args, **kwargs) -> None:
        self._broadcast('set_temp_derating', *args, **kwargs)

    # Container protocol

    def __iter__(self):      return iter(self._motors)
    def __len__(self):       return len(self._motors)
    def __getitem__(self, i): return self._motors[i]

    # Internal

    def _broadcast(self, method: str, *args, **kwargs) -> None:
        for m in self._motors:
            fn = getattr(m, method, None)
            if callable(fn):
                fn(*args, **kwargs)
