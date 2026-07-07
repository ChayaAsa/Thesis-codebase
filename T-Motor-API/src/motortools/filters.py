class IIRFilter:

    def __init__(self, alpha: float, initial: float = 0.0) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._y = initial

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not (0.0 < value <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {value}")
        self._alpha = value

    @property
    def value(self) -> float:
        return self._y

    def reset(self, value: float = 0.0) -> None:
        self._y = value

    def __call__(self, x: float) -> float:
        self._y = self._alpha * x + (1.0 - self._alpha) * self._y
        return self._y


class FiniteDiff:

    def __init__(self) -> None:
        self._prev_val: float | None = None
        self._prev_t:   float | None = None

    def reset(self) -> None:
        self._prev_val = None
        self._prev_t   = None

    def __call__(self, value: float, t: float) -> float | None:
        result = None
        if self._prev_val is not None and self._prev_t is not None:
            dt = t - self._prev_t
            if dt > 0:
                result = (value - self._prev_val) / dt
        self._prev_val = value
        self._prev_t   = t
        return result


class PositionUnwrapper:

    def __init__(self, period: float) -> None:
        self._period = period
        self._prev:   float | None = None
        self._offset: float        = 0.0

    def reset(self) -> None:
        self._prev   = None
        self._offset = 0.0

    def __call__(self, raw: float) -> float:
        if self._prev is not None:
            diff = raw - self._prev
            # Use >= / <= so a diff of exactly ±period/2 is decisive
            # (a strict > / < leaves the boundary undetected).
            if diff >=  self._period / 2:
                self._offset -= self._period
            elif diff <= -self._period / 2:
                self._offset += self._period
        self._prev = raw
        return raw + self._offset
