class SoftLimiter:

    def __init__(self) -> None:
        self._limits: dict[str, tuple[float, float]] = {}
        self._clamped: set[str] = set()

    def set(self, field: str, lo_hi: tuple[float, float] | None) -> None:
        if lo_hi is None:
            self._limits.pop(field, None)
            return
        lo, hi = lo_hi
        if lo >= hi:
            raise ValueError(
                f"{field} limits must be (lo, hi) with lo < hi, got {lo_hi}")
        self._limits[field] = (lo, hi)

    def get(self, field: str) -> tuple[float, float] | None:
        return self._limits.get(field)

    def active(self, field: str | None = None) -> bool:
        return field in self._limits if field else bool(self._limits)

    @property
    def last_clamped(self) -> set[str]:
        return set(self._clamped)

    def reset_tracking(self) -> None:
        self._clamped.clear()

    def clamp(self, field: str, value: float) -> float:
        lim = self._limits.get(field)
        if lim is None:
            return value
        lo, hi = lim
        if value < lo:
            self._clamped.add(field)
            return lo
        if value > hi:
            self._clamped.add(field)
            return hi
        return value


class ThermalDerate:

    def __init__(self, start_C: float, end_C: float) -> None:
        if end_C <= start_C:
            raise ValueError(
                f"end_C must be > start_C, got start={start_C}, end={end_C}")
        self._start = float(start_C)
        self._end   = float(end_C)

    @property
    def start_C(self) -> float:
        return self._start

    @property
    def end_C(self) -> float:
        return self._end

    def __call__(self, temp_C: float) -> float:
        if temp_C <= self._start:
            return 1.0
        if temp_C >= self._end:
            return 0.0
        return 1.0 - (temp_C - self._start) / (self._end - self._start)
