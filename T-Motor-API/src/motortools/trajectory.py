from math import sqrt


class TrapezoidalProfile:

    def __init__(self, v_max: float, a_max: float) -> None:
        if v_max <= 0:
            raise ValueError(f"v_max must be > 0, got {v_max}")
        if a_max <= 0:
            raise ValueError(f"a_max must be > 0, got {a_max}")
        self._v_max = v_max
        self._a_max = a_max

        # set by plan()
        self._q0:      float = 0.0
        self._q1:      float = 0.0
        self._sign:    float = 1.0
        self._v_peak:  float = 0.0
        self._t_a:     float = 0.0   # accel phase duration
        self._t_c:     float = 0.0   # cruise phase duration
        self._d_a:     float = 0.0   # distance covered during accel

    # Public API

    def plan(self, q0: float, q1: float) -> float:
        self._q0   = q0
        self._q1   = q1
        d          = q1 - q0
        self._sign = 1.0 if d >= 0 else -1.0
        dist       = abs(d)

        d_a_full = 0.5 * self._v_max ** 2 / self._a_max  # accel dist at full v_max

        if dist >= 2 * d_a_full:
            # trapezoidal: reach v_max
            self._v_peak = self._v_max
            self._t_a    = self._v_max / self._a_max
            self._d_a    = d_a_full
            self._t_c    = (dist - 2 * d_a_full) / self._v_max
        else:
            # triangle: peak velocity limited by distance
            self._v_peak = sqrt(self._a_max * dist)
            self._t_a    = self._v_peak / self._a_max
            self._d_a    = 0.5 * self._v_peak * self._t_a
            self._t_c    = 0.0

        return self.duration

    @property
    def duration(self) -> float:
        return 2 * self._t_a + self._t_c

    def __call__(self, t: float) -> tuple[float, float, float]:
        t = max(0.0, t)
        t_total  = self.duration
        s        = self._sign
        a        = self._a_max
        v        = self._v_peak
        t_cruise = self._t_a + self._t_c

        if t >= t_total:
            return self._q1, 0.0, 0.0

        if t <= self._t_a:
            # accel phase
            pos = self._q0 + s * 0.5 * a * t ** 2
            vel = s * a * t
            acc = s * a

        elif t <= t_cruise:
            # cruise phase
            t2  = t - self._t_a
            pos = self._q0 + s * (self._d_a + v * t2)
            vel = s * v
            acc = 0.0

        else:
            # decel phase
            t3  = t - t_cruise
            pos = self._q0 + s * (self._d_a + v * self._t_c + v * t3 - 0.5 * a * t3 ** 2)
            vel = s * (v - a * t3)
            acc = -s * a

        return pos, vel, acc


class SCurveProfile:

    def __init__(self, *_, **__) -> None:
        raise NotImplementedError("SCurveProfile is not yet implemented")
