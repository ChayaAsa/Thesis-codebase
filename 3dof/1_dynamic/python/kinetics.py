from sympy import symbols, zeros, Matrix, S

from kinematics import Kinematics


class Kinetics:

    def __init__(self, kin: Kinematics) -> None:
        self.kin = kin

        # Inertial symbols
        self.m1, self.m2, self.m3 = symbols('m1 m2 m3', positive=True)
        self.m = [self.m1, self.m2, self.m3]

        self.qd1, self.qd2, self.qd3 = symbols('qd1 qd2 qd3', real=True)
        self.qd = Matrix([self.qd1, self.qd2, self.qd3])
        self.g = symbols('g', positive=True)

        # Populated by derive()
        self.M: Matrix | None = None
        self.C: Matrix | None = None
        self.G: Matrix | None = None
        self._derived = False

    # Symbolic derivation

    def derive(self) -> 'Kinetics':
        if self._derived:
            return self

        if not self.kin._derived:
            raise RuntimeError("Call Kinematics.derive() before Kinetics.derive().")

        q   = self.kin.q
        p_c = self.kin.p_c    # list of 3 sympy 3×1 matrices (base-frame COMs)
        m   = self.m
        qd  = self.qd
        g   = self.g
        n   = 3

        # Linear velocity Jacobians  Jv[i] = ∂p_c[i]/∂q   (3×n)
        Jv = [p_c[i].jacobian(q) for i in range(n)]

        # ── Mass matrix  M = Σ m_i · Jv_i^T · Jv_i ─────────────────────────
        # No simplify here — expressions are simplified after numerical subs
        # in Dynamic.build(), where they collapse to q-only trig.
        M = zeros(n)
        for i in range(n):
            M += m[i] * (Jv[i].T * Jv[i])

        # Coriolis via Christoffel symbols
        # C_{kj} = Σ_i  c_{ijk} · qd_i
        # c_{ijk} = ½(∂M_{kj}/∂q_i + ∂M_{ki}/∂q_j − ∂M_{ij}/∂q_k)
        dM = [M.diff(q[l]) for l in range(n)]
        C = zeros(n)
        for k in range(n):
            for j in range(n):
                s = S.Zero
                for i in range(n):
                    c_ijk = (dM[i][k, j] + dM[j][k, i] - dM[k][i, j]) / 2
                    s += c_ijk * qd[i]
                C[k, j] = s

        # ── Gravity  G = ∂V/∂q,   V = Σ m_i · g · z_i ──────────────────────
        V = S.Zero
        for i in range(n):
            V += m[i] * g * p_c[i][2]
        G = Matrix([V.diff(q[i]) for i in range(n)])

        self.M = M
        self.C = C
        self.G = G
        self._derived = True
        return self
