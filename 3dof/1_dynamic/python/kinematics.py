from sympy import symbols, cos, sin, pi, eye, simplify, Matrix


class Kinematics:

    def __init__(self) -> None:
        # Joint angles
        self.q1, self.q2, self.q3 = symbols('q1 q2 q3', real=True)
        self.q = Matrix([self.q1, self.q2, self.q3])

        # Link lengths
        self.l1, self.l2, self.l3 = symbols('l1 l2 l3', positive=True)

        # COM offsets — link i's COM expressed in frame (i-1)
        # link 1 → base frame, link 2 → frame 1, link 3 → frame 2
        self.rc1x, self.rc1y, self.rc1z = symbols('rc1x rc1y rc1z', real=True)
        self.rc2x, self.rc2y, self.rc2z = symbols('rc2x rc2y rc2z', real=True)
        self.rc3x, self.rc3y, self.rc3z = symbols('rc3x rc3y rc3z', real=True)

        # Populated by derive()
        self.T:   dict         = {}
        self.p_c: list         = []
        self.J:   Matrix | None = None
        self._derived = False

    # Symbolic derivation
    def _dh_sym(self, a, alpha, d, theta) -> Matrix:
        return Matrix([
            [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
            [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
            [0,           sin(alpha),             cos(alpha),            d           ],
            [0,           0,                      0,                     1           ],
        ])

    def derive(self) -> 'Kinematics':
        if self._derived:
            return self

        l1, l2, l3 = self.l1, self.l2, self.l3
        q1, q2, q3 = self.q1, self.q2, self.q3

        # Homogeneous transforms
        T01 = self._dh_sym(0,  pi/2, l1, q1)
        T12 = self._dh_sym(l2, 0,    0, q2)
        T23 = self._dh_sym(l3, 0,    0, q3)

        T02 = simplify(T01 * T12)
        T03 = simplify(T02 * T23)

        self.T = {0: eye(4), 1: T01, 2: T02, 3: T03}

        # COM positions in base frame
        rc = [
            Matrix([self.rc1x, self.rc1y, self.rc1z, 1]),
            Matrix([self.rc2x, self.rc2y, self.rc2z, 1]),
            Matrix([self.rc3x, self.rc3y, self.rc3z, 1]),
        ]
        # T_from[i] maps link i's COM (expressed in frame i-1) to base frame
        T_from = [T01, T02, T03]
        self.p_c = [simplify((T_from[i] * rc[i])[:3, :]) for i in range(3)]

        # ── Geometric Jacobian (6×3) ─────────────────────────────────────────
        # Revolute joint i:  Jv_i = z_{i-1} × (p_EE − p_{i-1}),  Jw_i = z_{i-1}
        p_ee     = T03[:3, 3]
        origins = [Matrix([0, 0, 0]), T01[:3, 3], T02[:3, 3]]
        axes    = [Matrix([0, 0, 1]), T01[:3, 2], T02[:3, 2]]

        Jv_cols = [axes[i].cross(p_ee - origins[i]) for i in range(3)]
        Jw_cols = axes

        Jv = Matrix.hstack(*Jv_cols)
        Jw = Matrix.hstack(*Jw_cols)
        self.J = simplify(Matrix.vstack(Jv, Jw))   # 6×3

        self._derived = True
        return self
