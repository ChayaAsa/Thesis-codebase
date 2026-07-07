from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import sympy as sp
from sympy import lambdify, pi

import sysid_config as cfg


# ── Shared helper (same as identify_sysid.py) ────────────────────────────────
def _rot(axis: str, a):
    c, s = sp.cos(a), sp.sin(a)
    if axis == 'x':
        return sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])          # z


class SysIDDynamic:

    def __init__(
        self,
        params_json: str | None = None,
        theta: np.ndarray | None = None,
    ) -> None:
        if theta is not None:
            self._theta = np.asarray(theta, dtype=float)
        elif params_json is not None:
            data = json.loads(Path(params_json).read_text())
            self._theta = np.array(list(data['parameters'].values()), dtype=float)
        else:
            raise ValueError("Provide params_json= or theta=.")

        if len(self._theta) != 36:
            raise ValueError(
                f"Expected 36-element θ (30 inertial + 6 friction), got {len(self._theta)}."
            )

        self._M_fn  = None
        self._C_fn  = None
        self._G_fn  = None
        self._J_fn  = None
        self._T_fn: dict = {}   # 1=T01, 2=T02, 3=T_EE, 'elbow'=T03 (for link-3 COM)
        self._built = False

    # Build

    def build(self) -> None:
        q1, q2, q3    = sp.symbols('q1 q2 q3',    real=True)
        dq1, dq2, dq3 = sp.symbols('dq1 dq2 dq3', real=True)
        q  = [q1, q2, q3]
        dq = sp.Matrix([dq1, dq2, dq3])

        L1s, L2s, L3s, Gs = sp.symbols('L1 L2 L3 G', positive=True)
        geom_subs = {L1s: cfg.L1, L2s: cfg.L2, L3s: cfg.L3, Gs: cfg.G}

        # ── Forward kinematics (same convention as identify_sysid.py) ─────────
        z      = sp.Matrix([0, 0, 1])
        R01    = _rot('z', q1)
        p1     = sp.Matrix([0, 0, 0])

        R_pre2 = R01 * _rot('x', -pi / 2)          # fixed Rx(−90°) at shoulder
        p2     = sp.Matrix([0, 0, L1s])             # shoulder origin
        R02    = R_pre2 * _rot('z', q2)

        p3     = p2 + R02 * sp.Matrix([L2s, 0, 0]) # elbow origin
        R03    = R02 * _rot('z', q3)

        p_EE   = p3 + R03 * sp.Matrix([L3s, 0, 0]) # end-effector

        # joint revolute axes in world frame
        a_axis = [z, R_pre2 * z, R02 * z]          # [J1, J2, J3]
        p_o    = [p1, p2, p3]
        R0     = [R01, R02, R03]

        # 6×3 Geometric Jacobian  J = [Jv; Jw]
        J_sym = sp.zeros(6, 3)
        for i in range(3):
            dp = p_EE - p_o[i]
            J_sym[:3, i] = a_axis[i].cross(dp)     # linear-velocity part
            J_sym[3:, i] = a_axis[i]               # angular-velocity part

        # 4×4 homogeneous transforms
        def _T(R, p):
            T = sp.eye(4)
            T[:3, :3] = R
            T[:3, 3]  = p
            return T

        T01  = _T(R01, p1)
        T02  = _T(R02, p2)
        T03  = _T(R03, p3)      # link-3 frame origin (elbow), used for link-3 COM
        T_EE = _T(R03, p_EE)   # end-effector transform (returned as 'T03' to match Dynamic)

        # ── Lagrangian: KE and PE linear in inertial parameters ───────────────
        omega = [
            a_axis[0] * dq1,
            a_axis[0] * dq1 + a_axis[1] * dq2,
            a_axis[0] * dq1 + a_axis[1] * dq2 + a_axis[2] * dq3,
        ]
        grav = sp.Matrix([0, 0, -Gs])

        inertial_syms: list = []
        KE = sp.Integer(0)
        PE = sp.Integer(0)

        for i in range(3):
            m   = sp.Symbol(f'm{i+1}',   real=True)
            hx  = sp.Symbol(f'hx{i+1}',  real=True)
            hy  = sp.Symbol(f'hy{i+1}',  real=True)
            hz  = sp.Symbol(f'hz{i+1}',  real=True)
            Lxx = sp.Symbol(f'Lxx{i+1}', real=True)
            Lyy = sp.Symbol(f'Lyy{i+1}', real=True)
            Lzz = sp.Symbol(f'Lzz{i+1}', real=True)
            Lxy = sp.Symbol(f'Lxy{i+1}', real=True)
            Lxz = sp.Symbol(f'Lxz{i+1}', real=True)
            Lyz = sp.Symbol(f'Lyz{i+1}', real=True)
            inertial_syms += [m, hx, hy, hz, Lxx, Lyy, Lzz, Lxy, Lxz, Lyz]

            h = sp.Matrix([hx, hy, hz])
            I = sp.Matrix([[Lxx, Lxy, Lxz],
                           [Lxy, Lyy, Lyz],
                           [Lxz, Lyz, Lzz]])

            v_o = p_o[i].jacobian(sp.Matrix(q)) * dq
            vL  = R0[i].T * v_o
            wL  = R0[i].T * omega[i]

            KE += (sp.Rational(1, 2) * m * (vL.T * vL)[0]
                   + (vL.T * wL.cross(h))[0]
                   + sp.Rational(1, 2) * (wL.T * (I * wL))[0])
            PE += -(grav.T * (m * p_o[i] + R0[i] * h))[0]

        # M, C, G from the Lagrangian
        M_sym = sp.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                M_sym[i, j] = sp.diff(KE, dq[i], dq[j])

        C_sym = sp.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                cij = sp.Integer(0)
                for k in range(3):
                    cij += sp.Rational(1, 2) * (
                        sp.diff(M_sym[i, j], q[k])
                        + sp.diff(M_sym[i, k], q[j])
                        - sp.diff(M_sym[j, k], q[i])
                    ) * dq[k]
                C_sym[i, j] = cij

        G_sym = sp.Matrix([sp.diff(PE, q[i]) for i in range(3)])

        # ── Substitute identified θ and geometry, then lambdify ───────────────
        theta_in      = self._theta[:30]
        inertial_subs = {inertial_syms[k]: float(theta_in[k]) for k in range(30)}
        all_subs      = {**geom_subs, **inertial_subs}

        q_v  = (q1, q2, q3)
        qd_v = (dq1, dq2, dq3)

        self._M_fn = lambdify(q_v,        M_sym.subs(all_subs), 'numpy', cse=True)
        self._C_fn = lambdify(q_v + qd_v, C_sym.subs(all_subs), 'numpy', cse=True)
        self._G_fn = lambdify(q_v,        G_sym.subs(all_subs), 'numpy', cse=True)
        self._J_fn = lambdify(q_v,        J_sym.subs(geom_subs), 'numpy', cse=True)

        self._T_fn[1]       = lambdify(q_v, T01.subs(geom_subs),  'numpy', cse=True)
        self._T_fn[2]       = lambdify(q_v, T02.subs(geom_subs),  'numpy', cse=True)
        self._T_fn[3]       = lambdify(q_v, T_EE.subs(geom_subs), 'numpy', cse=True)
        self._T_fn['elbow'] = lambdify(q_v, T03.subs(geom_subs),  'numpy', cse=True)

        self._built = True

    # Numerical evaluation

    def evaluate_MCG(
        self, q: np.ndarray, qd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._require_built()
        q, qd = np.asarray(q, float), np.asarray(qd, float)
        M = np.reshape(np.asarray(self._M_fn(*q),      dtype=float), (3, 3))
        C = np.reshape(np.asarray(self._C_fn(*q, *qd), dtype=float), (3, 3))
        G = np.asarray(self._G_fn(*q), dtype=float).ravel()
        return M, C, G

    def evaluate_jacobian(self, q: np.ndarray) -> np.ndarray:
        self._require_built()
        return np.array(self._J_fn(*np.asarray(q, float)), dtype=float)

    def evaluate_inv_jacobian(self, q: np.ndarray) -> np.ndarray:
        return np.linalg.pinv(self.evaluate_jacobian(q))

    def evaluate_fk(self, q: np.ndarray) -> dict:
        self._require_built()
        q  = np.asarray(q, float)
        T1 = np.array(self._T_fn[1](*q),       dtype=float)   # T01
        T2 = np.array(self._T_fn[2](*q),       dtype=float)   # T02
        T3 = np.array(self._T_fn[3](*q),       dtype=float)   # T_EE (≡ "T03" externally)
        Te = np.array(self._T_fn['elbow'](*q), dtype=float)   # T03 (elbow frame for COM)

        # link COMs in base frame:  p_o_i + R0_i * (h_i / m_i)
        th     = self._theta
        T_from = [np.eye(4), T1, Te]          # base frames for links 1, 2, 3
        coms   = []
        for i in range(3):
            m_i  = float(th[10 * i])
            h_i  = th[10 * i + 1: 10 * i + 4]
            c_lf = h_i / m_i if abs(m_i) > 1e-9 else np.zeros(3)
            coms.append((T_from[i] @ np.r_[c_lf, 1.])[:3])

        return {
            'T01': T1, 'T02': T2, 'T03': T3,
            'joints':      [np.zeros(3), T1[:3, 3], T2[:3, 3], T3[:3, 3]],
            'joint_axes':  [np.array([0., 0., 1.]), T2[:3, 2].copy(), T3[:3, 2].copy()],
            'coms':        coms,
            'ee_position': T3[:3, 3].copy(),
            'ee_pose':     T3,
        }

    # Persistence

    def save(self, path: str = 'sid_cache.pkl') -> None:
        import dill
        with open(path, 'wb') as f:
            dill.dump(self, f)
        print(f"SysIDDynamic saved → {path}")

    @classmethod
    def load(cls, path: str = 'sid_cache.pkl') -> 'SysIDDynamic':
        import dill
        with open(path, 'rb') as f:
            obj = dill.load(f)
        print(f"SysIDDynamic loaded ← {path}")
        return obj

    @classmethod
    def get_or_build(
        cls,
        cache_path: str = 'sid_cache.pkl',
        params_json: str | None = None,
        theta: np.ndarray | None = None,
    ) -> 'SysIDDynamic':
        if os.path.exists(cache_path):
            return cls.load(cache_path)

        # auto-discover newest params JSON if not given
        if params_json is None and theta is None:
            cache_dir  = Path(cache_path).parent
            candidates = sorted((cache_dir / 'data').glob('*.params.json'))
            if not candidates:
                candidates = sorted(cache_dir.glob('*.params.json'))
            if not candidates:
                raise FileNotFoundError(
                    f"No cache at '{cache_path}' and no *.params.json found in "
                    f"'{cache_dir / 'data'}' or '{cache_dir}'.\n"
                    "Run identify_sysid.py first, or pass params_json= explicitly."
                )
            params_json = str(candidates[-1])
            print(f"Auto-selected params: {params_json}")

        print("Cache not found — building (~40 s) …", flush=True)
        sid = cls(params_json=params_json, theta=theta)
        sid.build()
        sid.save(cache_path)
        return sid

    # Internal

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError(
                "Call build() (or get_or_build()) before calling evaluate_*."
            )


# Quick smoke-test
if __name__ == '__main__':
    import sys
    import glob

    # find a params JSON to test with
    data_dir = Path(__file__).parent / 'data'
    jsons    = sorted(data_dir.glob('*.params.json'))
    if not jsons:
        print("No *.params.json found in ./data — run identify_sysid.py first.")
        sys.exit(1)

    print(f"Loading params from: {jsons[-1]}")
    sid = SysIDDynamic(params_json=str(jsons[-1]))

    print("Building (this takes ~40 s) …")
    sid.build()

    q0  = np.array([0.0, 0.78, -0.78])
    qd0 = np.zeros(3)

    M, C, G = sid.evaluate_MCG(q0, qd0)
    print(f"\nM(q0) =\n{M}")
    print(f"C(q0,0) =\n{C}")
    print(f"G(q0) = {G}")

    J  = sid.evaluate_jacobian(q0)
    print(f"\nJ(q0) [6×3] =\n{J}")

    fk = sid.evaluate_fk(q0)
    print(f"\nEE position = {fk['ee_position']}")
    print(f"joints      = {[np.round(p, 4) for p in fk['joints']]}")
