import numpy as np
from sympy import simplify, lambdify

from kinematics import Kinematics
from kinetics   import Kinetics
from param      import robot_params


class Dynamic:

    def __init__(self, params: dict | None = None) -> None:
        self.params     = params if params is not None else robot_params()
        self.kinematics = Kinematics()
        self.kinetics   = Kinetics(self.kinematics)

        self._T_fn: dict = {}   # _T_fn[i](q1,q2,q3) → 4×4
        self._J_fn       = None
        self._M_fn       = None
        self._C_fn       = None
        self._G_fn       = None
        self._built      = False

    # Build

    def build(self) -> None:
        self.kinematics.derive()
        self.kinetics.derive()

        kin = self.kinematics
        dyn = self.kinetics
        p   = self.params

        # Combined substitution: geometry + inertia
        subs = {
            kin.l1: p['L1'], kin.l2: p['L2'], kin.l3: p['L3'],
            kin.rc1x: p['c1'][0], kin.rc1y: p['c1'][1], kin.rc1z: p['c1'][2],
            kin.rc2x: p['c2'][0], kin.rc2y: p['c2'][1], kin.rc2z: p['c2'][2],
            kin.rc3x: p['c3'][0], kin.rc3y: p['c3'][1], kin.rc3z: p['c3'][2],
            dyn.m1: p['m1'], dyn.m2: p['m2'], dyn.m3: p['m3'],
            dyn.g:  p['g'],
        }

        q_vars  = list(kin.q)            # [q1, q2, q3]
        qd_vars = list(dyn.qd)           # [qd1, qd2, qd3]
        all_vars = q_vars + qd_vars      # [q1..q3, qd1..qd3]

        # Transforms
        for i in [1, 2, 3]:
            T_num = simplify(kin.T[i].subs(subs))
            self._T_fn[i] = lambdify(q_vars, T_num, 'numpy')

        # Jacobian
        J_num = simplify(kin.J.subs(subs))
        self._J_fn = lambdify(q_vars, J_num, 'numpy')

        # Kinetics
        self._M_fn = lambdify(all_vars, simplify(dyn.M.subs(subs)), 'numpy')
        self._C_fn = lambdify(all_vars, simplify(dyn.C.subs(subs)), 'numpy')
        self._G_fn = lambdify(q_vars,   simplify(dyn.G.subs(subs)), 'numpy')

        self._built = True

    # Numerical evaluation

    def evaluate_fk(self, q: np.ndarray) -> dict:
        self._require_built()
        T1 = np.array(self._T_fn[1](*q), dtype=float)
        T2 = np.array(self._T_fn[2](*q), dtype=float)
        T3 = np.array(self._T_fn[3](*q), dtype=float)

        p = self.params
        T_from = [np.eye(4), T1, T2]
        coms = [(T_from[i] @ np.r_[p[f'c{i+1}'], 1.])[:3] for i in range(3)]

        return {
            'T01': T1, 'T02': T2, 'T03': T3,
            'joints':      [np.zeros(3), T1[:3, 3], T2[:3, 3], T3[:3, 3]],
            'joint_axes':  [np.array([0., 0., 1.]), T1[:3, 2], T2[:3, 2]],
            'coms':        coms,
            'ee_position': T3[:3, 3].copy(),
            'ee_pose':     T3,
        }

    def evaluate_jacobian(self, q: np.ndarray) -> np.ndarray:
        self._require_built()
        return np.array(self._J_fn(*q), dtype=float)

    def evaluate_inv_jacobian(self, q: np.ndarray) -> np.ndarray:
        return np.linalg.pinv(self.evaluate_jacobian(q))

    def evaluate_MCG(
        self, q: np.ndarray, qd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._require_built()
        args = (*q, *qd)
        M = np.array(self._M_fn(*args), dtype=float)
        C = np.array(self._C_fn(*args), dtype=float)
        G = np.array(self._G_fn(*q),    dtype=float).ravel()
        return M, C, G

    # Persistence

    def save(self, path: str = 'dynamic_cache.pkl') -> None:
        import dill
        with open(path, 'wb') as f:
            dill.dump(self, f)
        print(f"Dynamic saved → {path}")

    @classmethod
    def load(cls, path: str = 'dynamic_cache.pkl') -> 'Dynamic':
        import dill
        with open(path, 'rb') as f:
            obj = dill.load(f)
        print(f"Dynamic loaded ← {path}")
        return obj

    @classmethod
    def get_or_build(
        cls,
        cache_path: str = 'dynamic_cache.pkl',
        params: dict | None = None,
    ) -> 'Dynamic':
        import os
        if os.path.exists(cache_path):
            return cls.load(cache_path)
        print(f"Cache not found — building (~20 s) …", flush=True)
        dyn = cls(params)
        dyn.build()
        dyn.save(cache_path)
        return dyn

    # Internal

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError("Call build() before evaluating.")


# Quick smoke-test

if __name__ == '__main__':
    dyn = Dynamic()
    p   = dyn.params
    q0  = p['q0']

    print("Deriving symbolic expressions ...")
    dyn.kinematics.derive()
    print("T02 (symbolic):\n", dyn.kinematics.T[2])
    print("J  (symbolic):\n",  dyn.kinematics.J)
    print("M  (symbolic):\n",  dyn.kinetics.derive().M)

    print("\nBuilding numerical functions ...")
    dyn.build()

    fk = dyn.evaluate_fk(q0)
    print("\nFK at q0:")
    for i, j in enumerate(fk['joints']):
        print(f"  p{i} = {j}")
    print("  EE =", fk['ee_position'])

    J = dyn.evaluate_jacobian(q0)
    print("\nJ(q0) =\n", J)

    J_inv = dyn.evaluate_inv_jacobian(q0)
    print("\nJ†(q0) =\n", J_inv)

    M, C, G = dyn.evaluate_MCG(q0, np.zeros(3))
    print("\nM(q0) =\n",    M)
    print("C(q0, 0) =\n", C)
    print("G(q0) =\n",    G)
    dyn.save('dynamic_cache.pkl')
