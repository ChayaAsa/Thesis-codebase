import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from dynamic import Dynamic

# Inputs

# Joint state
q   = np.array([0.0,  0.78,  -0.78])      # joint positions [rad]
dq  = np.array([0.0,  0.0,  0.0])      # joint velocities [rad/s]
ddq = np.array([0.0,  0.0,  0.0])      # joint accelerations [rad/s²]

# Desired EE force in world frame [N]
F   = np.array([0.0,  0.0,  0.0])      # e.g. 5 N downward in Z


CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dynamic_cache.pkl')

print("Loading dynamic model …")
dyn = Dynamic.get_or_build(CACHE)

# Inverse dynamics: tau = M*ddq + C*dq + G
M, C, G = dyn.evaluate_MCG(q, dq)
tau_dyn = M @ ddq + C @ dq + G

# Jacobian transpose: tau = Jv^T * F
Jv      = dyn.evaluate_jacobian(q)[:3, :]   # 3×3 linear rows
tau_jt  = Jv.T @ F

# EE position
fk = dyn.evaluate_fk(q)
ee  = fk['ee_position']

print(f"\nq   = {np.round(q,   4)} rad")
print(f"dq  = {np.round(dq,  4)} rad/s")
print(f"ddq = {np.round(ddq, 4)} rad/s²")
print(f"F   = {F} N")
print(f"EE  = {np.round(ee,  4)} m")

print(f"\n{'':4s}  {'Joint 1':>12s}  {'Joint 2':>12s}  {'Joint 3':>12s}  [N·m]")
print(f"{'M*ddq+C*dq+G':4s}  {tau_dyn[0]:+12.4f}  {tau_dyn[1]:+12.4f}  {tau_dyn[2]:+12.4f}")
print(f"{'Jv^T * F':4s}  {tau_jt[0]:+12.4f}  {tau_jt[1]:+12.4f}  {tau_jt[2]:+12.4f}")

print(f"\nDiff = {np.round(tau_dyn - tau_jt, 6)} N·m")
print("\nNote: diff is zero when ddq=0, dq=0 (static) and F balances gravity."
      "\n      For motion, ddq and dq contribute dynamic terms not captured by Jv^T*F alone.")
