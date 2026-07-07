import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from param import robot_params 
from dynamic import Dynamic


# Cylinder surface (plot_surface compatible)
def _cyl_surf(p1, p2, r, n=18):
    v = np.asarray(p2, float) - np.asarray(p1, float)
    L = np.linalg.norm(v)
    if L < 1e-9:
        return None
    v /= L
    ref = np.array([1.,0.,0.]) if abs(v[0]) < 0.9 else np.array([0.,1.,0.])
    e1 = np.cross(v, ref);  e1 /= np.linalg.norm(e1)
    e2 = np.cross(v, e1)

    t, s = np.meshgrid(np.linspace(0, 2*np.pi, n), np.array([0., 1.]))
    ct, st_ = np.cos(t), np.sin(t)
    P = np.asarray(p1, float)
    X = P[0] + v[0]*s*L + r*(ct*e1[0] + st_*e2[0])
    Y = P[1] + v[1]*s*L + r*(ct*e1[1] + st_*e2[1])
    Z = P[2] + v[2]*s*L + r*(ct*e1[2] + st_*e2[2])
    return X, Y, Z


# Circular torque arrow
def _torque_arrow(ax, center, axis, radius, tau, color):
    if abs(tau) < 3e-4:
        return
    ax_n = np.asarray(axis, float); ax_n /= np.linalg.norm(ax_n) + 1e-12
    ref  = np.array([1.,0.,0.]) if abs(ax_n[0]) < 0.9 else np.array([0.,1.,0.])
    e1   = np.cross(ax_n, ref);  e1 /= np.linalg.norm(e1)
    e2   = np.cross(ax_n, e1)

    span = 1.45 * np.pi * np.sign(tau)
    t    = np.linspace(0., span, 45)
    c    = np.asarray(center, float)
    arc  = c + radius * (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2))
    ax.plot(arc[:,0], arc[:,1], arc[:,2], color=color, lw=2.2, zorder=9)

    tip  = arc[-1]
    tang = arc[-1] - arc[-2]; tang /= np.linalg.norm(tang) + 1e-12
    ax.quiver(*tip, *(tang * radius * 0.45), color=color, arrow_length_ratio=0.9, lw=1.5)


# Bar indicator
def _bar(v, scale=0.45, width=10):
    n = min(width, int(abs(v) / scale))
    if n == 0:
        return '·'
    sym = '▶' if v > 0 else '◀'
    return sym * n

def _tau_col(v):
    if abs(v) < 0.005: return '# 555577'
    return '# ff6b6b' if v > 0 else '#6b9fff'


# ── Build MCG numerical functions (sympy – takes ~20 s) ───────────────────────
p    = robot_params()
_dyn = Dynamic.get_or_build('dynamic_cache.pkl', p)
MCG  = _dyn.evaluate_MCG


# ── Figure & layout ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), facecolor='# 12131a')
fig.canvas.manager.set_window_title('3-DOF RRR — MCG Dynamics Visualizer')

ax3d = fig.add_axes([0.01, 0.14, 0.62, 0.84], projection='3d')
ax3d.set_facecolor('# 1a1c2c')

ax_tx = fig.add_axes([0.64, 0.02, 0.35, 0.96])
ax_tx.set_facecolor('# 12131a'); ax_tx.axis('off')

# Sliders (bottom left, under the 3-D view)
q0 = p['q0'].copy()
sl_axes = [
    fig.add_axes([0.06, 0.09, 0.55, 0.023]),
    fig.add_axes([0.06, 0.06, 0.55, 0.023]),
    fig.add_axes([0.06, 0.03, 0.55, 0.023]),
]
sliders = [
    Slider(sl_axes[0], 'q1  yaw  ', -np.pi,   np.pi,   valinit=q0[0], color='# 4ecdc4'),
    Slider(sl_axes[1], 'q2  pitch', -np.pi/2,  np.pi,   valinit=q0[1], color='# 45b7d1'),
    Slider(sl_axes[2], 'q3  elbow', -np.pi,    np.pi/2, valinit=q0[2], color='# 96ceb4'),
]
for sl in sliders:
    sl.label.set_color('white'); sl.valtext.set_color('white')
    sl.ax.set_facecolor('# 1a1c2c')

LINK_COL  = ['# 4ecdc4', '#45b7d1', '#96ceb4']
JOINT_COL = '# ffd93d'
TIP_COL   = '# ff6b9d'
COM_COL   = '# ff9ff3'
LINK_RAD  = [0.021, 0.016, 0.013]


# Update callback
def update(_=None):
    q  = np.array([sl.val for sl in sliders])
    qd = np.zeros(3)
    M, C, G = MCG(q, qd)
    tau = G.ravel()   # static holding torque (qd=0, qdd=0 → τ = G)

    fk_out = _dyn.evaluate_fk(q)
    joints = fk_out['joints']
    axes   = fk_out['joint_axes']
    T01, T02, T03 = fk_out['T01'], fk_out['T02'], fk_out['T03']
    # COMs consistent with kinematics.p_c:  T0i @ rci  (each ci in its own frame i)
    coms = [
        (T01 @ np.r_[p['c1'], 1.])[:3],
        (T02 @ np.r_[p['c2'], 1.])[:3],
        (T03 @ np.r_[p['c3'], 1.])[:3],
    ]
    p3 = fk_out['ee_position']

    # 3-D scene
    ax3d.cla()
    ax3d.set_facecolor('# 1a1c2c')

    # Cylinder links
    for i, (a, b, r, col) in enumerate(zip(joints, joints[1:], LINK_RAD, LINK_COL)):
        surf = _cyl_surf(a, b, r)
        if surf is not None:
            ax3d.plot_surface(*surf, color=col, alpha=0.84, linewidth=0, antialiased=True)

    # Joint spheres
    for pt in joints[:3]:
        ax3d.scatter(*pt, s=90,  c=JOINT_COL, depthshade=False, zorder=6)
    ax3d.scatter(*p3, s=120, c=TIP_COL, marker='D', depthshade=False, zorder=7)
    ax3d.text(p3[0]+.005, p3[1]+.005, p3[2]+.012, 'Tip', color=TIP_COL, fontsize=8)

    # COM markers
    for i, com in enumerate(coms):
        ax3d.scatter(*com, s=80, c=COM_COL, marker='*', depthshade=False, zorder=8)
        ax3d.text(com[0]+.006, com[1]+.006, com[2]+.010,
                  f'c{i+1}', color=COM_COL, fontsize=8)

    # Circular torque arrows at each joint
    arrow_r = [0.040, 0.033, 0.027]
    for pt, axis, ti, ar in zip(joints[:3], axes, tau, arrow_r):
        _torque_arrow(ax3d, pt, axis, ar, ti, _tau_col(ti))

    # Ground grid
    g = 0.34
    gx, gy = np.meshgrid(np.linspace(-g, g, 7), np.linspace(-g, g, 7))
    ax3d.plot_wireframe(gx, gy, np.zeros_like(gx), color='# 2a2a4a', alpha=0.45, lw=0.6)

    lim = 0.40
    ax3d.set_xlim(-lim, lim); ax3d.set_ylim(-lim, lim); ax3d.set_zlim(0, lim*2.2)
    ax3d.set_xlabel('X', color='# 999'); ax3d.set_ylabel('Y', color='#999')
    ax3d.set_zlabel('Z', color='# 999')
    ax3d.tick_params(colors='# 777', labelsize=7)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor('# 2a2a4a')
    ax3d.set_title('3-DOF RRR  ·  Cylinder Links  ·  MCG Dynamics',
                   color='# cccccc', fontsize=10, pad=4)

    # Text panel
    ax_tx.cla(); ax_tx.axis('off'); ax_tx.set_facecolor('# 12131a')

    def T(text, y, col='# cccccc', sz=8.5, bold=False):
        ax_tx.text(0.04, y, text, transform=ax_tx.transAxes,
                   color=col, fontsize=sz, va='top',
                   fontweight='bold' if bold else 'normal',
                   fontfamily='monospace')

    y = 0.987; dy = 0.034

    T('MCG  DYNAMICS  PANEL', y, '# ffd93d', 11, True); y -= dy*1.3
    q_str = '  '.join(f'q{i+1}={np.rad2deg(q[i]):+6.1f}°' for i in range(3))
    T(q_str, y, '# aaaaff', 8.5); y -= dy*1.3

    # ── Static torque = G ─────────────────────────────────────────────────
    T('── Static Torque  τ = G(q) ──', y, '# ffd93d', 9, True); y -= dy
    labels = ['τ₁ yaw  ', 'τ₂ pitch', 'τ₃ elbow']
    for i in range(3):
        col = _tau_col(tau[i])
        T(f'  {labels[i]}: {tau[i]:+8.4f} Nm  {_bar(tau[i])}', y, col, 8.5)
        y -= dy
    y -= dy*0.3

    # Mass matrix
    T('── Mass Matrix  M(q) ──', y, '# ffd93d', 9, True); y -= dy
    for row in M:
        T(f'  [{row[0]:+7.4f}  {row[1]:+7.4f}  {row[2]:+7.4f}]', y, '# 88ccff', 8.5)
        y -= dy
    y -= dy*0.3

    # Coriolis
    T('── Coriolis  C(q, q̇=0) ──', y, '# ffd93d', 9, True); y -= dy
    for row in C:
        T(f'  [{row[0]:+7.4f}  {row[1]:+7.4f}  {row[2]:+7.4f}]', y, '# 88ffcc', 8.5)
        y -= dy
    y -= dy*0.3

    # Gravity vector
    T('── Gravity Vector  G(q) ──', y, '# ffd93d', 9, True); y -= dy
    T(f'  G = [{G[0]:+8.5f},', y, '# ffcc88', 8.5); y -= dy
    T(f'       {G[1]:+8.5f},', y, '# ffcc88', 8.5); y -= dy
    T(f'       {G[2]:+8.5f}]', y, '# ffcc88', 8.5); y -= dy
    y -= dy*0.3

    # COM positions
    T('── COM Positions (base frame) ──', y, '# ffd93d', 9, True); y -= dy
    masses = [p['m1'], p['m2'], p['m3']]
    for i, (com, mv) in enumerate(zip(coms, masses), 1):
        T(f'  c{i} ({mv:.2f} kg)  [{com[0]:+5.3f}, {com[1]:+5.3f}, {com[2]:+5.3f}]',
          y, COM_COL, 8.5)
        y -= dy
    y -= dy*0.3

    # Tip position
    T('── Tip Position ──', y, '# ffd93d', 9, True); y -= dy
    T(f'  [{p3[0]:+6.3f},  {p3[1]:+6.3f},  {p3[2]:+6.3f}] m', y, TIP_COL, 8.5)

    fig.canvas.draw_idle()


for sl in sliders:
    sl.on_changed(update)

update()

fig.text(0.32, 0.005,
         'Drag sliders ← → to rotate joints  ·  Colored circular arrows = required torque direction',
         color='# 555577', ha='center', fontsize=8)

plt.show()
