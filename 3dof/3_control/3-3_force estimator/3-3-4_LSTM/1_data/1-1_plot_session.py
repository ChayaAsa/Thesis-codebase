from __future__ import annotations

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# colour palette (matches live plotter)
_C = ['steelblue', 'firebrick', 'goldenrod']   # joint 1, 2, 3
_ALPHA_DES  = 0.85
_ALPHA_MEAS = 0.65
_LW_DES     = 1.6
_LW_MEAS    = 1.1

# helpers

def _stat_str(arr: np.ndarray) -> str:
    return f"min {arr.min():+.3f}  mean {arr.mean():+.3f}  max {arr.max():+.3f}"


def _session_title(df: pd.DataFrame, csv_path: str) -> str:
    phase = df['phase'].iloc[0] if 'phase' in df.columns else '?'
    t_total = df['time_s'].iloc[-1] - df['time_s'].iloc[0]
    n = len(df)
    dt_med = np.median(np.diff(df['time_s'].values)) * 1000  # ms
    name = os.path.splitext(os.path.basename(csv_path))[0]
    return (
        f"Phase {phase}   |   {name}\n"
        f"Duration {t_total:.1f} s   {n} samples   "
        f"median Δt {dt_med:.1f} ms  ({1000/dt_med:.0f} Hz)"
    )


def _ax_stats(ax, lines_data: list[tuple[str, np.ndarray]]) -> None:
    parts = [f"{lbl}: |max|={np.abs(arr).max():.3f}" for lbl, arr in lines_data]
    ax.text(0.01, 0.97, "   ".join(parts),
            transform=ax.transAxes, fontsize=6.5, va='top',
            color='# 444444',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))


# main plot function

def plot_session(csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    t  = df['time_s'].values

    phase = df['phase'].iloc[0] if 'phase' in df.columns else '?'
    sup   = _session_title(df, csv_path)

    fig = plt.figure(figsize=(14, 18), dpi=120)
    fig.suptitle(sup, fontsize=11, fontweight='bold', y=0.995)

    gs = gridspec.GridSpec(5, 1, figure=fig,
                           hspace=0.55, left=0.07, right=0.93,
                           top=0.965, bottom=0.04)

    # 1  f_push
    ax1 = fig.add_subplot(gs[0])
    fpush = df['f_push_N'].values
    ax1.plot(t, fpush, color='black', lw=1.4, label='f_push_N')
    ax1.axhline(0, color='grey', lw=0.5, ls='--')
    ax1.set_ylabel('Force [N]')
    ax1.set_title(
        f'① f_push  (−ATI Fz)   phase={phase}   '
        f'{_stat_str(fpush)}',
        fontsize=8)
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, lw=0.3)

    # 2  q  vs  q_des
    ax2 = fig.add_subplot(gs[1])
    stat_data = []
    for i in range(3):
        qm = df[f'q{i+1}_rad'].values
        qd = df[f'q_des{i+1}_rad'].values
        ax2.plot(t, qm, color=_C[i], lw=_LW_MEAS, alpha=_ALPHA_MEAS, label=f'q{i+1}')
        ax2.plot(t, qd, color=_C[i], lw=_LW_DES,  alpha=_ALPHA_DES,  ls='--',
                 label=f'q{i+1}_des')
        stat_data.append((f'q{i+1}', qm))
    ax2.set_ylabel('Angle [rad]')
    ax2.set_title('② Joint position  q (solid) vs q_des (dashed)', fontsize=8)
    _ax_stats(ax2, stat_data)
    ax2.legend(fontsize=7, ncol=6, loc='upper right')
    ax2.grid(True, lw=0.3)

    # 3  qdot  vs  qdot_des
    ax3 = fig.add_subplot(gs[2])
    stat_data = []
    for i in range(3):
        qdm  = df[f'qdot{i+1}_rad_s'].values
        qddes = df[f'qdot_des{i+1}_rad_s'].values
        ax3.plot(t, qdm,   color=_C[i], lw=_LW_MEAS, alpha=_ALPHA_MEAS, label=f'qdot{i+1}')
        ax3.plot(t, qddes, color=_C[i], lw=_LW_DES,  alpha=_ALPHA_DES,  ls='--',
                 label=f'qdot{i+1}_des')
        stat_data.append((f'qdot{i+1}', qdm))
    ax3.set_ylabel('Vel [rad/s]')
    ax3.set_title('③ Joint velocity  qdot (solid) vs qdot_des (dashed)', fontsize=8)
    _ax_stats(ax3, stat_data)
    ax3.legend(fontsize=7, ncol=6, loc='upper right')
    ax3.grid(True, lw=0.3)

    # 4  tau_cmd  vs  tau_meas
    ax4 = fig.add_subplot(gs[3])
    stat_data = []
    for i in range(3):
        tm  = df[f'tau_meas{i+1}_Nm'].values
        tc  = df[f'tau_cmd{i+1}_Nm'].values
        ax4.plot(t, tm, color=_C[i], lw=_LW_MEAS, alpha=_ALPHA_MEAS, label=f'τ_meas{i+1}')
        ax4.plot(t, tc, color=_C[i], lw=_LW_DES,  alpha=_ALPHA_DES,  ls='--',
                 label=f'τ_cmd{i+1}')
        stat_data.append((f'τ_meas{i+1}', tm))
    ax4.axhline(0, color='grey', lw=0.5, ls='--')
    ax4.set_ylabel('Torque [N·m]')
    ax4.set_title('④ Torque  τ_meas (solid) vs τ_cmd (dashed)', fontsize=8)
    _ax_stats(ax4, stat_data)
    ax4.legend(fontsize=7, ncol=6, loc='upper right')
    ax4.grid(True, lw=0.3)

    # ── 5  ATI  dual-axis (Fx Fy Fz | Mx My Mz) ──────────────────────────────
    ax5l = fig.add_subplot(gs[4])
    ax5r = ax5l.twinx()

    force_cols   = [('ati_fx_N', 'Fx'), ('ati_fy_N', 'Fy'), ('ati_fz_N', 'Fz')]
    moment_cols  = [('ati_mx_Nm', 'Mx'), ('ati_my_Nm', 'My'), ('ati_mz_Nm', 'Mz')]
    force_colors = ['tab:blue', 'tab:orange', 'tab:green']
    mom_colors   = ['tab:red',  'tab:purple', 'tab:brown']

    for (col, lbl), col_c in zip(force_cols, force_colors):
        ax5l.plot(t, df[col].values, color=col_c, lw=1.1, label=lbl)
    for (col, lbl), col_c in zip(moment_cols, mom_colors):
        ax5r.plot(t, df[col].values, color=col_c, lw=1.1, ls='--', label=lbl)

    ax5l.axhline(0, color='grey', lw=0.4, ls='--')
    ax5l.set_ylabel('Force [N]',    color='tab:blue')
    ax5r.set_ylabel('Moment [N·m]', color='tab:red')
    ax5l.tick_params(axis='y', labelcolor='tab:blue')
    ax5r.tick_params(axis='y', labelcolor='tab:red')

    # combined legend
    h_l, lb_l = ax5l.get_legend_handles_labels()
    h_r, lb_r = ax5r.get_legend_handles_labels()
    ax5l.legend(h_l + h_r, lb_l + lb_r, fontsize=7, ncol=6, loc='upper right')

    fx_max = max(abs(df[c].values).max() for c, _ in force_cols)
    mx_max = max(abs(df[c].values).max() for c, _ in moment_cols)
    ax5l.set_title(
        f'⑤ ATI 6-axis   |F|max={fx_max:.2f} N   |M|max={mx_max:.3f} N·m   '
        f'(left=forces, right=moments, dashed)',
        fontsize=8)
    ax5l.grid(True, lw=0.3)
    ax5l.set_xlabel('Time [s]')

    # save
    out_path = os.path.splitext(csv_path)[0] + '.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved -> {out_path}")
    return out_path


# CLI

def _find_csvs_here() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    return sorted(glob.glob(os.path.join(here, 'train_phase*.csv')))


def main() -> None:
    p = argparse.ArgumentParser(description='Export 5-panel PNG from training CSVs.')
    p.add_argument('csv', nargs='?', help='Path to CSV (default: newest in this dir)')
    p.add_argument('--all', action='store_true', help='Export PNG for every CSV here')
    args = p.parse_args()

    if args.all:
        csvs = _find_csvs_here()
        if not csvs:
            sys.exit('No train_phase*.csv found in this directory.')
        print(f"Exporting {len(csvs)} file(s) …")
        for c in csvs:
            plot_session(c)
    elif args.csv:
        plot_session(os.path.abspath(args.csv))
    else:
        csvs = _find_csvs_here()
        if not csvs:
            sys.exit('No train_phase*.csv found — pass a path explicitly.')
        newest = max(csvs, key=os.path.getmtime)
        print(f"Auto-selected: {os.path.basename(newest)}")
        plot_session(newest)


if __name__ == '__main__':
    main()
