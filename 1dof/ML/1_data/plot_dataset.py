import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))


def plot_csv(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    t  = df['time_s'].to_numpy()

    stem    = os.path.splitext(csv_path)[0]
    out_png = stem + '.png'
    title   = os.path.basename(stem)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.06, hspace=0.35)

    # ── panel 1 : force ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, df['f_des_N'],  'b--', lw=1.2, label='f_des')
    ax.plot(t, df['f_raw_N'],  'r-',  lw=0.8, label='f_raw', alpha=0.8)
    ax.set_ylabel('Force [N]')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── panel 2 : torque ──────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, df['tau_cmd_Nm'],  'g--', lw=1.2, label='τ_cmd')
    ax.plot(t, df['tau_meas_Nm'], 'm-',  lw=0.8, label='τ_meas', alpha=0.8)
    ax.set_ylabel('Torque [N·m]')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── panel 3 : pos + vel (dual axis) ──────────────────────────────────────
    ax_vel = axes[2]
    ax_pos = ax_vel.twinx()

    ln1, = ax_vel.plot(t, df['pos_rad'],   'k-',  lw=1.2, label='pos')
    ln2, = ax_pos.plot(t, df['vel_rad_s'], 'c-',  lw=0.9, label='vel', alpha=0.9)

    ax_vel.set_ylabel('Position [rad]',   color='k')
    ax_pos.set_ylabel('Velocity [rad/s]', color='c')
    ax_vel.tick_params(axis='y', labelcolor='k')
    ax_pos.tick_params(axis='y', labelcolor='c')
    ax_vel.grid(True, alpha=0.3)

    lines = [ln1, ln2]
    ax_vel.legend(lines, [l.get_label() for l in lines], loc='upper right', fontsize=8)

    axes[2].set_xlabel('Time [s]')

    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f'  saved  {os.path.relpath(out_png, HERE)}')


def main() -> None:
    pattern = os.path.join(HERE, '**', '*.csv')
    csv_files = sorted(glob.glob(pattern, recursive=True))

    # skip this script's own output if somehow a csv snuck in
    csv_files = [p for p in csv_files if os.path.basename(p) != 'plot_dataset.csv']

    print(f'Found {len(csv_files)} CSV file(s) under {HERE}\n')
    for path in csv_files:
        print(f'  plotting {os.path.relpath(path, HERE)} ...', end=' ', flush=True)
        try:
            plot_csv(path)
        except Exception as e:
            print(f'FAILED: {e}')

    print('\nDone.')


if __name__ == '__main__':
    main()
