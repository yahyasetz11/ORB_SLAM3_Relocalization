"""
Interactive benchmark: manually align Our Pipeline + ORB-SLAM3 Localization to GT.

No Umeyama. You drive the translation, rotation (ZYX-Euler degrees), and scale of
each trajectory with sliders, and the top-down trajectory + ATE-over-time plots
update live. ATE RMSE is shown in each legend.

Usage:
    python3 plot_benchmark_2.py
    python3 plot_benchmark_2.py --ours comparison_log.csv --orb localization_test_log.csv \
                                --gt data/.../groundtruth.txt
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider, Button


GROUNDTRUTH_DEFAULT = "data/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt"
OURS_DEFAULT = "comparison_log.csv"
ORB_DEFAULT = "localization_test_log.csv"


# ── Loaders (same as plot_benchmark.py) ──────────────────────────────────────
def load_groundtruth(path):
    gt = pd.read_csv(path, sep=r"\s+", comment="#",
                     names=["t", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
    return gt.sort_values("t").reset_index(drop=True)


def load_ours(path):
    df = pd.read_csv(path)
    df = df[df["is_localized"] == 1].copy()
    df = df.rename(columns={
        "timestamp": "t",
        "result_tx": "tx", "result_ty": "ty", "result_tz": "tz",
    })
    return df[["t", "tx", "ty", "tz"]].sort_values("t").reset_index(drop=True)


def load_orb(path):
    df = pd.read_csv(path)
    df = df[df["is_localized"] == 1].copy()
    df["t"] = df["timestamp"]
    return df[["t", "tx", "ty", "tz"]].sort_values("t").reset_index(drop=True)


def match_to_gt(df, gt, max_diff):
    """Return (est_xyz, gt_xyz, est_t) for frames matched within max_diff seconds."""
    gt_ts = gt["t"].to_numpy()
    gt_xyz = gt[["tx", "ty", "tz"]].to_numpy()
    est_t = df["t"].to_numpy()
    est_xyz = df[["tx", "ty", "tz"]].to_numpy()
    keep_est, keep_gt = [], []
    for i, ts in enumerate(est_t):
        j = int(np.argmin(np.abs(gt_ts - ts)))
        if abs(gt_ts[j] - ts) <= max_diff:
            keep_est.append(i); keep_gt.append(j)
    return est_xyz[keep_est], gt_xyz[keep_gt], est_t[keep_est]


# ── Manual transform ─────────────────────────────────────────────────────────
def euler_to_R(roll, pitch, yaw):
    """ZYX intrinsic: R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Inputs in degrees."""
    rx, ry, rz = np.radians([roll, pitch, yaw])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def apply_transform(xyz, params):
    """params = (tx, ty, tz, roll, pitch, yaw, scale). Returns transformed xyz."""
    tx, ty, tz, roll, pitch, yaw, s = params
    R = euler_to_R(roll, pitch, yaw)
    return s * (R @ xyz.T).T + np.array([tx, ty, tz])


def compute_ate(est_xyz, gt_xyz):
    err = np.linalg.norm(est_xyz - gt_xyz, axis=1)
    return err, float(np.sqrt((err ** 2).mean()))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ours", default=OURS_DEFAULT)
    p.add_argument("--orb",  default=ORB_DEFAULT)
    p.add_argument("--gt",   default=GROUNDTRUTH_DEFAULT)
    p.add_argument("--max-ts-diff", type=float, default=0.02)
    args = p.parse_args()

    try:
        gt = load_groundtruth(args.gt)
        ours_df = load_ours(args.ours)
        orb_df = load_orb(args.orb)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)

    ours_est, ours_gt, ours_t = match_to_gt(ours_df, gt, args.max_ts_diff)
    orb_est,  orb_gt,  orb_t  = match_to_gt(orb_df,  gt, args.max_ts_diff)
    print(f"Matched frames: Ours={len(ours_est)}  ORB={len(orb_est)}")
    if len(ours_est) < 5 or len(orb_est) < 5:
        print("Too few matched frames — check --gt and --max-ts-diff."); sys.exit(1)

    # Full GT path (every row, not just matched) for the trajectory plot.
    gt_full = gt[["tx", "ty", "tz"]].to_numpy()

    # ── Figure layout: 2x2 top half (XY | XZ over YZ | ATE), sliders below ──
    fig = plt.figure(figsize=(16, 10))
    ax_xy = fig.add_axes([0.05, 0.70, 0.42, 0.23])
    ax_xz = fig.add_axes([0.55, 0.70, 0.42, 0.23])
    ax_yz = fig.add_axes([0.05, 0.43, 0.42, 0.23])
    ax_ate = fig.add_axes([0.55, 0.43, 0.42, 0.23])

    # GT (static) on each trajectory view
    ax_xy.plot(gt_full[:, 0], gt_full[:, 1], "k-", lw=1.2, alpha=0.5, label="Groundtruth")
    ax_xz.plot(gt_full[:, 0], gt_full[:, 2], "k-", lw=1.2, alpha=0.5, label="Groundtruth")
    ax_yz.plot(gt_full[:, 1], gt_full[:, 2], "k-", lw=1.2, alpha=0.5, label="Groundtruth")

    # Initial transforms = identity
    init_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # tx ty tz roll pitch yaw scale
    ours_params = list(init_params)
    orb_params  = list(init_params)

    ours_xfm = apply_transform(ours_est, ours_params)
    orb_xfm  = apply_transform(orb_est,  orb_params)

    ours_err, ours_rmse = compute_ate(ours_xfm, ours_gt)
    orb_err,  orb_rmse  = compute_ate(orb_xfm,  orb_gt)

    # Trajectory lines per view (each gets its own Line2D so we can update independently)
    def make_traj_line(ax, xyz, ix, iy, color, label):
        (ln,) = ax.plot(xyz[:, ix], xyz[:, iy], "-", color=color, lw=1.0, label=label)
        return ln

    ours_xy = make_traj_line(ax_xy, ours_xfm, 0, 1, "#2196F3",
                             f"Our Pipeline (RMSE {ours_rmse:.3f}m)")
    orb_xy  = make_traj_line(ax_xy, orb_xfm,  0, 1, "#F44336",
                             f"ORB-SLAM3 Loc (RMSE {orb_rmse:.3f}m)")
    ours_xz = make_traj_line(ax_xz, ours_xfm, 0, 2, "#2196F3", "Our Pipeline")
    orb_xz  = make_traj_line(ax_xz, orb_xfm,  0, 2, "#F44336", "ORB-SLAM3 Loc")
    ours_yz = make_traj_line(ax_yz, ours_xfm, 1, 2, "#2196F3", "Our Pipeline")
    orb_yz  = make_traj_line(ax_yz, orb_xfm,  1, 2, "#F44336", "ORB-SLAM3 Loc")

    for ax, (xlbl, ylbl, title) in zip(
        [ax_xy, ax_xz, ax_yz],
        [("X (m)", "Y (m)", "Top-down (XY)"),
         ("X (m)", "Z (m)", "Side view (XZ)"),
         ("Y (m)", "Z (m)", "Side view (YZ)")]):
        ax.set_xlabel(xlbl); ax.set_ylabel(ylbl); ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)

    (ours_ate_line,) = ax_ate.plot(ours_t - ours_t[0], ours_err,
                                   color="#2196F3", lw=1,
                                   label=f"Our Pipeline (RMSE {ours_rmse:.3f}m)")
    (orb_ate_line,)  = ax_ate.plot(orb_t - orb_t[0], orb_err,
                                   color="#F44336", lw=1,
                                   label=f"ORB-SLAM3 Loc (RMSE {orb_rmse:.3f}m)")
    ax_ate.set_xlabel("Time (s)"); ax_ate.set_ylabel("ATE (m)")
    ax_ate.set_title("Absolute Trajectory Error over time")
    ax_ate.grid(alpha=0.3); ax_ate.legend(loc="upper right", fontsize=8)

    # ── Sliders ──
    # Slider definitions: (label, vmin, vmax, vinit, valstep)
    slider_defs = [
        ("tx (m)",       -5.0,   5.0, 0.0, 0.001),
        ("ty (m)",       -5.0,   5.0, 0.0, 0.001),
        ("tz (m)",       -5.0,   5.0, 0.0, 0.001),
        ("roll (deg)", -180.0, 180.0, 0.0, 0.1),
        ("pitch (deg)",-180.0, 180.0, 0.0, 0.1),
        ("yaw (deg)",  -180.0, 180.0, 0.0, 0.1),
        ("scale",         0.1,  20.0, 1.0, 0.01),
    ]

    def make_slider_column(x0, width, label_color, header_text):
        """Create 7 sliders stacked vertically. Return list of Slider objects."""
        fig.text(x0 + width / 2, 0.40, header_text,
                 ha="center", color=label_color, fontsize=11, fontweight="bold")
        sliders = []
        for i, (lbl, vmin, vmax, vinit, vstep) in enumerate(slider_defs):
            y = 0.36 - i * 0.045
            ax_s = fig.add_axes([x0, y, width, 0.03])
            s = Slider(ax_s, lbl, vmin, vmax, valinit=vinit, valstep=vstep)
            s.label.set_fontsize(9)
            sliders.append(s)
        return sliders

    ours_sliders = make_slider_column(0.07, 0.30, "#2196F3", "Our Pipeline")
    orb_sliders  = make_slider_column(0.57, 0.30, "#F44336", "ORB-SLAM3 Localization")

    # ── Update callback ──
    def update(_val=None):
        ours_params[:] = [s.val for s in ours_sliders]
        orb_params[:]  = [s.val for s in orb_sliders]

        ours_xfm = apply_transform(ours_est, ours_params)
        orb_xfm  = apply_transform(orb_est,  orb_params)
        ours_err, ours_rmse = compute_ate(ours_xfm, ours_gt)
        orb_err,  orb_rmse  = compute_ate(orb_xfm,  orb_gt)

        # Update all three trajectory views
        ours_xy.set_data(ours_xfm[:, 0], ours_xfm[:, 1])
        orb_xy.set_data( orb_xfm[:, 0],  orb_xfm[:, 1])
        ours_xz.set_data(ours_xfm[:, 0], ours_xfm[:, 2])
        orb_xz.set_data( orb_xfm[:, 0],  orb_xfm[:, 2])
        ours_yz.set_data(ours_xfm[:, 1], ours_xfm[:, 2])
        orb_yz.set_data( orb_xfm[:, 1],  orb_xfm[:, 2])

        ours_xy.set_label(f"Our Pipeline (RMSE {ours_rmse:.3f}m)")
        orb_xy.set_label(f"ORB-SLAM3 Loc (RMSE {orb_rmse:.3f}m)")
        ax_xy.legend(loc="upper right", fontsize=8)
        for ax in (ax_xy, ax_xz, ax_yz):
            ax.relim(); ax.autoscale_view()

        # ATE lines
        ours_ate_line.set_ydata(ours_err)
        orb_ate_line.set_ydata(orb_err)
        ours_ate_line.set_label(f"Our Pipeline (RMSE {ours_rmse:.3f}m)")
        orb_ate_line.set_label(f"ORB-SLAM3 Loc (RMSE {orb_rmse:.3f}m)")
        ax_ate.legend(loc="upper right", fontsize=8)
        ax_ate.relim(); ax_ate.autoscale_view()

        fig.canvas.draw_idle()

    for s in ours_sliders + orb_sliders:
        s.on_changed(update)

    # ── Reset buttons ──
    def make_reset_button(x0, width, sliders, label):
        ax_b = fig.add_axes([x0, 0.005, width, 0.03])
        b = Button(ax_b, label)
        def reset(_event):
            for s in sliders:
                s.reset()
        b.on_clicked(reset)
        return b

    btn_ours = make_reset_button(0.07, 0.30, ours_sliders, "Reset Our Pipeline")
    btn_orb  = make_reset_button(0.57, 0.30, orb_sliders,  "Reset ORB-SLAM3")

    print("Sliders ready. Adjust each parameter; trajectory and ATE update live.")
    print("Tip: start by setting scale, then translation, then rotation.")
    plt.show()


if __name__ == "__main__":
    main()
