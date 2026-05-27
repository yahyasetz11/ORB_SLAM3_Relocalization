"""
Subplot comparison: Std PnP on top, one WPnP subplot per CSV below.

Layout for N CSVs → N+1 subplots (shared x/y axes):
  [0]  Std PnP reference   — taken from the first CSV
  [1]  WPnP  label[0]
  [2]  WPnP  label[1]
  ...
  [N]  WPnP  label[N-1]

Each subplot has exactly one line and one legend entry.

Usage:
    python3 plot3.py \\
        --csv run_w10.csv run_w07.csv run_w03.csv run_w00.csv \\
        --gt  gt.txt gt.txt gt.txt gt.txt \\
        --labels "bg-weight=1.0" "bg-weight=0.7" "bg-weight=0.3" "bg-weight=0.0" \\
        --save sweep_subplots.png
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import svd


GROUNDTRUTH_DEFAULT = (
    "data/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt"
)
CSV_DEFAULT = "comparison_log.csv"

WPNP_COLORS = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
]

STD_COLOR = "#555555"


def load_groundtruth(path: str) -> pd.DataFrame:
    gt = pd.read_csv(
        path, sep=r"\s+", comment="#",
        names=["t", "tx", "ty", "tz", "qx", "qy", "qz", "qw"],
    )
    return gt.sort_values("t").reset_index(drop=True)


def match_timestamps(query_ts: np.ndarray, gt_ts: np.ndarray) -> np.ndarray:
    return np.array([np.argmin(np.abs(gt_ts - ts)) for ts in query_ts])


def umeyama_alignment(src: np.ndarray, dst: np.ndarray):
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    sigma_s = (src_c ** 2).sum() / len(src)
    H = src_c.T @ dst_c / len(src)
    U, S, Vt = svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    scale = (S * np.diag(D)).sum() / sigma_s
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def apply_alignment(xyz, scale, R, t):
    return scale * (R @ xyz.T).T + t


def translation_error(est, ref):
    return np.linalg.norm(est - ref, axis=1)


def process_pair(csv_path, gt_path, no_align, max_ts_diff, label):
    try:
        csv = pd.read_csv(csv_path)
    except FileNotFoundError:
        sys.exit(f"CSV not found: {csv_path}")
    try:
        gt = load_groundtruth(gt_path)
    except FileNotFoundError:
        sys.exit(f"Ground truth not found: {gt_path}")

    required = {"timestamp", "std_x", "std_y", "std_z", "wpnp_x", "wpnp_y", "wpnp_z"}
    if not required.issubset(csv.columns):
        sys.exit(f"[{label}] CSV missing columns. Got: {set(csv.columns)}")

    print(f"\n── {label} ───────────────────────────────────────")
    print(f"  CSV : {csv_path}  ({len(csv)} frames)")
    print(f"  GT  : {gt_path}  ({len(gt)} poses)")

    gt_ts  = gt["t"].to_numpy()
    csv_ts = csv["timestamp"].to_numpy()
    idx    = match_timestamps(csv_ts, gt_ts)

    ts_diff = np.abs(csv_ts - gt_ts[idx])
    valid   = ts_diff <= max_ts_diff

    if valid.sum() == 0:
        sys.exit(f"[{label}] No timestamps matched within {max_ts_diff}s.")

    print(f"  Matched: {valid.sum()}/{len(csv)}  "
          f"(max ts diff: {ts_diff[valid].max():.4f}s)")

    csv_valid  = csv[valid].reset_index(drop=True)
    gt_matched = gt.iloc[idx[valid]].reset_index(drop=True)
    gt_xyz     = gt_matched[["tx", "ty", "tz"]].to_numpy()

    std_xyz  = csv_valid[["std_x",  "std_y",  "std_z"]].to_numpy()
    wpnp_xyz = csv_valid[["wpnp_x", "wpnp_y", "wpnp_z"]].to_numpy()
    wpnp_ran = ~(
        (wpnp_xyz[:, 0] == 0.0) &
        (wpnp_xyz[:, 1] == 0.0) &
        (wpnp_xyz[:, 2] == 0.0)
    )

    if not no_align:
        scale, R, t = umeyama_alignment(std_xyz, gt_xyz)
        std_xyz_plot  = apply_alignment(std_xyz,  scale, R, t)
        wpnp_xyz_plot = apply_alignment(wpnp_xyz, scale, R, t)
        print(f"  Umeyama  scale={scale:.4f}  t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
    else:
        std_xyz_plot  = std_xyz
        wpnp_xyz_plot = wpnp_xyz

    std_err  = translation_error(std_xyz_plot, gt_xyz)
    wpnp_err = translation_error(wpnp_xyz_plot[wpnp_ran], gt_xyz[wpnp_ran])

    print(f"  Std PnP  — RMSE: {np.sqrt((std_err**2).mean()):.4f}m  median: {np.median(std_err):.4f}m")
    print(f"  WPnP     — RMSE: {np.sqrt((wpnp_err**2).mean()):.4f}m  median: {np.median(wpnp_err):.4f}m")

    timestamps_all  = csv_valid["timestamp"].to_numpy()
    timestamps_wpnp = timestamps_all[wpnp_ran]

    return {
        "label":           label,
        "timestamps_std":  timestamps_all,
        "timestamps_wpnp": timestamps_wpnp,
        "std_err":         std_err,
        "wpnp_err":        wpnp_err,
    }


def main():
    parser = argparse.ArgumentParser(description="Subplot WPnP sweep (Std PnP on top)")
    parser.add_argument("--csv",    nargs="+", default=[CSV_DEFAULT], metavar="CSV")
    parser.add_argument("--gt",     nargs="+", default=[GROUNDTRUTH_DEFAULT], metavar="GT")
    parser.add_argument("--labels", nargs="+", default=None, metavar="LABEL")
    parser.add_argument("--save",   default=None)
    parser.add_argument("--max-ts-diff", type=float, default=0.05)
    parser.add_argument("--no-align", action="store_true")
    args = parser.parse_args()

    if len(args.csv) != len(args.gt):
        sys.exit("--csv and --gt must have the same count")

    labels = args.labels or [p.split("/")[-1].replace(".csv", "") for p in args.csv]
    if len(labels) != len(args.csv):
        sys.exit("--labels count must match --csv count")

    align_label = " (raw)" if args.no_align else " (Umeyama-aligned)"

    results = [
        process_pair(csv_path, gt_path, args.no_align, args.max_ts_diff, label)
        for csv_path, gt_path, label in zip(args.csv, args.gt, labels)
    ]

    t0 = min(r["timestamps_std"][0] for r in results)

    # N CSVs → N+1 subplots: [0]=Std PnP ref, [1..N]=WPnP per CSV
    n_rows = len(results) + 1
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(13, 3.0 * n_rows),
        sharex=True,
        sharey=True,
    )

    fig.suptitle(
        f"Relocalization error vs TUM ground truth{align_label}",
        fontsize=12,
    )

    # ── Subplot 0: Std PnP reference (from first CSV) ────────────────────────
    ref = results[0]
    ax0 = axes[0]
    ts_std = ref["timestamps_std"] - t0
    ax0.plot(
        ts_std, ref["std_err"],
        color=STD_COLOR, linewidth=1.2, alpha=0.85, linestyle="-",
        label=f"Std PnP  RMSE={np.sqrt((ref['std_err']**2).mean()):.3f}m",
    )
    ax0.axhline(np.sqrt((ref["std_err"]**2).mean()), color=STD_COLOR, linestyle="--",
                linewidth=0.8, alpha=0.35)
    ax0.set_title("Standard PnP (reference)", fontsize=10, loc="left", pad=4)
    ax0.set_ylabel("Translation error (m)")
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(True, alpha=0.3)
    ax0.set_ylim(bottom=0)

    # ── Subplots 1..N: one WPnP line each ────────────────────────────────────
    for i, (r, ax) in enumerate(zip(results, axes[1:])):
        color   = WPNP_COLORS[i % len(WPNP_COLORS)]
        ts_wpnp = r["timestamps_wpnp"] - t0

        if len(r["wpnp_err"]) > 0:
            ax.plot(
                ts_wpnp, r["wpnp_err"],
                color=color, linewidth=1.2, alpha=0.85, linestyle="-",
                label=f"WPnP  RMSE={np.sqrt((r['wpnp_err']**2).mean()):.3f}m",
            )
            ax.axhline(np.sqrt((r["wpnp_err"]**2).mean()), color=color, linestyle="--",
                       linewidth=0.8, alpha=0.35)

        ax.set_title(r["label"], fontsize=10, loc="left", pad=4)
        ax.set_ylabel("Translation error (m)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Time from sequence start (s)")

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"\nFigure saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
