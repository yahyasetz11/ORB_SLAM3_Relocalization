"""
Compare translation error of Standard PnP vs Weighted PnP against TUM ground truth.

Umeyama alignment is applied before computing errors to remove the coordinate-frame
mismatch (different origin, orientation, and monocular scale) between ORB-SLAM3's
map frame and TUM's global motion-capture frame.  Alignment is fitted on the
Standard PnP trajectory and the same transform is reused for Weighted PnP.

Usage:
    # Single pair (default files)
    python3 plot.py

    # Single pair (explicit)
    python3 plot.py --csv run1.csv --gt groundtruth.txt --labels "Run 1"

    # Multiple pairs on the same plot
    python3 plot.py \\
        --csv run1.csv run2.csv run3.csv \\
        --gt  gt1.txt  gt2.txt  gt3.txt \\
        --labels "w=1.0/0.3" "w=1.0/0.5" "w=1.0/0.7"

    # Skip Umeyama (show raw coordinate-frame error)
    python3 plot.py --no-align

    # Save instead of showing
    python3 plot.py --save error_comparison.png
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

# Color palette: each pair gets one base color; std=solid, wpnp=dashed
PAIR_COLORS = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
]


def load_groundtruth(path: str) -> pd.DataFrame:
    gt = pd.read_csv(
        path, sep=r"\s+", comment="#",
        names=["t", "tx", "ty", "tz", "qx", "qy", "qz", "qw"],
    )
    gt = gt.sort_values("t").reset_index(drop=True)
    return gt


def match_timestamps(query_ts: np.ndarray, gt_ts: np.ndarray) -> np.ndarray:
    return np.array([np.argmin(np.abs(gt_ts - ts)) for ts in query_ts])


def umeyama_alignment(src: np.ndarray, dst: np.ndarray):
    """
    Least-squares Sim(3) alignment: find scale s, rotation R, translation t
    such that  s * R @ src[i] + t  ≈  dst[i].

    Returns (scale, R, t) where aligned = scale * (R @ src.T).T + t.
    """
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


def apply_alignment(xyz: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return scale * (R @ xyz.T).T + t


def translation_error(est: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.linalg.norm(est - ref, axis=1)


def process_pair(csv_path, gt_path, no_align, max_ts_diff, label):
    """Load, match, align, and compute errors for one CSV/GT pair."""
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
        sys.exit(f"[{label}] CSV missing columns. Expected: {required}\nGot: {set(csv.columns)}")

    print(f"\n── {label} ───────────────────────────────────────")
    print(f"  CSV : {csv_path}  ({len(csv)} frames)")
    print(f"  GT  : {gt_path}  ({len(gt)} poses)")

    gt_ts  = gt["t"].to_numpy()
    csv_ts = csv["timestamp"].to_numpy()
    idx    = match_timestamps(csv_ts, gt_ts)

    ts_diff = np.abs(csv_ts - gt_ts[idx])
    valid   = ts_diff <= max_ts_diff

    if valid.sum() == 0:
        sys.exit(
            f"[{label}] No timestamps matched within {max_ts_diff}s.\n"
            "Check timestamp_start in relocalization_params.yaml."
        )

    print(f"  Matched: {valid.sum()}/{len(csv)}  "
          f"(max ts diff: {ts_diff[valid].max():.4f}s)")

    csv_valid  = csv[valid].reset_index(drop=True)
    gt_matched = gt.iloc[idx[valid]].reset_index(drop=True)
    gt_xyz     = gt_matched[["tx", "ty", "tz"]].to_numpy()

    std_xyz  = csv_valid[["std_x", "std_y", "std_z"]].to_numpy()
    wpnp_xyz = csv_valid[["wpnp_x", "wpnp_y", "wpnp_z"]].to_numpy()
    if not no_align:
        scale, R, t = umeyama_alignment(std_xyz, gt_xyz)
        std_xyz_plot  = apply_alignment(std_xyz, scale, R, t)
        wpnp_xyz_plot = apply_alignment(wpnp_xyz, scale, R, t)
        print(f"  Umeyama  scale={scale:.4f}  t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
    else:
        std_xyz_plot  = std_xyz
        wpnp_xyz_plot = wpnp_xyz

    std_err  = translation_error(std_xyz_plot, gt_xyz)
    wpnp_err = translation_error(wpnp_xyz_plot, gt_xyz)

    print(f"  Std PnP  — frames: {len(std_err):4d}  "
          f"mean: {std_err.mean():.4f}m  median: {np.median(std_err):.4f}m  "
          f"max: {std_err.max():.4f}m")
    print(f"  WPnP     — frames: {len(wpnp_err):4d}  "
          f"mean: {wpnp_err.mean():.4f}m  median: {np.median(wpnp_err):.4f}m  "
          f"max: {wpnp_err.max():.4f}m")

    timestamps_all  = csv_valid["timestamp"].to_numpy()
    timestamps_wpnp = timestamps_all

    return {
        "label":           label,
        "timestamps_std":  timestamps_all,
        "timestamps_wpnp": timestamps_wpnp,
        "std_err":         std_err,
        "wpnp_err":        wpnp_err,
    }


def main():
    parser = argparse.ArgumentParser(description="Plot PnP vs ground truth error")
    parser.add_argument("--csv",    nargs="+", default=[CSV_DEFAULT],
                        metavar="CSV",
                        help="One or more CSV log files")
    parser.add_argument("--gt",     nargs="+", default=[GROUNDTRUTH_DEFAULT],
                        metavar="GT",
                        help="One or more ground truth files (must match --csv count)")
    parser.add_argument("--labels", nargs="+", default=None,
                        metavar="LABEL",
                        help="Legend labels for each pair (defaults to CSV filename)")
    parser.add_argument("--save",   default=None,
                        help="Save figure to this path instead of showing")
    parser.add_argument("--max-ts-diff", type=float, default=0.05,
                        help="Reject matches where |timestamp - gt_timestamp| > this (s)")
    parser.add_argument("--no-align", action="store_true",
                        help="Skip Umeyama alignment (shows raw coordinate-frame error)")
    args = parser.parse_args()

    if len(args.csv) != len(args.gt):
        sys.exit(f"--csv and --gt must have the same number of entries "
                 f"(got {len(args.csv)} csv, {len(args.gt)} gt)")

    labels = args.labels or [p.split("/")[-1].replace(".csv", "") for p in args.csv]
    if len(labels) != len(args.csv):
        sys.exit(f"--labels count ({len(labels)}) must match --csv count ({len(args.csv)})")

    align_label = " (raw — no alignment)" if args.no_align else " (Umeyama-aligned)"

    # ── Process all pairs ─────────────────────────────────────────────────────
    results = [
        process_pair(csv_path, gt_path, args.no_align, args.max_ts_diff, label)
        for csv_path, gt_path, label in zip(args.csv, args.gt, labels)
    ]

    # ── Plot ──────────────────────────────────────────────────────────────────
    # Use relative time (seconds from start of first sequence) so the x-axis
    # shows a human-readable 0-based offset instead of raw Unix timestamps.
    t0 = min(r["timestamps_std"][0] for r in results)

    fig, ax = plt.subplots(figsize=(13, 5))

    for i, r in enumerate(results):
        color = PAIR_COLORS[i % len(PAIR_COLORS)]
        label = r["label"]

        ts_std  = r["timestamps_std"]  - t0
        ts_wpnp = r["timestamps_wpnp"] - t0

        ax.plot(
            ts_std, r["std_err"],
            color=color, linewidth=1.0, alpha=0.85, linestyle="-",
            label=f"[{label}] Std PnP   mean={r['std_err'].mean():.3f}m",
        )
        ax.axhline(r["std_err"].mean(), color=color, linestyle="--",
                   linewidth=0.8, alpha=0.4)

        if len(r["wpnp_err"]) > 0:
            ax.plot(
                ts_wpnp, r["wpnp_err"],
                color=color, linewidth=1.0, alpha=0.65, linestyle=":",
                label=f"[{label}] WPnP      mean={r['wpnp_err'].mean():.3f}m",
            )

    ax.set_xlabel("Time from sequence start (s)")
    ax.set_ylabel("Translation error (m)")
    ax.set_title(
        f"Relocalization error vs TUM ground truth{align_label}\n"
        "Standard PnP (solid) vs Weighted PnP (dotted)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"\nFigure saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
