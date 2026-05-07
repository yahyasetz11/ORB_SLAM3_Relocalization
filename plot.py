"""
Compare translation error of Standard PnP vs Weighted PnP against TUM ground truth.

Umeyama alignment is applied before computing errors to remove the coordinate-frame
mismatch (different origin, orientation, and monocular scale) between ORB-SLAM3's
map frame and TUM's global motion-capture frame.  Alignment is fitted on the
Standard PnP trajectory and the same transform is reused for Weighted PnP.

Usage:
    python3 plot.py
    python3 plot.py --csv comparison_log.csv --gt data/rgbd_dataset_freiburg3_long_office_household_validation/groundtruth.txt
    python3 plot.py --no-align          # raw (unaligned) errors for comparison
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


def load_groundtruth(path: str) -> pd.DataFrame:
    gt = pd.read_csv(
        path, sep=r"\s+", comment="#",
        names=["t", "tx", "ty", "tz", "qx", "qy", "qz", "qw"],
    )
    gt = gt.sort_values("t").reset_index(drop=True)
    return gt


def match_timestamps(query_ts: np.ndarray, gt_ts: np.ndarray) -> np.ndarray:
    """Return index into gt_ts closest to each query timestamp."""
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
    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])

    R = Vt.T @ D @ U.T
    scale = (S * np.diag(D)).sum() / sigma_s
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def apply_alignment(xyz: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return scale * (R @ xyz.T).T + t


def translation_error(est: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Euclidean distance per row between two (N,3) arrays."""
    return np.linalg.norm(est - ref, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Plot PnP vs ground truth error")
    parser.add_argument("--csv", default=CSV_DEFAULT)
    parser.add_argument("--gt",  default=GROUNDTRUTH_DEFAULT)
    parser.add_argument("--save", default=None, help="Save figure to this path instead of showing")
    parser.add_argument("--max-ts-diff", type=float, default=0.05,
                        help="Reject matches where |timestamp - gt_timestamp| > this value (s)")
    parser.add_argument("--no-align", action="store_true",
                        help="Skip Umeyama alignment (shows raw coordinate-frame error)")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        csv = pd.read_csv(args.csv)
    except FileNotFoundError:
        sys.exit(f"CSV not found: {args.csv}")

    try:
        gt = load_groundtruth(args.gt)
    except FileNotFoundError:
        sys.exit(f"Ground truth not found: {args.gt}")

    required = {"timestamp", "std_x", "std_y", "std_z", "wpnp_x", "wpnp_y", "wpnp_z"}
    if not required.issubset(csv.columns):
        sys.exit(f"CSV is missing columns. Expected: {required}\nGot: {set(csv.columns)}")

    print(f"Loaded {len(csv)} localized frames from {args.csv}")
    print(f"Loaded {len(gt)} ground truth poses from {args.gt}")

    # ── Match timestamps ───────────────────────────────────────────────────────
    gt_ts  = gt["t"].to_numpy()
    csv_ts = csv["timestamp"].to_numpy()
    idx    = match_timestamps(csv_ts, gt_ts)

    ts_diff = np.abs(csv_ts - gt_ts[idx])
    valid   = ts_diff <= args.max_ts_diff

    if valid.sum() == 0:
        sys.exit(
            f"No CSV timestamps matched within {args.max_ts_diff}s of any ground truth entry.\n"
            "Check that timestamp_start in relocalization_params.yaml matches the ground truth file."
        )

    print(f"Frames matched to ground truth: {valid.sum()}/{len(csv)}  "
          f"(max timestamp diff: {ts_diff[valid].max():.4f}s)")

    csv_valid   = csv[valid].reset_index(drop=True)
    gt_matched  = gt.iloc[idx[valid]].reset_index(drop=True)
    gt_xyz      = gt_matched[["tx", "ty", "tz"]].to_numpy()

    std_xyz  = csv_valid[["std_x", "std_y", "std_z"]].to_numpy()

    wpnp_xyz = csv_valid[["wpnp_x", "wpnp_y", "wpnp_z"]].to_numpy()
    wpnp_ran = ~(
        (wpnp_xyz[:, 0] == 0.0) &
        (wpnp_xyz[:, 1] == 0.0) &
        (wpnp_xyz[:, 2] == 0.0)
    )

    # ── Umeyama alignment ──────────────────────────────────────────────────────
    align_label = ""
    if not args.no_align:
        scale, R, t = umeyama_alignment(std_xyz, gt_xyz)
        std_xyz_plot  = apply_alignment(std_xyz, scale, R, t)
        wpnp_xyz_plot = apply_alignment(wpnp_xyz, scale, R, t)
        align_label   = " (Umeyama-aligned)"
        print(f"\nUmeyama alignment  scale={scale:.4f}  "
              f"t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
    else:
        std_xyz_plot  = std_xyz
        wpnp_xyz_plot = wpnp_xyz
        align_label   = " (raw — no alignment)"

    # ── Compute errors ────────────────────────────────────────────────────────
    std_err  = translation_error(std_xyz_plot, gt_xyz)
    wpnp_err = translation_error(wpnp_xyz_plot[wpnp_ran], gt_xyz[wpnp_ran])

    print(f"\nStandard PnP  — frames: {len(std_err):4d}  "
          f"mean: {std_err.mean():.4f}m  median: {np.median(std_err):.4f}m  "
          f"max: {std_err.max():.4f}m")
    print(f"Weighted PnP  — frames: {wpnp_ran.sum():4d}  "
          f"mean: {wpnp_err.mean():.4f}m  median: {np.median(wpnp_err):.4f}m  "
          f"max: {wpnp_err.max():.4f}m")

    # ── Plot ──────────────────────────────────────────────────────────────────
    timestamps_all  = csv_valid["timestamp"].to_numpy()
    timestamps_wpnp = timestamps_all[wpnp_ran]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        timestamps_all, std_err,
        color="#2196F3", linewidth=1.0, alpha=0.85,
        label=f"Standard PnP (OpenCV)  mean={std_err.mean():.3f}m",
    )
    ax.plot(
        timestamps_wpnp, wpnp_err,
        color="#F44336", linewidth=1.0, alpha=0.85,
        label=f"Weighted PnP            mean={wpnp_err.mean():.3f}m",
    )

    ax.axhline(std_err.mean(),  color="#2196F3", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(wpnp_err.mean(), color="#F44336", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Timestamp (s)")
    ax.set_ylabel("Translation error (m)")
    ax.set_title(
        f"Relocalization error vs TUM ground truth{align_label}\n"
        "Standard PnP vs Weighted PnP"
    )
    ax.legend(loc="upper right")
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
