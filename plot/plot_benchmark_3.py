"""
Benchmark: ORB-SLAM3 built-in Localization vs. Our Pipeline (single WPnP weight config).

Reads two CSVs and a TUM-format groundtruth file, aligns each estimated trajectory to
GT with a FIXED Sim(3) transform (manually tuned, no Umeyama), and reports:
    - ATE (RMSE of translation residuals)
    - RPE (translational + rotational, over a configurable time delta)
    - Mean reprojection error (Our Pipeline only)
    - Success rate (localized frames / total frames)
    - Umeyama-derived alignment params for Our Pipeline (printed for reference)

Alignment parameters are hardcoded below (from benchmark_params.yaml):
    ORB_SLAM3: tx=-0.4, ty=2.72, tz=1.17, roll=73°, pitch=180°, yaw=-7°, scale=2.54
    Ours:      tx=-1.01, ty=0.6, tz=1.54, roll=59.8°, pitch=1°, yaw=8.5°, scale=1.47

Usage:
    python3 plot_benchmark_3.py
    python3 plot_benchmark_3.py --ours comparison_log.csv --orb localization_test_log.csv \
                                --gt data/.../groundtruth.txt --rpe-delta 1.0
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import svd


GROUNDTRUTH_DEFAULT = "data/rgbd_dataset_freiburg3_long_office_household_validation/groundtruth.txt"
OURS_DEFAULT = "comparison_log.csv"
ORB_DEFAULT = "localization_test_log.csv"

# ── Hardcoded alignment parameters (from benchmark_params.yaml) ───────────────
ALIGN_ORB = {
    "tx": -0.4, "ty": 2.72, "tz": 1.17,
    "roll": 73.0, "pitch": 180.0, "yaw": -7.0,
    "scale": 2.54,
}
ALIGN_OURS = {
    "tx": -0.4531, "ty": 2.6504, "tz": 1.7391,
    "roll": -119.16, "pitch": -3.02, "yaw": 173.66,
    "scale": 2.5062,
}


# ── Loaders ──────────────────────────────────────────────────────────────────
def load_groundtruth(path: str) -> pd.DataFrame:
    gt = pd.read_csv(
        path, sep=r"\s+", comment="#",
        names=["t", "tx", "ty", "tz", "qx", "qy", "qz", "qw"],
    )
    return gt.sort_values("t").reset_index(drop=True)


def load_ours(path: str) -> pd.DataFrame:
    """Our Pipeline CSV — uses wpnp_x/y/z (same SLAM-frame coords as plot3.py)."""
    df = pd.read_csv(path)
    ran = ~((df["wpnp_x"] == 0.0) & (df["wpnp_y"] == 0.0) & (df["wpnp_z"] == 0.0))
    df = df[ran].copy()
    df = df.rename(columns={
        "timestamp": "t",
        "wpnp_x": "tx", "wpnp_y": "ty", "wpnp_z": "tz",
        "result_qx": "qx", "result_qy": "qy", "result_qz": "qz", "result_qw": "qw",
    })
    return df.sort_values("t").reset_index(drop=True)


def load_orb(path: str) -> pd.DataFrame:
    """ORB-SLAM3 localization_test_node CSV. Keeps every frame for success-rate calc."""
    df = pd.read_csv(path)
    df["t"] = df["timestamp"]
    return df.sort_values("t").reset_index(drop=True)


# ── Alignment ─────────────────────────────────────────────────────────────────
def euler_to_rotmat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Build R = Rz(yaw) @ Ry(pitch) @ Rx(roll) from degrees."""
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    Rx = np.array([[1, 0, 0],
                   [0,  np.cos(r), -np.sin(r)],
                   [0,  np.sin(r),  np.cos(r)]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0,          1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0,          0,         1]])
    return Rz @ Ry @ Rx


def rotmat_to_euler(R: np.ndarray):
    """Extract roll, pitch, yaw (degrees) from R = Rz @ Ry @ Rx (ZYX intrinsic)."""
    pitch = np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def fixed_alignment(params: dict):
    """Return (scale, R, t) from hardcoded params dict."""
    R = euler_to_rotmat(params["roll"], params["pitch"], params["yaw"])
    t = np.array([params["tx"], params["ty"], params["tz"]])
    return params["scale"], R, t


def umeyama_alignment(src: np.ndarray, dst: np.ndarray):
    """Least-squares Sim(3): return (scale, R, t) so dst ≈ scale*(R@src) + t."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    sigma_s = (src_c ** 2).sum() / len(src)
    cov = (dst_c.T @ src_c) / len(src)
    U, D, Vt = svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    scale = (S * np.diag(D)).sum() / sigma_s
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def apply_alignment(xyz: np.ndarray, s: float, R: np.ndarray, t: np.ndarray):
    return s * (R @ xyz.T).T + t


# ── Timestamp matching ────────────────────────────────────────────────────────
def match_timestamps(query_ts: np.ndarray, gt_ts: np.ndarray,
                     max_diff: float) -> np.ndarray:
    """Return GT index for each query timestamp; -1 if no GT within max_diff."""
    out = np.full(len(query_ts), -1, dtype=int)
    for i, ts in enumerate(query_ts):
        j = int(np.argmin(np.abs(gt_ts - ts)))
        if abs(gt_ts[j] - ts) <= max_diff:
            out[i] = j
    return out


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_ate(est_xyz: np.ndarray, gt_xyz: np.ndarray):
    """Per-frame translation residual + RMSE/mean/median."""
    err = np.linalg.norm(est_xyz - gt_xyz, axis=1)
    return err, {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "std": float(err.std()),
        "max": float(err.max()),
    }


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """q: (N,4) in (qx,qy,qz,qw). Returns (N,3,3)."""
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2*(qy*qy + qz*qz)
    R[:, 0, 1] = 2*(qx*qy - qz*qw)
    R[:, 0, 2] = 2*(qx*qz + qy*qw)
    R[:, 1, 0] = 2*(qx*qy + qz*qw)
    R[:, 1, 1] = 1 - 2*(qx*qx + qz*qz)
    R[:, 1, 2] = 2*(qy*qz - qx*qw)
    R[:, 2, 0] = 2*(qx*qz - qy*qw)
    R[:, 2, 1] = 2*(qy*qz + qx*qw)
    R[:, 2, 2] = 1 - 2*(qx*qx + qy*qy)
    return R


def compute_rpe(t_arr: np.ndarray, est_xyz: np.ndarray, est_q: np.ndarray,
                gt_xyz: np.ndarray, gt_q: np.ndarray, delta: float):
    R_est = quat_to_rotmat(est_q)
    R_gt  = quat_to_rotmat(gt_q)

    trans_errs, rot_errs, mid_ts = [], [], []
    for i in range(len(t_arr)):
        target = t_arr[i] + delta
        j = int(np.searchsorted(t_arr, target))
        if j >= len(t_arr) or abs(t_arr[j] - target) > delta * 0.5:
            continue
        R_est_rel = R_est[i].T @ R_est[j]
        t_est_rel = R_est[i].T @ (est_xyz[j] - est_xyz[i])
        R_gt_rel  = R_gt[i].T  @ R_gt[j]
        t_gt_rel  = R_gt[i].T  @ (gt_xyz[j] - gt_xyz[i])
        R_err = R_gt_rel.T @ R_est_rel
        t_err = R_gt_rel.T @ (t_est_rel - t_gt_rel)
        trans_errs.append(np.linalg.norm(t_err))
        cos_a = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
        rot_errs.append(np.degrees(np.arccos(cos_a)))
        mid_ts.append(t_arr[i])

    trans_errs = np.array(trans_errs)
    rot_errs   = np.array(rot_errs)
    mid_ts     = np.array(mid_ts)

    if len(trans_errs) == 0:
        return mid_ts, trans_errs, rot_errs, {"trans_rmse": np.nan, "rot_rmse_deg": np.nan}

    return mid_ts, trans_errs, rot_errs, {
        "trans_rmse":    float(np.sqrt((trans_errs**2).mean())),
        "trans_mean":    float(trans_errs.mean()),
        "rot_rmse_deg":  float(np.sqrt((rot_errs**2).mean())),
        "rot_mean_deg":  float(rot_errs.mean()),
    }


# ── Pipeline ─────────────────────────────────────────────────────────────────
def rotmat_to_quat_batch(R: np.ndarray) -> np.ndarray:
    N = R.shape[0]
    q = np.empty((N, 4))
    for i in range(N):
        m = R[i]
        tr = m[0,0] + m[1,1] + m[2,2]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            qw = 0.25*s; qx = (m[2,1]-m[1,2])/s; qy = (m[0,2]-m[2,0])/s; qz = (m[1,0]-m[0,1])/s
        elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
            s = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
            qw = (m[2,1]-m[1,2])/s; qx = 0.25*s; qy = (m[0,1]+m[1,0])/s; qz = (m[0,2]+m[2,0])/s
        elif m[1,1] > m[2,2]:
            s = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
            qw = (m[0,2]-m[2,0])/s; qx = (m[0,1]+m[1,0])/s; qy = 0.25*s; qz = (m[1,2]+m[2,1])/s
        else:
            s = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
            qw = (m[1,0]-m[0,1])/s; qx = (m[0,2]+m[2,0])/s; qy = (m[1,2]+m[2,1])/s; qz = 0.25*s
        q[i] = [qx, qy, qz, qw]
    return q


def process_run(df: pd.DataFrame, gt: pd.DataFrame, align_params: dict,
                max_ts_diff: float, rpe_delta: float, label: str):
    """Match → apply fixed alignment → compute ATE/RPE. Also returns matched arrays for Umeyama."""
    ts    = df["t"].to_numpy()
    est_xyz = df[["tx","ty","tz"]].to_numpy()
    est_q   = df[["qx","qy","qz","qw"]].to_numpy()

    gt_ts  = gt["t"].to_numpy()
    gt_xyz = gt[["tx","ty","tz"]].to_numpy()
    gt_q   = gt[["qx","qy","qz","qw"]].to_numpy()

    idx  = match_timestamps(ts, gt_ts, max_ts_diff)
    keep = idx >= 0
    if keep.sum() < 5:
        print(f"[{label}] only {keep.sum()} frames matched groundtruth — skipping")
        return None

    est_xyz_m = est_xyz[keep]
    est_q_m   = est_q[keep]
    ts_m      = ts[keep]
    gt_xyz_m  = gt_xyz[idx[keep]]
    gt_q_m    = gt_q[idx[keep]]

    s, R, t = fixed_alignment(align_params)
    est_aligned   = apply_alignment(est_xyz_m, s, R, t)
    est_R_aligned = R @ quat_to_rotmat(est_q_m)

    ate_per_frame, ate_stats = compute_ate(est_aligned, gt_xyz_m)

    est_q_aligned = rotmat_to_quat_batch(est_R_aligned)
    rpe_t, rpe_trans, rpe_rot, rpe_stats = compute_rpe(
        ts_m, est_aligned, est_q_aligned, gt_xyz_m, gt_q_m, rpe_delta)

    return {
        "label":         label,
        "ts":            ts_m,
        "est_aligned":   est_aligned,
        "est_xyz_m":     est_xyz_m,   # raw matched xyz — used for Umeyama
        "gt":            gt_xyz_m,
        "scale":         s,
        "align":         align_params,
        "ate_per_frame": ate_per_frame,
        "ate":           ate_stats,
        "rpe_ts":        rpe_t,
        "rpe_trans":     rpe_trans,
        "rpe_rot":       rpe_rot,
        "rpe":           rpe_stats,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ours", default=OURS_DEFAULT)
    p.add_argument("--orb",  default=ORB_DEFAULT)
    p.add_argument("--gt",   default=GROUNDTRUTH_DEFAULT)
    p.add_argument("--rpe-delta",    type=float, default=1.0)
    p.add_argument("--max-ts-diff",  type=float, default=0.02)
    p.add_argument("--out", default="benchmark_plot_3.png")
    args = p.parse_args()

    print(f"Loading: ours={args.ours}  orb={args.orb}  gt={args.gt}")
    try:
        gt   = load_groundtruth(args.gt)
        ours = load_ours(args.ours)
        orb  = load_orb(args.orb)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)

    ours_raw = pd.read_csv(args.ours)
    print(f"  GT rows: {len(gt)}")
    print(f"  Ours rows (WPnP ran): {len(ours)}, "
          f"of which localized: {int(ours_raw['is_localized'].sum())}")
    print(f"  ORB rows (all): {len(orb)}, "
          f"localized: {int(orb['is_localized'].sum())}")

    total_frames = len(orb)
    ours_success = int(ours_raw["is_localized"].sum()) / total_frames if total_frames else 0.0
    orb_success  = orb["is_localized"].sum() / total_frames if total_frames else 0.0

    ours_reproj_mean = ours["wpnp_reproj_px"].dropna().mean() if "wpnp_reproj_px" in ours else float("nan")
    ours_reproj_inl  = ours["std_reproj_inliers_only"].dropna().mean() if "std_reproj_inliers_only" in ours else float("nan")

    orb_loc  = orb[orb["is_localized"] == 1].copy()
    ours_run = process_run(ours, gt, ALIGN_OURS, args.max_ts_diff, args.rpe_delta, "Our Pipeline")
    orb_run  = process_run(orb_loc, gt, ALIGN_ORB, args.max_ts_diff, args.rpe_delta, "ORB-SLAM3 Localization")

    # ── Umeyama reference for Our Pipeline ───────────────────────────────────
    if ours_run:
        u_scale, u_R, u_t = umeyama_alignment(ours_run["est_xyz_m"], ours_run["gt"])
        u_roll, u_pitch, u_yaw = rotmat_to_euler(u_R)
        print("\n── Umeyama alignment for Our Pipeline (reference only, not used in plot) ──")
        print(f"  scale : {u_scale:.4f}")
        print(f"  tx    : {u_t[0]:.4f}   ty: {u_t[1]:.4f}   tz: {u_t[2]:.4f}")
        print(f"  roll  : {u_roll:.2f}°   pitch: {u_pitch:.2f}°   yaw: {u_yaw:.2f}°")

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Metric':<32} {'Our Pipeline':>18} {'ORB-SLAM3 Loc':>18}")
    print("=" * 72)

    def row(name, ours_v, orb_v, fmt="{:>18.4f}"):
        ours_s = fmt.format(ours_v) if ours_v == ours_v else f"{'N/A':>18}"
        orb_s  = fmt.format(orb_v)  if orb_v  == orb_v  else f"{'N/A':>18}"
        print(f"{name:<32} {ours_s} {orb_s}")

    if ours_run and orb_run:
        row("ATE RMSE (m)",         ours_run["ate"]["rmse"],         orb_run["ate"]["rmse"])
        row("ATE mean (m)",         ours_run["ate"]["mean"],         orb_run["ate"]["mean"])
        row("ATE median (m)",       ours_run["ate"]["median"],       orb_run["ate"]["median"])
        row(f"RPE trans RMSE (m/{args.rpe_delta}s)",
            ours_run["rpe"]["trans_rmse"], orb_run["rpe"]["trans_rmse"])
        row(f"RPE rot RMSE (deg/{args.rpe_delta}s)",
            ours_run["rpe"]["rot_rmse_deg"], orb_run["rpe"]["rot_rmse_deg"])
        row("Fixed scale", ours_run["scale"], orb_run["scale"])
    row("Mean reproj (WPnP, px)", ours_reproj_mean, float("nan"))
    row("Mean reproj (inliers, px)", ours_reproj_inl, float("nan"))
    row("Success rate (%)", 100.0*ours_success, 100.0*orb_success, fmt="{:>17.2f}%")
    print("=" * 72)

    if not (ours_run and orb_run):
        print("Skipping plot (one or both trajectories had too few GT matches).")
        return

    # ── Plot (2×3 grid: trajectory | ATE | summary / RPE-trans | RPE-rot | empty) ──
    fig = plt.figure(figsize=(16, 10))

    # (0,0) Top-down XY — fixed-aligned
    ax = fig.add_subplot(2, 3, 1)
    ax.plot(ours_run["gt"][:, 0], ours_run["gt"][:, 1],
            "k-", lw=1.5, label="Groundtruth", alpha=0.6)
    ax.plot(ours_run["est_aligned"][:, 0], ours_run["est_aligned"][:, 1],
            "-", color="#2196F3", lw=1.0, label="Our Pipeline")
    ax.plot(orb_run["est_aligned"][:, 0], orb_run["est_aligned"][:, 1],
            "-", color="#F44336", lw=1.0, label="ORB-SLAM3 Loc")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Top-down trajectory (XY) — fixed alignment")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

    # (0,1) ATE over time
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(ours_run["ts"] - ours_run["ts"][0], ours_run["ate_per_frame"],
            color="#2196F3", lw=1, label=f"Our Pipeline (RMSE {ours_run['ate']['rmse']:.3f}m)")
    ax.plot(orb_run["ts"] - orb_run["ts"][0], orb_run["ate_per_frame"],
            color="#F44336", lw=1, label=f"ORB-SLAM3 Loc (RMSE {orb_run['ate']['rmse']:.3f}m)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("ATE (m)")
    ax.set_title("Absolute Trajectory Error over time")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (0,2) Summary text panel
    ax = fig.add_subplot(2, 3, 3)
    ax.axis("off")
    lines = [
        "Summary",
        "─" * 38,
        f"{'':25s}{'Ours':>6s}{'ORB':>7s}",
        f"{'ATE RMSE (m)':25s}{ours_run['ate']['rmse']:>6.3f}{orb_run['ate']['rmse']:>7.3f}",
        f"{'ATE median (m)':25s}{ours_run['ate']['median']:>6.3f}{orb_run['ate']['median']:>7.3f}",
        f"{'RPE trans RMSE (m)':25s}{ours_run['rpe']['trans_rmse']:>6.3f}{orb_run['rpe']['trans_rmse']:>7.3f}",
        f"{'RPE rot RMSE (deg)':25s}{ours_run['rpe']['rot_rmse_deg']:>6.2f}{orb_run['rpe']['rot_rmse_deg']:>7.2f}",
        f"{'Fixed scale':25s}{ours_run['scale']:>6.2f}{orb_run['scale']:>7.2f}",
        f"{'Success rate (%)':25s}{100*ours_success:>6.1f}{100*orb_success:>7.1f}",
        "",
        f"{'Mean reproj WPnP (px)':25s}{ours_reproj_mean:>6.2f}{'N/A':>7s}",
        f"{'Mean reproj inliers (px)':25s}{ours_reproj_inl:>6.2f}{'N/A':>7s}",
        "",
        "Umeyama (Ours, ref only)",
        "─" * 38,
        f"scale={u_scale:.3f}",
        f"tx={u_t[0]:.3f}  ty={u_t[1]:.3f}  tz={u_t[2]:.3f}",
        f"roll={u_roll:.1f}°  pitch={u_pitch:.1f}°  yaw={u_yaw:.1f}°",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=9,
            va="top", ha="left", transform=ax.transAxes)

    # (1,0) RPE translational
    ax = fig.add_subplot(2, 3, 4)
    if len(ours_run["rpe_trans"]):
        ax.plot(ours_run["rpe_ts"] - ours_run["rpe_ts"][0], ours_run["rpe_trans"],
                color="#2196F3", lw=1,
                label=f"Our Pipeline (RMSE {ours_run['rpe']['trans_rmse']:.3f}m)")
    if len(orb_run["rpe_trans"]):
        ax.plot(orb_run["rpe_ts"] - orb_run["rpe_ts"][0], orb_run["rpe_trans"],
                color="#F44336", lw=1,
                label=f"ORB-SLAM3 Loc (RMSE {orb_run['rpe']['trans_rmse']:.3f}m)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel(f"RPE trans (m / {args.rpe_delta}s)")
    ax.set_title("Relative Pose Error — translational")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,1) RPE rotational
    ax = fig.add_subplot(2, 3, 5)
    if len(ours_run["rpe_rot"]):
        ax.plot(ours_run["rpe_ts"] - ours_run["rpe_ts"][0], ours_run["rpe_rot"],
                color="#2196F3", lw=1,
                label=f"Our Pipeline (RMSE {ours_run['rpe']['rot_rmse_deg']:.2f}°)")
    if len(orb_run["rpe_rot"]):
        ax.plot(orb_run["rpe_ts"] - orb_run["rpe_ts"][0], orb_run["rpe_rot"],
                color="#F44336", lw=1,
                label=f"ORB-SLAM3 Loc (RMSE {orb_run['rpe']['rot_rmse_deg']:.2f}°)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel(f"RPE rot (deg / {args.rpe_delta}s)")
    ax.set_title("Relative Pose Error — rotational")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
