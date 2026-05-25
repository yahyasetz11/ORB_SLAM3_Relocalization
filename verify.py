import pandas as pd
import numpy as np
import sys

df = pd.read_csv("comparison_log.csv")

# 1. Coverage check: wpnp should now have an entry on every row where std succeeded
total = len(df)
std_rows = df[df["std_x"].notna()].shape[0]
wpnp_rows = df[df["wpnp_x"].notna() & (df["wpnp_inliers"] > 0)].shape[0]
print(f"Total rows: {total}")
print(f"Std PnP rows: {std_rows}")
print(f"WPnP rows: {wpnp_rows}")
if wpnp_rows < std_rows * 0.9:
    print("FAIL: WPnP coverage is still much lower than Std PnP")
    sys.exit(1)
print("PASS: coverage looks comparable")

# 2. Agreement check: on frames with wpnp_success, positions should be very close
# when bg_weight=1.0 (all weights uniform) — both do LM(uniform) on same inlier set
# Filter to rows where wpnp succeeded
ok = df[df["wpnp_inliers"] > 0].copy()
dx = ok["wpnp_x"] - ok["std_x"]
dy = ok["wpnp_y"] - ok["std_y"]
dz = ok["wpnp_z"] - ok["std_z"]
dist = np.sqrt(dx**2 + dy**2 + dz**2)
print(f"\nPose delta (wpnp vs std) — mean: {dist.mean():.4f}m  max: {dist.max():.4f}m  median: {dist.median():.4f}m")
print("(With bg_weight=1.0 and landmark_weight=1.0, mean should be < 0.005m on most sequences)")