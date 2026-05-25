# Absolute Trajectory Error (ATE)

## Per-frame error

$$
\text{ATE}_i = \left\| \mathbf{p}_{\text{est},i} - \mathbf{p}_{\text{gt},i} \right\|_2
$$

## RMSE over all frames

$$
\text{ATE}_{\text{RMSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left\| \mathbf{p}_{\text{est},i} - \mathbf{p}_{\text{gt},i} \right\|_2^2}
$$

**Where:**
- $\mathbf{p}_{\text{est},i}$ — estimated camera position at frame $i$, after alignment to GT frame
- $\mathbf{p}_{\text{gt},i}$ — ground-truth camera position at frame $i$
- $N$ — total number of matched frames
- $\|\cdot\|_2$ — Euclidean (L2) norm in 3D space (metres)

## Alignment

Before computing ATE, the estimated trajectory is aligned to the ground-truth frame using a Sim(3) transform:

$$
\hat{\mathbf{p}}_{\text{est},i} = s \cdot \mathbf{R}\, \mathbf{p}_{\text{est},i} + \mathbf{t}
$$

where $s$ is scale, $\mathbf{R} \in SO(3)$ is rotation, and $\mathbf{t} \in \mathbb{R}^3$ is translation.

In `plot3.py` this transform is found automatically via **Umeyama** (least-squares Sim(3) fit).  
In `plot_benchmark_3.py` $(s, \mathbf{R}, \mathbf{t})$ are fixed parameters hardcoded from `benchmark_params.yaml`.
