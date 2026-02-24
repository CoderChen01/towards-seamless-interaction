# metrics_fast.py
# Optimized, reusable metrics for DualTalk-style evaluations.
# Pure functions over arrays; no file I/O inside.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import linalg
from sklearn.cluster import KMeans

# ----------------------------- Utilities -----------------------------

EPS = 1e-6


def _activation_stats(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exact replica of reference: mean + np.cov(rowvar=False) with default dtypes."""
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def _frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = EPS,
) -> float:
    """Exact replica of reference FID implementation."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError("Training and test mean vectors have different lengths")
    if sigma1.shape != sigma2.shape:
        raise ValueError("Training and test covariances have different dimensions")

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # mirror the reference code's stabilization path
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        # keep the same tolerance and handling as reference
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def _variance_mean(X: np.ndarray) -> float:
    """Exact replica: mean of per-dimension var with NumPy defaults."""
    return float(np.mean(np.var(X, axis=0)))


def _diversity(X: np.ndarray, diversity_times: int) -> float:
    """Exact replica of reference calculate_diversity (asserts + scipy.linalg.norm)."""
    assert len(X.shape) == 2
    assert X.shape[0] >= diversity_times
    num_samples = X.shape[0]
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(X[first_indices] - X[second_indices], axis=1)
    return float(dist.mean())


def _sid(gt: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Exact replica of reference calcuate_sid (KMeans settings + looped histogram + log2)."""
    # run kmeans on gt
    kmeans_gt = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(gt)
    # predict on pred
    labels = kmeans_gt.predict(pred)

    # manual histogram loop (to match reference exactly)
    hist_cnt = [0] * k
    for i in range(len(labels)):
        hist_cnt[labels[i]] += 1
    hist_cnt = np.array(hist_cnt, dtype=float)
    hist_cnt = hist_cnt / np.sum(hist_cnt)

    # entropy with eps and base-2 log as in reference
    entropy_val = 0.0
    eps = 1e-6
    for i in range(k):
        entropy_val += hist_cnt[i] * np.log2(hist_cnt[i] + eps)
    return float(-entropy_val)


def _sts(x: np.ndarray, y: np.ndarray, timestep: float = 0.1) -> float:
    """Exact replica of reference sts (double for-loops and per-dimension accumulation)."""
    ans = 0.0
    total_sample, dim = x.shape
    for di in range(dim):
        for i in range(1, total_sample):
            ans += (
                (x[i][di] - x[i - 1][di]) - (y[i][di] - y[i - 1][di])
            ) ** 2 / timestep
    return float(np.sqrt(ans))


def _rpcc(gt: np.ndarray, pred: np.ndarray, anchor: np.ndarray) -> float:
    """Exact replica using np.corrcoef like reference."""
    pcc_xy = np.corrcoef(
        gt.reshape(
            -1,
        ),
        anchor.reshape(
            -1,
        ),
    )[0, 1]
    pcc_xypred = np.corrcoef(
        pred.reshape(
            -1,
        ),
        anchor.reshape(
            -1,
        ),
    )[0, 1]
    return float(abs(pcc_xy - pcc_xypred))


# ----------------------------- Public API -----------------------------


@dataclass(frozen=True)
class MetricConfig:
    """Configuration for metric computation (values match reference usage)."""

    sid_k_exp: int = 40
    sid_k_jaw: int = 10
    sid_k_pose: int = 10
    diversity_pairs: int = 100
    timestep: float = 0.1


def compute_sample_metrics(
    pre: np.ndarray,
    gt_exp: np.ndarray,
    gt_pose: np.ndarray,  # shape (T, >=3) where [:, :3] are pose, [:, 3:] are jaw
    anchor_exp: np.ndarray,
    anchor_pose: np.ndarray,  # same layout as gt_pose
    cfg: MetricConfig = MetricConfig(),
) -> Dict[str, float]:
    """Compute all metrics for a single sample EXACTLY as in the reference script."""
    # Split predicted channels
    pre_exp = pre[:, :50]
    pre_jaw = pre[:, 50:53]
    pre_pose = pre[:, 53:]

    # Split GT and anchor channels
    gt_rot = gt_pose[:, :3]
    gt_jaw = gt_pose[:, 3:]

    anchor_rot = anchor_pose[:, :3]
    anchor_jaw = anchor_pose[:, 3:]

    # Align by minimum length
    T = min(
        pre.shape[0],
        gt_exp.shape[0],
        gt_jaw.shape[0],
        gt_rot.shape[0],
        anchor_exp.shape[0],
        anchor_jaw.shape[0],
        anchor_rot.shape[0],
    )
    pre_exp = pre_exp[:T]
    pre_jaw = pre_jaw[:T]
    pre_pose = pre_pose[:T]
    gt_exp = gt_exp[:T]
    gt_jaw = gt_jaw[:T]
    gt_rot = gt_rot[:T]
    anchor_exp = anchor_exp[:T]
    anchor_jaw = anchor_jaw[:T]
    anchor_rot = anchor_rot[:T]

    # FID (exp, jaw, pose)
    m1, c1 = _activation_stats(pre_exp)
    m2, c2 = _activation_stats(gt_exp)
    fid_exp = _frechet_distance(m1, c1, m2, c2)

    m1, c1 = _activation_stats(pre_jaw)
    m2, c2 = _activation_stats(gt_jaw)
    fid_jaw = _frechet_distance(m1, c1, m2, c2)

    m1, c1 = _activation_stats(pre_pose)
    m2, c2 = _activation_stats(gt_rot)
    fid_pose = _frechet_distance(m1, c1, m2, c2)

    # Paired FID with anchor concatenation (axis = -1), exactly like reference
    gm, gc = _activation_stats(np.concatenate([anchor_exp, gt_exp], axis=-1))
    pm, pc = _activation_stats(np.concatenate([anchor_exp, pre_exp], axis=-1))
    p_fid_exp = _frechet_distance(gm, gc, pm, pc)

    gm, gc = _activation_stats(np.concatenate([anchor_jaw, gt_jaw], axis=-1))
    pm, pc = _activation_stats(np.concatenate([anchor_jaw, pre_jaw], axis=-1))
    p_fid_jaw = _frechet_distance(gm, gc, pm, pc)

    gm, gc = _activation_stats(np.concatenate([anchor_rot, gt_rot], axis=-1))
    pm, pc = _activation_stats(np.concatenate([anchor_rot, pre_pose], axis=-1))
    p_fid_pose = _frechet_distance(gm, gc, pm, pc)

    # MSE
    mse_exp = float(np.mean((pre_exp - gt_exp) ** 2))
    mse_jaw = float(np.mean((pre_jaw - gt_jaw) ** 2))
    mse_pose = float(np.mean((pre_pose - gt_rot) ** 2))

    # SID
    sid_exp = _sid(gt_exp, pre_exp, cfg.sid_k_exp)
    sid_jaw = _sid(gt_jaw, pre_jaw, cfg.sid_k_jaw)
    sid_pose = _sid(gt_rot, pre_pose, cfg.sid_k_pose)

    reverse_sid_exp = _sid(pre_exp, gt_exp, cfg.sid_k_exp)
    reverse_sid_jaw = _sid(pre_jaw, gt_jaw, cfg.sid_k_jaw)
    reverse_sid_pose = _sid(pre_pose, gt_rot, cfg.sid_k_pose)

    # STS
    sts_exp = _sts(pre_exp, gt_exp, cfg.timestep)
    sts_jaw = _sts(pre_jaw, gt_jaw, cfg.timestep)
    sts_pose = _sts(pre_pose, gt_rot, cfg.timestep)

    # Diversity (assert same sample-size condition)
    div_pre_exp = _diversity(pre_exp, cfg.diversity_pairs)
    div_pre_jaw = _diversity(pre_jaw, cfg.diversity_pairs)
    div_pre_pose = _diversity(pre_pose, cfg.diversity_pairs)

    div_gt_exp = _diversity(gt_exp, cfg.diversity_pairs)
    div_gt_jaw = _diversity(gt_jaw, cfg.diversity_pairs)
    div_gt_pose = _diversity(gt_rot, cfg.diversity_pairs)

    # Variance
    var_pre_exp = _variance_mean(pre_exp)
    var_pre_jaw = _variance_mean(pre_jaw)
    var_pre_pose = _variance_mean(pre_pose)

    var_gt_exp = _variance_mean(gt_exp)
    var_gt_jaw = _variance_mean(gt_jaw)
    var_gt_pose = _variance_mean(gt_rot)

    # RPCC
    rpcc_exp = _rpcc(gt_exp, pre_exp, anchor_exp)
    rpcc_jaw = _rpcc(gt_jaw, pre_jaw, anchor_jaw)
    rpcc_pose = _rpcc(gt_rot, pre_pose, anchor_rot)

    return {
        "fid_exp": fid_exp,
        "fid_jaw": fid_jaw,
        "fid_pose": fid_pose,
        "p_fid_exp": p_fid_exp,
        "p_fid_jaw": p_fid_jaw,
        "p_fid_pose": p_fid_pose,
        "mse_exp": mse_exp,
        "mse_jaw": mse_jaw,
        "mse_pose": mse_pose,
        "sid_exp": sid_exp,
        "sid_jaw": sid_jaw,
        "sid_pose": sid_pose,
        "reverse_sid_exp": reverse_sid_exp,
        "reverse_sid_jaw": reverse_sid_jaw,
        "reverse_sid_pose": reverse_sid_pose,
        "sts_exp": sts_exp,
        "sts_jaw": sts_jaw,
        "sts_pose": sts_pose,
        "diversity_pre_exp": div_pre_exp,
        "diversity_pre_jaw": div_pre_jaw,
        "diversity_pre_pose": div_pre_pose,
        "diversity_gt_exp": div_gt_exp,
        "diversity_gt_jaw": div_gt_jaw,
        "diversity_gt_pose": div_gt_pose,
        "variance_pre_exp": var_pre_exp,
        "variance_pre_jaw": var_pre_jaw,
        "variance_pre_pose": var_pre_pose,
        "variance_gt_exp": var_gt_exp,
        "variance_gt_jaw": var_gt_jaw,
        "variance_gt_pose": var_gt_pose,
        "rpcc_exp": rpcc_exp,
        "rpcc_jaw": rpcc_jaw,
        "rpcc_pose": rpcc_pose,
    }


def average_metrics(
    samples: Dict[str, float] | list[Dict[str, float]],
) -> Dict[str, float]:
    """Mean aggregation with NumPy defaults (matches reference np.mean behavior)."""
    if isinstance(samples, dict):
        return samples
    if not samples:
        return {}
    keys = samples[0].keys()
    return {k: float(np.mean([d[k] for d in samples])) for k in keys}
