# driver_parallel.py
# Parallel driver that calls your reusable metrics functions without coupling them to I/O.

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# import from your module
from util.metrics import MetricConfig, average_metrics, compute_sample_metrics

# --------------------------- Environment knobs ---------------------------
# Prevent over-subscription: processes handle parallelism; BLAS stays single-threaded.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# --------------------------- Helper: file manifest ---------------------------
def list_pred_files(pred_dir: Path) -> List[str]:
    """Return a stable-sorted list of prediction .npy file names."""
    return sorted([f for f in os.listdir(pred_dir) if f.endswith(".npy")])


def infer_anchor_npz(gt_dir: Path, pred_file: str) -> Tuple[Path, Path]:
    """Given a prediction file name, return (gt_npz_path, anchor_npz_path)."""
    gt_npz = gt_dir / pred_file.replace(".npy", ".npz")
    if pred_file.endswith("speaker1.npy"):
        anchor_npz = gt_dir / pred_file.replace("speaker1.npy", "speaker2.npz")
    else:
        anchor_npz = gt_dir / pred_file.replace("speaker2.npy", "speaker1.npz")
    return gt_npz, anchor_npz


# --------------------------- Worker ---------------------------
def process_one_sample(
    pred_dir: Path,
    gt_dir: Path,
    pred_file: str,
    seed: int,
) -> dict:
    """Load arrays for one sample and compute metrics."""
    # Independent RNG per worker for diversity sampling, derived from a global seed + filename hash
    # This keeps results reproducible across runs and independent of chunking.
    cfg = MetricConfig()

    # Load prediction (npy supports mmap)
    pre = np.load(pred_dir / pred_file, mmap_mode="r")

    # Load GT/anchor (npz cannot mmap). Avoid allow_pickle for safety.
    gt_npz, anchor_npz = infer_anchor_npz(gt_dir, pred_file)
    gt = np.load(gt_npz, allow_pickle=False)
    anchor = np.load(anchor_npz, allow_pickle=False)

    # Compute metrics using the pure functions
    return compute_sample_metrics(
        pre=pre,
        gt_exp=gt["exp"],
        gt_pose=gt["pose"],
        anchor_exp=anchor["exp"],
        anchor_pose=anchor["pose"],
        cfg=cfg,
    )


def format_metric_for_pub(metrics: dict) -> dict:
    metrics = metrics.copy()
    rv = OrderedDict()
    rv["fid_exp"] = round(metrics["fid_exp"], 2)
    rv["fid_jaw"] = round(metrics["fid_jaw"] * 10**3, 2)
    rv["fid_pose"] = round(metrics["fid_pose"] * 10**2, 2)
    rv["p_fid_exp"] = round(metrics["p_fid_exp"], 2)
    rv["p_fid_jaw"] = round(metrics["p_fid_jaw"] * 10**3, 2)
    rv["p_fid_pose"] = round(metrics["p_fid_pose"] * 10**2, 2)
    rv["mse_exp"] = round(metrics["mse_exp"] * 10, 2)
    rv["mse_jaw"] = round(metrics["mse_jaw"] * 10**3, 2)
    rv["mse_pose"] = round(metrics["mse_pose"] * 10**2, 2)
    rv["reverse_sid_exp"] = round(metrics["reverse_sid_exp"], 2)
    rv["reverse_sid_jaw"] = round(metrics["reverse_sid_jaw"], 2)
    rv["reverse_sid_pose"] = round(metrics["reverse_sid_pose"], 2)
    rv["sid_exp"] = round(metrics["sid_exp"], 2)
    rv["sid_jaw"] = round(metrics["sid_jaw"], 2)
    rv["sid_pose"] = round(metrics["sid_pose"], 2)
    rv["rpcc_exp"] = round(metrics["rpcc_exp"] * 10**2, 2)
    rv["rpcc_jaw"] = round(metrics["rpcc_jaw"] * 10, 2)
    rv["rpcc_pose"] = round(metrics["rpcc_pose"] * 10, 2)
    return rv


def calc_metrics(args):
    pred_dir = Path(args.pred_motion_path)
    gt_dir = Path(args.gt_motion_path)
    if args.output_path is None:
        out_path = pred_dir.with_name(pred_dir.name + ".metric.json")
    else:
        out_path = Path(args.output_path)
    global_seed = 1234

    pred_files = list_pred_files(pred_dir)

    # Filter NAN files
    filtered_pred_files = []
    nan_file_names = []
    for f in pred_files:
        pred_fpath = pred_dir / f
        pred_array = np.load(pred_fpath, mmap_mode="r")
        if np.isnan(pred_array).any():
            print("NaN in pred, skip", f)
            nan_file_names.append(f)
            continue
        filtered_pred_files.append(f)
    pred_files = filtered_pred_files
    print(f"Found {len(pred_files)} prediction files after filtering NaN.")
    if len(nan_file_names) > 0:
        with open(out_path.with_suffix(".nan_files.txt"), "w") as f:
            for name in nan_file_names:
                f.write(name + "\n")
    if not (len(pred_files) == 1066 or len(pred_files) == 754):
        raise ValueError("Predictions is incomplete!!!")
    # Parallel execution across samples
    # backend="loky" -> process-based; good for CPU-bound numpy/scikit code.
    if not args.is_serial_computing:
        results = Parallel(n_jobs=-1, backend="loky", batch_size=1, verbose=0)(
            delayed(process_one_sample)(pred_dir, gt_dir, f, global_seed)
            for f in tqdm(pred_files, desc="Computing metrics", ncols=100)
        )
    else:
        results = []
        for f in tqdm(pred_files, desc="Computing metrics", ncols=100):
            results.append(process_one_sample(pred_dir, gt_dir, f, global_seed))

    mean_res = average_metrics(results)
    res4pub = format_metric_for_pub(mean_res)
    # Print a concise summary
    print("=== Final averaged metrics ===")
    data_keys = "\t".join(res4pub.keys())
    print(data_keys)
    data_items = "\t".join([str(v) for v in res4pub.values()])
    print(data_items)

    # Persist to JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(mean_res, f, ensure_ascii=False, indent=2)
    with open(out_path.with_suffix(".for_pub.tsv"), "w") as f:
        f.write(data_keys + "\n")
        f.write(data_items)


# --------------------------- Orchestrator ---------------------------
def main():
    parser = argparse.ArgumentParser("MAR training with Diffusion Loss", add_help=False)
    parser.add_argument(
        "--pred_motion_path",
        type=str,
    )
    parser.add_argument("--gt_motion_path", type=str)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--is_serial_computing", action="store_true")

    args = parser.parse_args()
    calc_metrics(args)


if __name__ == "__main__":
    main()
