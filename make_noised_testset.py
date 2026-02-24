import argparse
import json
import math
import os
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


def swap_speaker(name: str) -> str:
    if name.endswith("speaker1"):
        return name.replace("speaker1", "speaker2")
    if name.endswith("speaker2"):
        return name.replace("speaker2", "speaker1")
    return name


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def geom_len_from_uniform(u: float, mean_len: int) -> int:
    """
    Sample geometric length L>=1 with mean approximately mean_len using inverse CDF.
    """
    mean_len = max(1, int(mean_len))
    p = 1.0 / float(mean_len)
    u = min(max(u, 1e-6), 1.0 - 1e-6)
    denom = math.log(max(1e-6, 1.0 - p))
    L = int((math.log1p(-u) / denom)) + 1
    return max(1, L)


def make_burst_mask(
    T: int, prob: float, mean_len: int, rng: np.random.RandomState
) -> np.ndarray:
    """
    Burst mask with approximate overall ratio ~ prob via burst starts:
      p_start * mean_len ≈ prob
    """
    if prob <= 0:
        return np.zeros((T,), dtype=bool)
    if prob >= 1:
        return np.ones((T,), dtype=bool)

    mean_len = max(1, int(mean_len))
    p_start = min(1.0, float(prob) / float(mean_len))

    mask = np.zeros((T,), dtype=bool)
    t = 0
    while t < T:
        if rng.rand() < p_start:
            L = geom_len_from_uniform(float(rng.rand()), mean_len)
            mask[t : min(T, t + L)] = True
            t += L
        else:
            t += 1
    return mask


def hold_last_fill(x: np.ndarray, drop_mask: np.ndarray) -> np.ndarray:
    """
    x: [T, D], drop_mask: [T] bool. For dropped frames, copy previous frame.
    """
    if not drop_mask.any():
        return x
    out = x.copy()
    for t in range(1, out.shape[0]):
        if drop_mask[t]:
            out[t] = out[t - 1]
    return out


def sample_laplace(shape, scale: float, rng: np.random.RandomState) -> np.ndarray:
    u = rng.rand(*shape) - 0.5
    return -scale * np.sign(u) * np.log1p(-2.0 * np.abs(u))


def load_stat_info(stat_path: str):
    with open(stat_path, "r") as f:
        stat_dict = json.load(f)
    mean_exp = stat_dict["exp"]["mean"]
    std_exp = stat_dict["exp"]["std"]
    mean_jaw = stat_dict["jawpose"]["mean"]
    std_jaw = stat_dict["jawpose"]["std"]
    mean_neck = stat_dict["neck"]["mean"]
    std_neck = stat_dict["neck"]["std"]

    mean_all = np.concatenate([mean_exp, mean_jaw, mean_neck], axis=0)
    std_all = np.concatenate([std_exp, std_jaw, std_neck], axis=0)
    return mean_all, std_all


def add_motion_noise_burst(
    x: np.ndarray,
    burst_mask: np.ndarray,
    noise_type: str,
    noise_scale: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Apply noise only on frames where burst_mask=True.
    noise_scale:
      - gaussian: std
      - laplace:  scale (b)
      - uniform:  half-range (U[-scale, +scale])
    """
    if noise_scale <= 0 or not burst_mask.any():
        return x

    out = x.copy()
    idx = np.where(burst_mask)[0]
    n = len(idx)
    D = x.shape[1]

    if noise_type == "gaussian":
        eps = rng.randn(n, D).astype(np.float32) * float(noise_scale)
        out[idx] = out[idx] + eps
    elif noise_type == "laplace":
        out[idx] = out[idx] + sample_laplace((n, D), float(noise_scale), rng).astype(
            np.float32
        )
    elif noise_type == "uniform":
        eps = rng.uniform(low=-noise_scale, high=noise_scale, size=(n, D)).astype(
            np.float32
        )
        out[idx] = out[idx] + eps
    else:
        raise ValueError(f"Unknown motion noise_type: {noise_type}")

    return out


def apply_speech_silence_burst(
    wav: np.ndarray,
    prob: float,
    mean_len: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Burst silence: set samples to 0 on a burst mask.
    Returns (wav_out, silence_mask) where silence_mask is sample-level bool [T].
    """
    T = wav.shape[0]
    if prob <= 0:
        return wav.astype(np.float32), np.zeros((T,), dtype=bool)

    mask = make_burst_mask(T, prob, mean_len, rng)
    out = wav.astype(np.float32).copy()
    out[mask] = 0.0
    return out, mask


def apply_speech_noise_burst(
    wav: np.ndarray,
    prob: float,
    mean_len: int,
    snr_db: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Burst noise: add AWGN only on burst segments (mask=True).
    We compute noise std from the *full* signal power (stable), then add noise on masked samples.
    Returns (wav_out, noise_mask) sample-level bool [T].
    """
    T = wav.shape[0]
    if prob <= 0:
        return wav.astype(np.float32), np.zeros((T,), dtype=bool)

    mask = make_burst_mask(T, prob, mean_len, rng)
    out = wav.astype(np.float32).copy()

    # compute noise std from global SNR target
    sig_pow = float(np.mean(out**2))
    sig_pow = max(sig_pow, 1e-12)
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    noise_pow = sig_pow / snr_lin
    noise_std = math.sqrt(max(noise_pow, 1e-12))

    noise = rng.randn(T).astype(np.float32) * noise_std
    out[mask] = out[mask] + noise[mask]
    return out, mask


def maybe_corrupt_speech_burst(
    wav: np.ndarray,
    sr: int,
    # burst noise
    do_noise: bool,
    noise_prob: float,
    noise_mean_sec: float,
    snr_db: float,
    # burst silence
    do_silence: bool,
    silence_prob: float,
    silence_mean_sec: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, dict]:
    """
    Burst-only speech corruption:
      - noise_mask: burst mask for AWGN
      - silence_mask: burst mask for silencing
    """
    out = wav.astype(np.float32)

    meta = {
        "noise_mask": np.zeros((out.shape[0],), dtype=bool),
        "silence_mask": np.zeros((out.shape[0],), dtype=bool),
    }

    if do_noise and noise_prob > 0:
        mean_len = max(1, int(float(noise_mean_sec) * sr))
        out, nmask = apply_speech_noise_burst(
            out, prob=noise_prob, mean_len=mean_len, snr_db=snr_db, rng=rng
        )
        meta["noise_mask"] = nmask

    if do_silence and silence_prob > 0:
        mean_len = max(1, int(float(silence_mean_sec) * sr))
        out, smask = apply_speech_silence_burst(
            out, prob=silence_prob, mean_len=mean_len, rng=rng
        )
        meta["silence_mask"] = smask

    return out.astype(np.float32), meta


def load_motion_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path)
    exp = d["exp"].astype(np.float32)
    pose = d["pose"].astype(np.float32)
    return exp, pose


def save_motion_npz(out_path: str, exp: np.ndarray, pose: np.ndarray):
    np.savez(out_path, exp=exp.astype(np.float32), pose=pose.astype(np.float32))


def build_config_tag(args) -> str:
    tags = []

    # speech (user) - burst only
    if args.user_speech_noise and args.user_speech_noise_prob > 0:
        tags.append(f"uSpN_B{args.user_speech_noise_prob}")
        tags.append(f"snr{args.user_speech_snr_db}")
        tags.append(f"nlen{args.user_speech_noise_mean_sec}")

    if args.user_speech_silence and args.user_speech_silence_prob > 0:
        tags.append(f"uSpSil_B{args.user_speech_silence_prob}")
        tags.append(f"slen{args.user_speech_silence_mean_sec}")

    # speech (agent) - burst only
    if args.agent_speech_noise and args.agent_speech_noise_prob > 0:
        tags.append(f"aSpN_B{args.agent_speech_noise_prob}")
        tags.append(f"snr{args.agent_speech_snr_db}")
        tags.append(f"nlen{args.agent_speech_noise_mean_sec}")

    if args.agent_speech_silence and args.agent_speech_silence_prob > 0:
        tags.append(f"aSpSil_B{args.agent_speech_silence_prob}")
        tags.append(f"slen{args.agent_speech_silence_mean_sec}")

    # motion (user) - burst only
    if args.user_motion_drop_prob > 0:
        tags.append(f"uMoD_B{args.user_motion_drop_prob}")
        tags.append(f"dlen{args.user_motion_drop_mean_sec}")

    if args.user_motion_noise_prob > 0 and args.user_motion_noise_scale > 0:
        ntype = args.user_motion_noise_type[0].upper()
        tags.append(f"uMoN_B{ntype}{args.user_motion_noise_prob}")
        tags.append(f"nsc{args.user_motion_noise_scale}")

    if len(tags) == 0:
        return "clean"

    return "_".join(tags)


def main():
    parser = argparse.ArgumentParser("Make noised test set (user/agent) - burst-only")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Original dataset directory containing .wav and .npz",
    )
    parser.add_argument("--seed", type=int, default=6666)

    # speech (user): burst noise/silence only
    parser.add_argument("--user_speech_noise", action="store_true")
    parser.add_argument(
        "--user_speech_noise_prob",
        type=float,
        default=0.0,
        help="Burst occupancy ratio for noise.",
    )
    parser.add_argument(
        "--user_speech_noise_mean_sec",
        type=float,
        default=0.2,
        help="Mean burst length (sec) for noise.",
    )
    parser.add_argument("--user_speech_snr_db", type=float, default=60.0)

    parser.add_argument("--user_speech_silence", action="store_true")
    parser.add_argument(
        "--user_speech_silence_prob",
        type=float,
        default=0.0,
        help="Burst occupancy ratio for silence.",
    )
    parser.add_argument(
        "--user_speech_silence_mean_sec",
        type=float,
        default=1.0,
        help="Mean burst length (sec) for silence.",
    )

    # speech (agent): burst noise/silence only
    parser.add_argument("--agent_speech_noise", action="store_true")
    parser.add_argument("--agent_speech_noise_prob", type=float, default=0.0)
    parser.add_argument("--agent_speech_noise_mean_sec", type=float, default=0.2)
    parser.add_argument("--agent_speech_snr_db", type=float, default=60.0)

    parser.add_argument("--agent_speech_silence", action="store_true")
    parser.add_argument("--agent_speech_silence_prob", type=float, default=0.0)
    parser.add_argument("--agent_speech_silence_mean_sec", type=float, default=0.05)

    # motion (user): burst drop + burst noise
    parser.add_argument(
        "--user_motion_drop_prob",
        type=float,
        default=0.0,
        help="Burst occupancy ratio for motion drop.",
    )
    parser.add_argument("--user_motion_drop_mean_sec", type=float, default=0.2)
    parser.add_argument("--motion_fps", type=int, default=25)

    parser.add_argument(
        "--user_motion_noise_prob",
        type=float,
        default=0.0,
        help="Burst occupancy ratio for motion noise.",
    )
    parser.add_argument(
        "--user_motion_noise_mean_sec",
        type=float,
        default=0.2,
        help="Mean burst length (sec) for motion noise.",
    )
    parser.add_argument(
        "--user_motion_noise_type",
        type=str,
        default="laplace",
        choices=["laplace", "uniform", "gaussian"],
        help="Motion noise distribution type.",
    )

    # io
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--stat_path",
        type=str,
        default=None,
        help="Path to stat_flame.json. If None, use data_dir.parent/stat_flame.json",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    assert data_dir.exists(), f"Missing data_dir: {data_dir}"

    stat_path = args.stat_path
    if stat_path is None:
        stat_path = str(data_dir.parent / "stat_flame.json")
    _, std_all = load_stat_info(stat_path)

    # set a default noise scale from stats (kept identical to your original intent)
    args.user_motion_noise_scale = round(float(np.mean(std_all).item()), 3)

    config_tag = build_config_tag(args)
    out_dir = data_dir.parent / f"noised_{config_tag}_{data_dir.name}"
    user_dir = out_dir / "user"
    agent_dir = out_dir / "agent"
    ensure_dir(str(user_dir))
    ensure_dir(str(agent_dir))

    rng = np.random.RandomState(args.seed)

    wav_files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() == ".wav"])
    for wav_path in tqdm(wav_files, desc="generate noised set", ncols=100):
        stem = wav_path.stem
        swapped = swap_speaker(stem)

        wav_agent_path = data_dir / f"{stem}.wav"
        wav_user_path = data_dir / f"{swapped}.wav"
        npz_user_path = data_dir / f"{swapped}.npz"

        if (
            (not wav_user_path.exists())
            or (not wav_agent_path.exists())
            or (not npz_user_path.exists())
        ):
            continue

        sample_id = swapped

        out_user_wav = user_dir / f"{sample_id}.wav"
        out_user_npz = user_dir / f"{sample_id}.npz"
        out_agent_wav = agent_dir / f"{sample_id}.wav"

        if (
            (not args.overwrite)
            and out_user_wav.exists()
            and out_user_npz.exists()
            and out_agent_wav.exists()
        ):
            continue

        # --- load raw speech ---
        wav_u, _ = librosa.load(str(wav_user_path), sr=args.sr)
        wav_a, _ = librosa.load(str(wav_agent_path), sr=args.sr)
        wav_u = wav_u.astype(np.float32)
        wav_a = wav_a.astype(np.float32)

        # --- burst corrupt user speech ---
        wav_u_noised, _meta_u = maybe_corrupt_speech_burst(
            wav_u,
            sr=args.sr,
            do_noise=args.user_speech_noise,
            noise_prob=args.user_speech_noise_prob,
            noise_mean_sec=args.user_speech_noise_mean_sec,
            snr_db=args.user_speech_snr_db,
            do_silence=args.user_speech_silence,
            silence_prob=args.user_speech_silence_prob,
            silence_mean_sec=args.user_speech_silence_mean_sec,
            rng=rng,
        )

        # --- burst corrupt agent speech ---
        wav_a_noised, _meta_a = maybe_corrupt_speech_burst(
            wav_a,
            sr=args.sr,
            do_noise=args.agent_speech_noise,
            noise_prob=args.agent_speech_noise_prob,
            noise_mean_sec=args.agent_speech_noise_mean_sec,
            snr_db=args.agent_speech_snr_db,
            do_silence=args.agent_speech_silence,
            silence_prob=args.agent_speech_silence_prob,
            silence_mean_sec=args.agent_speech_silence_mean_sec,
            rng=rng,
        )

        # --- load user motion ---
        exp, pose = load_motion_npz(str(npz_user_path))
        Tm = exp.shape[0]
        assert pose.shape[0] == Tm, "exp/pose length mismatch"
        motion = np.concatenate(
            [exp, pose[:, 3:], pose[:, :3]], axis=1
        )  # [T,56] (exp, jaw, neck)

        # burst drop mask (frames)
        drop_mean_len = max(1, int(args.user_motion_drop_mean_sec * args.motion_fps))
        drop_mask = (
            make_burst_mask(Tm, args.user_motion_drop_prob, drop_mean_len, rng)
            if args.user_motion_drop_prob > 0
            else np.zeros((Tm,), dtype=bool)
        )

        # burst noise mask (frames)
        noise_mean_len = max(1, int(args.user_motion_noise_mean_sec * args.motion_fps))
        noise_burst_mask = (
            make_burst_mask(Tm, args.user_motion_noise_prob, noise_mean_len, rng)
            if args.user_motion_noise_prob > 0
            else np.zeros((Tm,), dtype=bool)
        )

        # apply burst noise then hold-last burst drop
        motion = add_motion_noise_burst(
            motion,
            noise_burst_mask,
            args.user_motion_noise_type,
            args.user_motion_noise_scale,
            rng,
        )
        motion = hold_last_fill(motion, drop_mask)

        # write back to exp/pose
        exp_new = motion[:, :50]
        jaw_new = motion[:, 50:53]
        neck_new = motion[:, 53:56]
        pose_new = np.concatenate([neck_new, jaw_new], axis=1)

        # save
        sf.write(str(out_user_wav), wav_u_noised, args.sr)
        sf.write(str(out_agent_wav), wav_a_noised, args.sr)
        save_motion_npz(str(out_user_npz), exp_new, pose_new)

    print(f"[DONE] noised dataset saved to: {out_dir}")


if __name__ == "__main__":
    main()
