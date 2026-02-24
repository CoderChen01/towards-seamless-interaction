import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
from transformers import Wav2Vec2Processor


def is_main_process() -> bool:
    # Minimal replacement: safe in single-node torchrun
    rank = int(os.environ.get("RANK", "0"))
    return rank == 0


class BSDatasetRobustNoised(data.Dataset):
    """
    BSDataset-like:
      __getitem__ returns (file_name, speech_1, motion_1, speech_2)
    """

    def __init__(self, meta: List[Dict], data_type: str = "test"):
        self.data = meta
        self.len = len(meta)
        self.data_type = data_type

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        d = self.data[index]
        file_name = d["name"]
        speech_1 = d["audio_user"]  # user speech (Wav2Vec input_values)
        speech_2 = d["audio_agent"]  # agent speech (Wav2Vec input_values)

        exp = d["exp"]
        jawpose = d["jawpose"]
        neck = d["neck"]
        motion_1 = torch.cat((exp, jawpose, neck), dim=-1)  # [T,56]

        return (file_name, speech_1, motion_1, speech_2)


def get_metadata_noised(
    noised_root_str: str, scale: str, sr: int = 16000
) -> List[Dict]:
    noised_root = Path(noised_root_str)
    user_dir = noised_root / "user"
    agent_dir = noised_root / "agent"
    assert (
        user_dir.exists() and agent_dir.exists()
    ), f"Invalid noised_root: {noised_root}"

    cache_path = noised_root.with_suffix("")  # folder -> use a sibling cache name
    cache_pkl = Path(str(cache_path) + "_cache.pkl")

    if cache_pkl.exists():
        print(f"Load cached noised meta from {cache_pkl}")
        with open(cache_pkl, "rb") as f:
            return pickle.load(f)

    if scale == "large":
        processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self"
        )
    else:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    meta: List[Dict] = []
    user_wavs = sorted([p for p in user_dir.iterdir() if p.suffix.lower() == ".wav"])

    for uw in tqdm(user_wavs, desc="index noised set", ncols=100):
        name = uw.stem
        aw = agent_dir / f"{name}.wav"
        npz = user_dir / f"{name}.npz"
        if (not aw.exists()) or (not npz.exists()):
            continue

        # load speech (already noised) -> process to Wav2Vec input_values
        wav_u, _ = librosa.load(str(uw), sr=sr)
        wav_a, _ = librosa.load(str(aw), sr=sr)
        audio_user = np.squeeze(processor(wav_u, sampling_rate=sr).input_values)
        audio_agent = np.squeeze(processor(wav_a, sampling_rate=sr).input_values)

        # load motion
        d = np.load(str(npz))
        exp = torch.from_numpy(d["exp"]).float()  # [T,50]
        pose = d["pose"].astype(np.float32)  # [T,6]
        jawpose = torch.from_numpy(pose[:, 3:]).float()  # [T,3]
        neck = torch.from_numpy(pose[:, :3]).float()  # [T,3]

        meta.append(
            {
                "name": name,
                "audio_user": audio_user,
                "audio_agent": audio_agent,
                "exp": exp,
                "jawpose": jawpose,
                "neck": neck,
            }
        )

    if is_main_process():
        with open(cache_pkl, "wb") as f:
            pickle.dump(meta, f)

    return meta


def get_test_dataset(args):
    meta = get_metadata_noised(
        args.test_data_path, args.scale, sr=args.speech_frequency
    )
    return BSDatasetRobustNoised(meta, "test")
