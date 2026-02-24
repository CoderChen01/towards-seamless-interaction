import os
import pickle
import random
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from util.misc import is_main_process


class BSDataset(data.Dataset):
    def __init__(self, data, data_type="None"):
        self.data = data
        self.len = len(self.data)
        self.data_type = data_type  # train\test\val

    def __getitem__(self, index):
        file_name = self.data[index]["name"]
        speech_1 = self.data[index]["audio1"]
        exp1 = self.data[index]["exp1"]
        jawpose1 = self.data[index]["jawpose1"]
        neck1 = self.data[index]["neck1"]
        motion_1 = torch.cat((exp1, jawpose1, neck1), dim=-1)

        speech_2 = self.data[index]["audio2"]
        exp2 = self.data[index]["exp2"]
        jawpose2 = self.data[index]["jawpose2"]
        neck2 = self.data[index]["neck2"]
        motion_2 = torch.cat((exp2, jawpose2, neck2), dim=-1)
        return (
            file_name,
            speech_1,
            motion_1,
            speech_2,
            motion_2,
        )

    def __len__(self):
        return self.len


def get_metadata(data_path, scale):
    data_path_obj = Path(data_path)

    cache_data_path = data_path_obj.with_name(
        data_path_obj.name + "_fitting"
    ).with_suffix(".pkl")
    if os.path.exists(cache_data_path):
        print(f"Load cached data from {cache_data_path}")
        with open(cache_data_path, "rb") as f:
            data = pickle.load(f)
        return data

    data = []
    if scale == "large":
        processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self"
        )
    else:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    for npz_path in tqdm(os.listdir(data_path)):
        if npz_path.endswith(".npz"):
            fileshort_name = npz_path.split("/")[-1].replace(".npz", "")
            file_meta_dict = dict()

            npz_disk_path1 = os.path.join(data_path, npz_path)
            if not os.path.exists(npz_disk_path1):
                continue
            flame_parms1 = np.load(npz_disk_path1)

            wav_path1 = os.path.join(data_path, fileshort_name + ".wav")
            if not os.path.exists(wav_path1):
                continue
            speech_array, sampling_rate = librosa.load(wav_path1, sr=16000)
            audio1 = np.squeeze(
                processor(speech_array, sampling_rate=16000).input_values
            )
            exp1 = torch.from_numpy(flame_parms1["exp"])
            jawpose1 = torch.from_numpy(flame_parms1["pose"][:, 3:])
            neck1 = torch.from_numpy(flame_parms1["pose"][:, :3])

            if fileshort_name.endswith("speaker1"):
                fileshort_name = fileshort_name.replace("speaker1", "speaker2")
            elif fileshort_name.endswith("speaker2"):
                fileshort_name = fileshort_name.replace("speaker2", "speaker1")

            npz_disk_path2 = os.path.join(data_path, fileshort_name + ".npz")
            if not os.path.exists(npz_disk_path2):
                continue
            flame_parms2 = np.load(npz_disk_path2)

            wav_path2 = os.path.join(data_path, fileshort_name + ".wav")
            if not os.path.exists(wav_path2):
                continue

            speech_array, sampling_rate = librosa.load(wav_path2, sr=16000)
            audio2 = np.squeeze(
                processor(speech_array, sampling_rate=16000).input_values
            )
            exp2 = torch.from_numpy(flame_parms2["exp"])
            jawpose2 = torch.from_numpy(flame_parms2["pose"][:, 3:])
            neck2 = torch.from_numpy(flame_parms2["pose"][:, :3])

            time_len = 200

            if exp1.shape[0] > time_len:
                for i in range(exp1.shape[0] // time_len):
                    file_meta_dict["exp1"] = exp1[i * time_len : (i + 1) * time_len]
                    file_meta_dict["jawpose1"] = jawpose1[
                        i * time_len : (i + 1) * time_len
                    ]
                    file_meta_dict["neck1"] = neck1[i * time_len : (i + 1) * time_len]

                    file_meta_dict["exp2"] = exp2[i * time_len : (i + 1) * time_len]
                    file_meta_dict["jawpose2"] = jawpose2[
                        i * time_len : (i + 1) * time_len
                    ]
                    file_meta_dict["neck2"] = neck2[i * time_len : (i + 1) * time_len]

                    file_meta_dict["name"] = fileshort_name + "_" + str(i)

                    file_meta_dict["audio1"] = audio1[
                        int(i * time_len / 25 * 16000) : int(
                            (i + 1) * time_len / 25 * 16000
                        )
                    ]
                    file_meta_dict["audio2"] = audio2[
                        int(i * time_len / 25 * 16000) : int(
                            (i + 1) * time_len / 25 * 16000
                        )
                    ]
                    data.append(file_meta_dict)
            else:
                continue

    if is_main_process():
        with open(cache_data_path, "wb") as f:
            pickle.dump(data, f)

    return data


def read_data(args):
    print("Loading data...")
    random.seed(args.seed)
    train_meta_list = []
    val_meta_list = []
    train_meta_list += get_metadata(args.train_data_path, args.scale)
    val_meta_list += get_metadata(args.val_data_path, args.scale)
    print(
        "{} sequences in train set; {} sequences in val set".format(
            len(train_meta_list), len(val_meta_list)
        )
    )
    return train_meta_list, val_meta_list


def get_fitting_datasets(args):
    train_data, val_data = read_data(args)
    train_data = BSDataset(train_data, "train")
    val_data = BSDataset(val_data, "ood")

    return train_data, val_data
