"""
Dataset loading and Text2Motion dataset implementation.

Handles loading HumanML3D dataset with text-motion pairs.
"""

import torch
import numpy as np
from os.path import join as pjoin
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from config import Config
from .motion_utils import get_feature_vec_subset


class Text2MotionDataset(Dataset):
    """
    Text-to-Motion Dataset compatible with HumanML3D format.

    Loads motion features with corresponding text descriptions.
    Supports variable-length sequences with time-stamped text annotations.
    """

    def __init__(
        self,
        config: Config,
        mean: np.ndarray,
        std: np.ndarray,
        split: str = "train",
        feature_dims: tuple[slice, ...] | None = None,
    ):
        self.config = config
        self.feature_dims = (
            feature_dims if feature_dims is not None else config.feature_dims
        )
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = config.max_motion_length
        min_motion_len = 40 if config.dataset_name == "t2m" else 24

        # Derive paths from config.dataset_path
        motion_dir = config.dataset_path / "new_joint_vecs"
        joints_dir = config.dataset_path / "new_joints"
        text_dir = config.dataset_path / "texts"
        split_file = config.dataset_path / f"{split}.txt"

        data_dict = {}
        id_list = []
        with open(str(split_file), "r", encoding="utf-8") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(str(motion_dir), name + ".npy"))
                joints = np.load(pjoin(str(joints_dir), name + ".npy"))

                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue

                text_data = []
                flag = False
                with open(
                    pjoin(str(text_dir), name + ".txt"), "r", encoding="utf-8"
                ) as f:
                    for line in f.readlines():
                        text_dict: Dict[str, Optional[Any]] = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20) : int(to_tag * 20)]
                                if (len(n_motion)) < min_motion_len or (
                                    len(n_motion) >= 200
                                ):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                    + "_"
                                    + name
                                )
                                while new_name in data_dict:
                                    new_name = (
                                        random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                        + "_"
                                        + name
                                    )
                                n_joints = joints[int(f_tag * 20) : int(to_tag * 20)]
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "joints": n_joints,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "joints": joints,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                pass

        name_list, length_list = new_name_list, length_list

        self.mean = torch.from_numpy(mean).float()
        self.std = torch.from_numpy(std).float()
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, joints, m_length, text_list = (
            data["motion"],
            data["joints"],
            data["length"],
            data["text"],
        )
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if self.config.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (
                m_length // self.config.unit_length - 1
            ) * self.config.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.config.unit_length) * self.config.unit_length
        # Convert to tensors
        motion = torch.from_numpy(motion).float()
        joints = torch.from_numpy(joints).float()

        # Normalize features
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            padding_len = self.max_motion_length - m_length

            # Pad motion features
            motion_padding = torch.zeros(
                (padding_len, motion.shape[1]), dtype=motion.dtype
            )
            motion = torch.cat([motion, motion_padding], dim=0)

            # Pad joint positions
            joints_padding = torch.zeros(
                (padding_len, joints.shape[1], joints.shape[2]), dtype=joints.dtype
            )
            joints = torch.cat([joints, joints_padding], dim=0)

        # Extract feature subset if specified
        # motion: (max_seq_len, 263)
        history_features = get_feature_vec_subset(motion, self.feature_dims)

        return caption, history_features, motion, joints, m_length

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)


from typing import List, Dict, Any


def text2motion_collate_fn(
    batch: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, int]],
) -> Dict[str, Any]:
    """
    Collate function for Text2MotionDataset.
    Expects each sample to be a tuple:
      - caption: str
      - history_features: (max_T, C_in = 137) torch.Tensor
      - motion: (max_T, 263) torch.Tensor
      - joints: (max_T, J, 3) torch.Tensor
      - length: int
    """
    # Lists of items
    captions = [b[0] for b in batch]
    cond_feats_list = [b[1] for b in batch]  # each (T, C_in)
    motions_list = [b[2] for b in batch]
    joints_list = [b[3] for b in batch]
    lengths = [b[4] for b in batch]

    # Stack tensors directly
    cond_feature_batch = torch.stack(cond_feats_list, dim=0)  # (B, T, C_in)
    motion_batch = torch.stack(motions_list, dim=0)  # (B, T, 263)
    joints_batch = torch.stack(joints_list, dim=0)  # (B, T, J, 3)
    length_batch = torch.tensor(lengths, dtype=torch.long)  # (B,)

    return {
        "captions": captions,
        "history_features": cond_feature_batch,
        "motion": motion_batch,
        "joints": joints_batch,
        "lengths": length_batch,
    }


def create_dataloader(
    config: Config,
    split: str = "train",
    shuffle: bool = True,
) -> DataLoader:
    """
    Create DataLoader for Text2MotionDataset.

    Automatically loads mean and std from Mean.npy and Std.npy in dataset_path.

    Args:
        config: Config object with dataset configuration
        split: Dataset split ("train", "val", "test"). Default: "train"
        shuffle: Whether to shuffle data

    Returns:
        DataLoader instance
    """
    mean_path = config.dataset_path / "Mean.npy"
    std_path = config.dataset_path / "Std.npy"

    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(
            f"Mean.npy and/or Std.npy not found in {config.dataset_path}. "
            "Please ensure Mean.npy and Std.npy exist in the dataset directory."
        )

    mean = np.load(mean_path)
    std = np.load(std_path)

    dataset_obj = Text2MotionDataset(
        config, mean, std, split, feature_dims=config.feature_dims
    )
    return DataLoader(
        dataset_obj,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=text2motion_collate_fn,
    )


def load_sample(dataset_path: Path, file_id: str) -> Dict[str, Optional[Any]]:
    """
    Load a single motion sample with features, joints, and text.

    Args:
        dataset_path: Path to HumanML3D dataset root
        file_id: Motion sample ID (without extension)

    Returns:
        Dictionary with keys:
        - 'features': Feature vectors (nframe, 263) from new_joint_vecs
        - 'joints': Joint positions (nframe, 22, 3) from new_joints
        - 'text': Text description from texts folder
        - 'file_id': Sample ID
    """
    features_path = dataset_path / "new_joint_vecs" / f"{file_id}.npy"
    joints_path = dataset_path / "new_joints" / f"{file_id}.npy"
    text_path = dataset_path / "texts" / f"{file_id}.txt"

    data: Dict[str, Optional[Any]] = {"file_id": file_id}

    # Load feature vectors
    if features_path.exists():
        data["features"] = np.load(features_path)
    else:
        print(f"Warning: Features not found for {file_id}")
        data["features"] = None

    # Load joint positions
    if joints_path.exists():
        data["joints"] = np.load(joints_path)
    else:
        print(f"Warning: Joints not found for {file_id}")
        data["joints"] = None

    # Load text description
    if text_path.exists():
        with open(text_path, "r") as f:
            descriptions = [line.strip().split("#")[0] for line in f.readlines()]
            data["text"] = descriptions[0] if descriptions else ""
    else:
        data["text"] = ""

    return data
