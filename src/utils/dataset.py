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

        # --- Text Embedding Caching ---
        self.text_cache_path = config.dataset_path / "text_embeddings_cache.pt"
        self.text_cache: Dict[str, torch.Tensor] = {}

        if self.text_cache_path.exists():
            print(f"Loading text embedding cache from {self.text_cache_path}...")
            self.text_cache = torch.load(self.text_cache_path)

        # Collect all unique captions
        all_captions = set()
        for key, data in self.data_dict.items():
            for text_item in data["text"]:
                all_captions.add(text_item["caption"])

        # Identify missing captions
        missing_captions = [cap for cap in all_captions if cap not in self.text_cache]

        if missing_captions:
            print(
                f"Computed {len(self.text_cache)}/{len(all_captions)} embeddings. Computing {len(missing_captions)} missing..."
            )

            # LAZY IMPORT: Only import when needed
            from .text_encoder import CLIPEncoder

            # Initialize CLIP Encoder only if needed (to save VRAM if cached)
            clip_encoder = CLIPEncoder(model_name="openai/clip-vit-base-patch32")
            clip_encoder.to(config.device)

            batch_size = 32
            for i in tqdm(
                range(0, len(missing_captions), batch_size), desc="Encoding Texts"
            ):
                batch_caps = missing_captions[i : i + batch_size]
                with torch.no_grad():
                    # (B, 512)
                    embeddings = clip_encoder(batch_caps).cpu()

                for cap, emb in zip(batch_caps, embeddings):
                    self.text_cache[cap] = emb

            # Save updated cache
            print(f"Saving updated cache to {self.text_cache_path}...")
            torch.save(self.text_cache, self.text_cache_path)

            # Cleanup to free VRAM
            del clip_encoder
            torch.cuda.empty_cache()
        else:
            print("All text embeddings are cached.")

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    """
    FINAL CORRECT __getitem__ implementation
    This is the ONLY version that works - replace everything else
    """

    def __getitem__(self, item):
        """
        Returns a single sample from the dataset.
        GUARANTEES: All returned tensors have shape (max_motion_length, features)
        """
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        # Get raw data from dict
        motion = data["motion"]  # numpy array (T, 263)
        joints = data["joints"]  # numpy array (T, 22, 3)
        original_length = data["length"]  # int, original number of frames
        text_list = data["text"]  # list of text dicts

        # Choose random text
        text_data = random.choice(text_list)
        caption = text_data["caption"]

        # ===== CONVERT TO TENSORS =====
        motion = torch.from_numpy(motion.copy()).float()  # (T, 263)
        joints = torch.from_numpy(joints.copy()).float()  # (T, 22, 3)

        # ===== NORMALIZE =====
        motion = (motion - self.mean) / self.std

        # ===== DETERMINE TARGET LENGTH =====
        # The m_length modification logic from config
        m_length = original_length
        if self.config.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (
                m_length // self.config.unit_length - 1
            ) * self.config.unit_length
        else:
            m_length = (m_length // self.config.unit_length) * self.config.unit_length

        # Clamp to actual available data
        m_length = min(m_length, len(motion))
        m_length = max(1, m_length)  # At least 1 frame

        # ===== TRUNCATE TO TARGET LENGTH =====
        motion = motion[:m_length]
        joints = joints[:m_length]

        # ===== PAD OR TRUNCATE TO MAX_MOTION_LENGTH =====
        target_len = self.max_motion_length
        current_len = len(motion)

        if current_len < target_len:
            # Pad with zeros
            pad_size = target_len - current_len
            motion = torch.cat(
                [
                    motion,
                    torch.zeros(
                        pad_size,
                        motion.shape[1],
                        dtype=motion.dtype,
                        device=motion.device,
                    ),
                ],
                dim=0,
            )
            joints = torch.cat(
                [
                    joints,
                    torch.zeros(
                        pad_size,
                        joints.shape[1],
                        joints.shape[2],
                        dtype=joints.dtype,
                        device=joints.device,
                    ),
                ],
                dim=0,
            )
        elif current_len > target_len:
            # Truncate
            motion = motion[:target_len]
            joints = joints[:target_len]

        # ===== FINAL SANITY CHECK =====
        assert (
            motion.shape[0] == target_len
        ), f"Motion shape[0]={motion.shape[0]}, expected {target_len}"
        assert (
            motion.shape[1] == 263
        ), f"Motion shape[1]={motion.shape[1]}, expected 263"
        assert (
            joints.shape[0] == target_len
        ), f"Joints shape[0]={joints.shape[0]}, expected {target_len}"

        # ===== EXTRACT FEATURES =====
        history_features = get_feature_vec_subset(motion, self.feature_dims)

        # Ensure it's a tensor
        if isinstance(history_features, np.ndarray):
            history_features = torch.from_numpy(history_features).float()
        else:
            history_features = history_features.float()

        assert (
            history_features.shape[0] == target_len
        ), f"history_features shape[0]={history_features.shape[0]}, expected {target_len}"

        # ===== GET TEXT EMBEDDING =====
        text_embedding = self.text_cache[caption]
        if isinstance(text_embedding, np.ndarray):
            text_embedding = torch.from_numpy(text_embedding).float()
        else:
            text_embedding = text_embedding.float()

        return caption, history_features, motion, joints, m_length, text_embedding

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)


from typing import List, Dict, Any


def text2motion_collate_fn(
    batch: List[
        Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]
    ],
) -> Dict[str, Any]:
    """
    Collate function for Text2MotionDataset.
    Expects each sample to be a tuple:
      - caption: str
      - history_features: (max_T, C_in = 137) torch.Tensor
      - motion: (max_T, 263) torch.Tensor
      - joints: (max_T, J, 3) torch.Tensor
      - length: int
      - text_embedding: (512,) torch.Tensor
    """
    # Lists of items
    captions = [b[0] for b in batch]
    cond_feats_list = [b[1] for b in batch]  # each (T, C_in)
    motions_list = [b[2] for b in batch]
    joints_list = [b[3] for b in batch]
    lengths = [b[4] for b in batch]
    text_embs_list = [b[5] for b in batch]

    # Helper to ensure all are tensors
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            return x.float()
        else:
            return torch.tensor(x).float()

    # Convert all to tensors (redundant safety check)
    cond_feats_list = [to_tensor(x) for x in cond_feats_list]
    motions_list = [to_tensor(x) for x in motions_list]
    joints_list = [to_tensor(x) for x in joints_list]
    text_embs_list = [to_tensor(x) for x in text_embs_list]

    # Stack tensors directly
    cond_feature_batch = torch.stack(cond_feats_list, dim=0)  # (B, T, C_in)
    motion_batch = torch.stack(motions_list, dim=0)  # (B, T, 263)
    joints_batch = torch.stack(joints_list, dim=0)  # (B, T, J, 3)
    length_batch = torch.tensor(lengths, dtype=torch.long)  # (B,)
    text_emb_batch = torch.stack(text_embs_list, dim=0)  # (B, 512)

    return {
        "captions": captions,
        "history_features": cond_feature_batch,
        "motion": motion_batch,
        "joints": joints_batch,
        "lengths": length_batch,
        "text_clip": text_emb_batch,
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
        num_workers=0,  # CRITICAL: Must be 0 to avoid serialization issues
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
