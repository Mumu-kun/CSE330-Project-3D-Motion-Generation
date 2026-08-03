"""
Dataset loading and Text2Motion dataset implementation.

Handles loading HumanML3D dataset with text-motion pairs.
"""

import random
from os.path import join as pjoin
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.config import Config
from utils.motion_utils import FeatureNormalizer


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
    ):
        self.config = config
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = config.max_motion_length
        self.current_horizon = config.max_motion_length
        min_motion_len = 40

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
                with open(pjoin(str(text_dir), name + ".txt"), "r", encoding="utf-8") as f:
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
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                while new_name in data_dict:
                                    new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
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
            except Exception:
                pass

        name_length_pairs = list(zip(new_name_list, length_list))
        name_length_pairs.sort(key=lambda x: x[1])  # Sort by length

        self.name_list = [pair[0] for pair in name_length_pairs]
        self.length_arr = np.array([pair[1] for pair in name_length_pairs])
        self.data_dict = data_dict
        self.mean = torch.from_numpy(mean).float()
        self.std = torch.from_numpy(std).float()

        # --- Text Embedding Caching ---
        self.text_cache_path = config.dataset_path / "text_embeddings_cache.pt"
        self.text_cache: Dict[str, torch.Tensor] = {}

        if self.text_cache_path.exists():
            print(f"Loading text embedding cache from {self.text_cache_path}...")
            self.text_cache = torch.load(self.text_cache_path, weights_only=False)

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
            from utils.text_encoder import CLIPEncoder

            # Initialize CLIP Encoder only if needed (to save VRAM if cached)
            clip_encoder = CLIPEncoder(model_name="openai/clip-vit-base-patch32")
            clip_encoder.to(config.device)

            batch_size = 32
            for i in tqdm(range(0, len(missing_captions), batch_size), desc="Encoding Texts"):
                batch_caps = missing_captions[i : i + batch_size]
                with torch.no_grad():
                    # (B, 1, 512) - pooled CLIP embeddings
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

    def get_normalizer(self) -> FeatureNormalizer:
        """Get a FeatureNormalizer instance for normalizing/denormalizing features."""
        return FeatureNormalizer(mean=self.mean.clone(), std=self.std.clone())

    def __len__(self):
        return len(self.data_dict) - self.pointer

    """
    FINAL CORRECT __getitem__ implementation
    This is the ONLY version that works - replace everything else
    """

    def __getitem__(self, item) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor, str, list[str]]:
        """
        Returns a single sample from the dataset.
        GUARANTEES: All returned tensors have shape (max_motion_length, features)

        NOTE: Returns RAW (unnormalized) features. Normalization should be done
        externally using FeatureNormalizer before passing to models.

        history_motion: (history_length, 271) RAW features from the frames
        immediately preceding the target window.  Zero-padded at the front when
        fewer than history_length frames are available before the target start.
        Used exclusively by Phase 3 (FlowMatchingPredictor) as the encoder
        conditioning context; ignored by pretrain and decoder trainers.
        """
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        # Get raw data from dict
        motion = data["motion"]  # numpy array (T, 271)
        joints = data["joints"]  # numpy array (T, 22, 3)
        original_length = data["length"]  # int, original number of frames
        text_list = data["text"]  # list of text dicts

        # Choose random text
        text_data = random.choice(text_list)
        caption = text_data["caption"]
        tokens: list[str] = text_data["tokens"]

        # ===== CONVERT TO TENSORS =====
        motion = torch.from_numpy(motion.copy()).float()  # (T, 271) - RAW features
        joints = torch.from_numpy(joints.copy()).float()  # (T, 22, 3)

        # ===== NO NORMALIZATION HERE =====
        # Normalization is handled externally via FeatureNormalizer
        # motion = (motion - self.mean) / self.std  # REMOVED

        # ===== PAD OR TRUNCATE TO MAX_MOTION_LENGTH =====
        target_len = self.current_horizon
        current_len = original_length
        history_length = self.config.history_length

        if current_len < target_len:
            # ── Short sequence: pad target, no history available ──────────────
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
            # No frames precede the target window — history is all zeros
            history_motion = torch.zeros(history_length, motion.shape[1], dtype=motion.dtype)
            history_valid_length = 0

        elif current_len > target_len:
            start_idx = random.randint(0, current_len - self.current_horizon)

            # ── History: frames strictly before start_idx ─────────────────────
            hist_end   = start_idx
            hist_start = max(0, start_idx - history_length)
            hist_slice = motion[hist_start:hist_end]            # (≤ history_length, 271)
            history_valid_length = hist_slice.shape[0]

            if hist_slice.shape[0] < history_length:
                # Zero-pad at the front so history is always (history_length, 271)
                pad_h = torch.zeros(
                    history_length - hist_slice.shape[0],
                    motion.shape[1],
                    dtype=motion.dtype,
                )
                history_motion = torch.cat([pad_h, hist_slice], dim=0)
            else:
                history_motion = hist_slice

            # ── Target window ─────────────────────────────────────────────────
            motion = motion[start_idx : start_idx + self.current_horizon]
            joints = joints[start_idx : start_idx + self.current_horizon]

        else:
            # current_len == target_len: no room for history
            history_motion = torch.zeros(history_length, motion.shape[1], dtype=motion.dtype)
            history_valid_length = 0

        valid_length = min(current_len, target_len)

        # ===== GET TEXT EMBEDDING =====
        text_embedding = self.text_cache[caption]
        if isinstance(text_embedding, np.ndarray):
            text_embedding = torch.from_numpy(text_embedding).float()
        else:
            text_embedding = text_embedding.float()

        # Normalize to strict shape: (1, 512)
        if text_embedding.ndim == 1:
            if text_embedding.shape[0] != CLIP_EMBED_DIM:
                raise ValueError(
                    f"Invalid 1D text embedding shape {tuple(text_embedding.shape)} for caption '{caption}'. "
                    f"Expected ({CLIP_EMBED_DIM},)."
                )
            text_embedding = text_embedding.unsqueeze(0)
        elif text_embedding.ndim == 2:
            if text_embedding.shape == (1, CLIP_EMBED_DIM):
                pass
            elif text_embedding.shape == (CLIP_MAX_SEQ_LEN, CLIP_EMBED_DIM):
                raise ValueError(
                    "Detected legacy CLIP sequence embedding shape (77, 512) in text cache. "
                    "Regenerate text_embeddings_cache.pt using pooled CLIP outputs (1, 512)."
                )
            else:
                raise ValueError(
                    f"Invalid 2D text embedding shape {tuple(text_embedding.shape)} for caption '{caption}'. "
                    f"Expected (1, {CLIP_EMBED_DIM})."
                )
        else:
            raise ValueError(
                f"Invalid text embedding rank {text_embedding.ndim} for caption '{caption}'. "
                "Expected rank 2 with shape (1, 512)."
            )

        # text_embedding shape: (1, 512) - pooled CLIP embedding
        # motion shape:         (target_len, 271)   - target window; used for z1 (decoder + predictor target)
        # history_motion shape: (history_length, 271) - frames strictly preceding motion; used for track_features
        # valid_length is the number of real frames before zero-padding, capped at target_len.
        sample_id = self.name_list[idx]

        return caption, motion, joints, history_motion, valid_length, text_embedding, sample_id, tokens, history_valid_length

    def reset_min_len(self, length: int | None = None):
        if length is None:
            self.pointer = 0
            return
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        # print("Pointer Pointing at %d" % self.pointer)

    def set_horizon(self, horizon: int | None = None):
        """
        Set the horizon for the dataset.
        If horizon is None, use the default horizon.
        If horizon is an integer, set the horizon to that value.
        Dynamic horizon update with filtering.
        """
        if horizon is None:
            self.current_horizon = self.max_motion_length
            self.reset_min_len()  # Reset pointer to start
            return

        assert 1 <= horizon <= self.max_motion_length
        self.current_horizon = horizon
        self.reset_min_len(horizon)  # Auto-filter


# CLIP constants
CLIP_MAX_SEQ_LEN = 77
CLIP_EMBED_DIM = 512


def text2motion_collate_fn(
    batch: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor, str, list[str], int]],
) -> Dict[str, Any]:
    """
    Collate function for Text2MotionDataset.
    Expects each sample to be a tuple:
      - caption:        str
      - motion:         (target_len, 271) torch.Tensor - target window, RAW 271D features
      - joints:         (target_len, J, 3) torch.Tensor
      - history_motion: (history_length, 271) torch.Tensor - frames preceding motion,
                        zero-padded at front when not enough history; RAW features.
                        Used only by FlowMatchingTrainer (Phase 3).
      - length:         int - valid frames in motion before padding, capped at target_len
      - text_embedding: (1, 512) torch.Tensor - pooled CLIP embeddings
      - sample_id:      str
      - tokens:         list[str]
      - history_valid_length: int - exact count of real non-zero history frames
    """
    # Lists of items
    captions         = [b[0] for b in batch]
    motions_list     = [b[1] for b in batch]
    joints_list      = [b[2] for b in batch]
    history_list     = [b[3] for b in batch]
    lengths          = [b[4] for b in batch]
    text_embs_list   = [b[5] for b in batch]
    sample_ids       = [b[6] for b in batch]
    tokens_list      = [b[7] for b in batch]
    hist_lengths     = [b[8] for b in batch]

    # Stack tensors directly
    motion_batch       = torch.stack(motions_list,   dim=0)  # (B, T, 271)
    joints_batch       = torch.stack(joints_list,    dim=0)  # (B, T, J, 3)
    history_batch      = torch.stack(history_list,   dim=0)  # (B, history_length, 271)
    length_batch       = torch.tensor(lengths, dtype=torch.long)  # (B,)
    text_emb_batch     = torch.stack(text_embs_list, dim=0)  # (B, 1, 512)
    hist_length_batch  = torch.tensor(hist_lengths, dtype=torch.long)  # (B,)

    return {
        "captions":             captions,
        "sample_ids":           sample_ids,
        "motion":               motion_batch,        # target window
        "history_motion":       history_batch,      # context window preceding target
        "joints":               joints_batch,
        "lengths":              length_batch,
        "text_clip":            text_emb_batch,
        "tokens":               tokens_list,
        "history_valid_length": hist_length_batch,
    }


def create_dataloader(
    config: Config,
    split: str = "train",
    shuffle: bool = True,
) -> Tuple[DataLoader, FeatureNormalizer]:
    """
    Create DataLoader for Text2MotionDataset.

    Automatically loads mean and std from Mean.npy and Std.npy in dataset_path.

    Args:
        config: Config object with dataset configuration
        split: Dataset split ("train", "val", "test"). Default: "train"
        shuffle: Whether to shuffle data

    Returns:
        Tuple of (DataLoader instance, FeatureNormalizer instance)
        - DataLoader provides RAW (unnormalized) features
        - FeatureNormalizer should be used to normalize features before passing to models
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

    dataset_obj = Text2MotionDataset(config, mean, std, split)

    # Create FeatureNormalizer for external normalization
    normalizer = FeatureNormalizer(
        mean=torch.from_numpy(mean).float(),
        std=torch.from_numpy(std).float(),
    )

    dataloader = DataLoader(
        dataset_obj,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=text2motion_collate_fn,
    )

    return dataloader, normalizer


def load_sample(dataset_path: Path, file_id: str) -> Dict[str, Optional[Any]]:
    """
    Load a single motion sample with features, joints, and text.

    Args:
        dataset_path: Path to HumanML3D dataset root
        file_id: Motion sample ID (without extension)

    Returns:
        Dictionary with keys:
        - 'features': Feature vectors (nframe, 271) from new_joint_vecs
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
