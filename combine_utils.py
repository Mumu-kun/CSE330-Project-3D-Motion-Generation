import importlib.util
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
UTILS_DIR = SRC_DIR / "utils"
OUTPUT_SRC = SRC_DIR / "utils_kaggle.py"
OUTPUT_BUILD = PROJECT_ROOT / "build" / "utils_kaggle.py"

HEADER = """\
# =============================================================================
# utils_kaggle.py  —  single-file combined module for Kaggle / Colab
# =============================================================================

from __future__ import annotations

import os
import sys
import copy
import re
import ast
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union, cast, reveal_type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Dataset
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from transformers import CLIPTokenizer, CLIPTextModel, ACT2FN
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    ACT2FN = None

try:
    from ignite.engine import Engine, Events, State
    from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan, global_step_from_engine
    from ignite.handlers.tqdm_logger import ProgressBar
    from ignite.metrics import RunningAverage
except ImportError:
    IGNITE_AVAILABLE = False
    Engine = Events = State = None
    Checkpoint = DiskSaver = TerminateOnNan = None
    ProgressBar = None
    RunningAverage = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    UserSecretsClient = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except ImportError:
    plt = None
    FuncAnimation = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
"""

# Order: dependencies BEFORE dependents.
# models/__init__.py defines AdaLN, TemporalRoPEAttention, GatedMLP, TemporalCacheState.
# models/motion_history_encoder.py subclasses TemporalRoPEAttention.
# models/flow_matching_predictor.py uses AdaLN, GatedMLP, KinematicChainEncoder.
# models/human_motion_generator.py is the high-level generation interface.
FILE_ORDER = [
    "quaternion.py",
    "motion_utils.py",
    "config.py",
    "text_encoder.py",
    "dataset.py",
    "visualization.py",
    "wandb_logger.py",
    "models/__init__.py",
    # "models/human_motion_generator.py",
    "models/motion_history_encoder.py",
    # "models/flow_matching_predictor.py",
    "models/pretrain_trainer.py",
    # "models/finetune_trainer.py",
]

_IMPORT_RE = re.compile(r"^(\s*)(from\s+(?:utils\.[A-Za-z_.][\w.]*|\.+[\w.]*)\s+import)")


def strip_internal_imports(source):
    out = []
    lines = source.split("\n")
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = _IMPORT_RE.match(line)
        if m:
            indent = m.group(1)
            stripped = line.lstrip()
            out.append(f"{indent}# [internal import removed]  {stripped}")
            i += 1
            depth = line.count("(") - line.count(")")
            while depth > 0 and i < n:
                depth += lines[i].count("(") - lines[i].count(")")
                i += 1
        else:
            out.append(line)
            i += 1
    return "\n".join(out)


def validate_modules():
    sys.path.insert(0, str(SRC_DIR))
    for rel in FILE_ORDER:
        fp = UTILS_DIR / rel
        if not fp.exists():
            print(f"  skip (not found): {rel}")
            continue
        key = f"utils.{rel.replace(os.sep, '.').replace('.py', '')}"
        try:
            spec = importlib.util.spec_from_file_location(key, fp)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[key] = mod
            spec.loader.exec_module(mod)
            print(f"  OK: {key}")
        except Exception as e:
            print(f"  WARN: {key} -> {e}")


def _process(rel):
    src = UTILS_DIR / rel
    if not src.exists():
        print(f"  skip: {rel}")
        return ""
    raw = src.read_text(encoding="utf-8", errors="replace")
    raw = re.sub(r"^(from __future__ import annotations)\n", "", raw, flags=re.MULTILINE)
    cleaned = strip_internal_imports(raw)
    print(f"  + {rel}")
    return cleaned


def build():
    print("Phase 1: validating modules via importlib...")
    validate_modules()

    print("\nPhase 2: building combined file...")
    sections = [HEADER]

    for rel in FILE_ORDER:
        code = _process(rel)
        if not code:
            continue
        sections.append(f"\n# ========== {rel} ==========\n")
        sections.append(code)

    combined = "\n".join(sections)

    for out_path in (OUTPUT_BUILD,):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(combined, encoding="utf-8")
        print(f"  Wrote {out_path}  ({out_path.stat().st_size:,} bytes)")

    print("\nDone.")


if __name__ == "__main__":
    build()
