#!/usr/bin/env python3
"""Symlink / wrapper to project-root train_phase3.py."""
import os
import sys
from pathlib import Path

target = Path(__file__).resolve().parent.parent / "train_phase3.py"
if target.exists():
    os.execv(sys.executable, [sys.executable, str(target)] + sys.argv[1:])
else:
    raise FileNotFoundError(f"Cannot find {target}")
