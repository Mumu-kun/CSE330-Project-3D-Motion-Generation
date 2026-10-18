#!/usr/bin/env python3
"""Safely clean up all checkpoint files except the 4 specified checkpoints:
1. pretrain_best_val_20261017_005458_425.pt (Encoder in src/phase3.ipynb)
2. finetune_best_val_20261017_011538_200.pt (Decoder in src/phase3.ipynb)
3. phase3_predictor_best_val_20261017_040556.pt (Predictor in src/phase3.ipynb)
4. phase3_predictor_best_val_20261018_220823.pt (Explicitly requested to exclude)
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# The exact 4 checkpoints to preserve
EXCLUDE_CHECKPOINTS = {
    # Encoder
    "pretrain_best_val_20261017_005458_425.pt",
    # Decoder
    "finetune_best_val_20261017_011538_200.pt",
    # Predictor (notebook)
    "phase3_predictor_best_val_20261017_040556.pt",
    # Predictor (active run best val - explicitly excluded by user)
    "phase3_predictor_best_val_20261018_220823.pt",
}

def find_all_checkpoints():
    checkpoint_dirs = [
        BASE_DIR / "checkpoints",
        BASE_DIR / "src" / "checkpoints",
    ]
    seen_files = set()
    found = []
    for cdir in checkpoint_dirs:
        if not cdir.exists():
            continue
        for p in cdir.rglob("*.pt"):
            real_path = p.resolve()
            if real_path not in seen_files:
                seen_files.add(real_path)
                found.append(real_path)
    return found

def main():
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    execute = "--execute" in sys.argv or "--yes" in sys.argv or "-y" in sys.argv

    if not dry_run and not execute:
        print("=" * 70)
        print("CHECKPOINT CLEANUP SCRIPT")
        print("=" * 70)
        print("Usage:")
        print("  python cleanup_checkpoints.py --dry-run   # Preview files to delete")
        print("  python cleanup_checkpoints.py --execute   # Permanently delete candidate checkpoints")
        print("=" * 70)
        sys.exit(0)

    all_ckpts = find_all_checkpoints()
    to_delete = []
    to_keep = []

    for real_path in all_ckpts:
        fname = real_path.name
        if fname in EXCLUDE_CHECKPOINTS:
            to_keep.append(real_path)
        else:
            to_delete.append(real_path)

    print("=" * 70)
    print(f"CHECKPOINTS TO KEEP ({len(to_keep)} files):")
    print("=" * 70)
    for p in sorted(to_keep, key=lambda x: x.name):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  [KEEP]   {p.name:<55} ({size_mb:>6.1f} MB)")

    print("\n" + "=" * 70)
    print(f"CHECKPOINTS TO DELETE ({len(to_delete)} files):")
    print("=" * 70)
    total_bytes = 0
    for p in sorted(to_delete, key=lambda x: x.name):
        size = p.stat().st_size
        total_bytes += size
        print(f"  [DELETE] {p.name:<55} ({size / (1024 * 1024):>6.1f} MB)")

    freed_gb = total_bytes / (1024 * 1024 * 1024)
    print("-" * 70)
    print(f"Total space to reclaim: {freed_gb:.2f} GB across {len(to_delete)} files")
    print("=" * 70)

    if execute:
        print("\nDeleting files now...")
        deleted_count = 0
        failed_count = 0
        for p in to_delete:
            try:
                p.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {p}: {e}")
                failed_count += 1
        print(f"\nCompleted! Deleted {deleted_count} files ({freed_gb:.2f} GB reclaimed).")
        if failed_count > 0:
            print(f"Failed to delete {failed_count} files.")
    else:
        print("\n[Dry-run mode] No files were deleted. Pass --execute to delete them.")

if __name__ == "__main__":
    main()
