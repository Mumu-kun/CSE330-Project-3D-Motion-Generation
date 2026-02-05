"""
BVH file handling utilities.

Handles conversion to/from BVH format and file I/O.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def _get_default_skeleton_hierarchy() -> Dict:
    """
    Get default skeleton hierarchy for HumanML3D (22 joints).
    Based on SMPL hierarchy flattened to 22 joints.
    """
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    joint_names = [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ]

    hierarchy = {}  # type: ignore

    for i, name in enumerate(joint_names):
        p_idx = parents[i]
        p_name = joint_names[p_idx] if p_idx != -1 else "root"

        if p_name not in hierarchy:
            hierarchy[p_name] = {"children": []}
        hierarchy[p_name]["children"].append(name)

        if name not in hierarchy:
            hierarchy[name] = {"children": []}

    return hierarchy


def joints_to_bvh(
    joint_positions: np.ndarray, fps: int = 20, skeleton_template: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Convert joint positions (nframe, 22, 3) to BVH format.

    Compatible with MoMask's BVH output format.

    Args:
        joint_positions: Joint positions (nframe, 22, 3)
        fps: Frames per second
        skeleton_template: Optional skeleton template for BVH structure

    Returns:
        BVH data dictionary with structure and motion data
    """
    nframe, num_joints, _ = joint_positions.shape

    bvh_data = {
        "hierarchy": skeleton_template or _get_default_skeleton_hierarchy(),
        "motion": {
            "frames": nframe,
            "fps": fps,
            "data": joint_positions.tolist(),
        },
    }

    print(f"TODO: Implement proper joints_to_bvh conversion")
    print(f"Input: {joint_positions.shape} -> BVH format")

    return bvh_data


def save_bvh(bvh_data: Dict[str, Any], output_path: Path) -> None:
    """
    Save BVH data to file.

    Compatible with MoMask's BVH file format.

    Args:
        bvh_data: BVH data dictionary
        output_path: Path to save BVH file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("HIERARCHY\n")
        f.write("ROOT root\n")
        f.write("{\n")
        f.write("  OFFSET 0.0 0.0 0.0\n")
        f.write(
            "  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
        )
        f.write("}\n")
        f.write("MOTION\n")
        f.write(f"Frames: {bvh_data['motion']['frames']}\n")
        f.write(f"Frame Time: {1.0 / bvh_data['motion']['fps']:.6f}\n")

    print(f"TODO: Implement complete BVH file writing")
    print(f"Saved BVH to {output_path}")


def save_joints(joint_positions: np.ndarray, output_path: Path) -> None:
    """
    Save joint positions as numpy file.

    Compatible with MoMask's output format.

    Args:
        joint_positions: Joint positions (nframe, 22, 3)
        output_path: Path to save numpy file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, joint_positions)
    print(f"Saved joints to {output_path}")


def validate_bvh(bvh_path: Path) -> bool:
    """
    Validate BVH file structure.

    Args:
        bvh_path: Path to BVH file

    Returns:
        True if valid, False otherwise
    """
    if not bvh_path.exists():
        return False

    try:
        with open(bvh_path, "r") as f:
            content = f.read()
            if "HIERARCHY" in content and "MOTION" in content:
                return True
    except Exception:
        return False

    return False
