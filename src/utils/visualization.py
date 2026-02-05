"""
Motion visualization utilities.

Provides 3D animation and comparison visualization for motion sequences.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from typing import Optional, Any
from .motion_utils import t2m_kinematic_chain


def plot_3d_motion(
    motion: np.ndarray,
    fps: float = 20,
    radius: float = 1.0,
    title: str = "Motion Visualization",
    follow_root: bool = False,
) -> FuncAnimation:
    """
    Create a 3D animation of motion joint positions.

    Args:
        motion: Joint positions (nframe, 22, 3)
        fps: Frames per second
        radius: Radius of the viewing box around the character
        title: Plot title
        follow_root: Whether the camera should follow the root joint

    Returns:
        Matplotlib FuncAnimation object
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=15, azim=-70)

    colors = ["#2980b9", "#c0392b", "#27ae60", "#f39c12", "#8e44ad"]
    lines = [
        ax.plot([], [], [], color=colors[i % len(colors)], marker="o", ms=2, lw=2)[0]
        for i in range(len(t2m_kinematic_chain))
    ]

    ax.set_xlabel("X (Side)")
    ax.set_ylabel("Z (Forward)")
    ax.set_zlabel("Y (Height)")
    ax.set_title(title)

    pos_min = motion.min(axis=(0, 1))
    pos_max = motion.max(axis=(0, 1))

    def update(frame):
        root = motion[frame, 0, :]

        if follow_root:
            ax.set_xlim3d([root[0] - radius, root[0] + radius])
            ax.set_ylim3d([root[2] - radius, root[2] + radius])
            ax.set_zlim3d([pos_min[1], pos_max[1] + radius * 0.5])
        else:
            ax.set_xlim3d([pos_min[0] - radius, pos_max[0] + radius])
            ax.set_ylim3d([pos_min[2] - radius, pos_max[2] + radius])
            ax.set_zlim3d([pos_min[1], pos_max[1] + radius * 0.5])

        for i, c_indices in enumerate(t2m_kinematic_chain):
            joints = motion[frame, c_indices, :]
            lines[i].set_data(joints[:, 0], joints[:, 2])
            lines[i].set_3d_properties(joints[:, 1])
        return lines

    ani = FuncAnimation(
        fig, update, frames=len(motion), interval=1000 / fps, blit=False
    )
    plt.close()
    return ani


def visualize_motion(
    joint_positions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    title: str = "Motion Visualization",
    save_path: Optional[Path] = None,
    fps: float = 20,
    skip_frames: int = 1,
    notebook: bool = True,
) -> Any:
    """
    Visualize motion from joint positions.

    Args:
        joint_positions: Joint positions (nframe, 22, 3)
        ground_truth: Optional ground truth for comparison
        title: Plot title
        save_path: Optional path to save visualization
        fps: Frames per second
        notebook: Whether to return HTML for notebook display
    """
    fps = fps / skip_frames
    ani = plot_3d_motion(joint_positions[::skip_frames], fps=fps, title=title)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ani.save(str(save_path), writer="ffmpeg", fps=int(fps))
        print(f"Saved animation to {save_path}")

    if notebook:
        from IPython.display import HTML

        return HTML(ani.to_html5_video())

    return ani


def compare_motions(
    generated_joints: np.ndarray,
    ground_truth_joints: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """
    Compare generated motion with ground truth.

    Args:
        generated_joints: Generated joint positions (nframe, 22, 3)
        ground_truth_joints: Ground truth joint positions (nframe, 22, 3)
        save_path: Optional path to save comparison
    """
    visualize_motion(
        generated_joints,
        ground_truth=ground_truth_joints,
        title="Generated vs Ground Truth",
        save_path=save_path,
    )
