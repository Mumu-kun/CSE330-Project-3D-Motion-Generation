"""
Motion visualization utilities.

Provides 3D animation and comparison visualization for motion sequences.
"""

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from utils.motion_utils import T2M_KINEMATIC_CHAIN


def probe_camera_state(ax) -> dict:
    """
    Print and return matplotlib 3D camera + scene state.

    Call this interactively after plot_3d_motion returns to capture the
    current view angle and grid sizing.

    Returns:
        dict with elev, azim, xlim, ylim, zlim, and x/y/z ranges
    """
    elev = ax.elev
    azim = ax.azim
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    xr = xlim[1] - xlim[0]
    yr = ylim[1] - ylim[0]
    zr = zlim[1] - zlim[0]

    state = dict(
        elev=elev,
        azim=azim,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        x_range=xr,
        y_range=yr,
        z_range=zr,
    )

    print("=" * 50)
    print(f"  elev       : {elev:.2f} deg")
    print(f"  azim       : {azim:.2f} deg")
    print(f"  xlim       : [{xlim[0]:.3f}, {xlim[1]:.3f}]  range={xr:.3f}")
    print(f"  ylim       : [{ylim[0]:.3f}, {ylim[1]:.3f}]  range={yr:.3f}")
    print(f"  zlim       : [{zlim[0]:.3f}, {zlim[1]:.3f}]  range={zr:.3f}")
    print("=" * 50)
    return state


def plot_3d_motion(
    motion: np.ndarray,
    fps: float = 20,
    radius: float = 1.0,
    title: str = "Motion Visualization",
    follow_root: bool = False,
    probe: bool = False,
    save_path: Optional[Path] = None,
):
    import base64
    import io

    import imageio
    from IPython.display import HTML

    colors = ["#2980b9", "#c0392b", "#27ae60", "#f39c12", "#8e44ad"]
    pos_min = motion.min(axis=(0, 1))
    pos_max = motion.max(axis=(0, 1))

    x_range = [pos_min[0] - radius, pos_max[0] + radius]
    y_range = [pos_min[2] - radius, pos_max[2] + radius]
    z_range = [pos_min[1], pos_max[1] + 0.5]

    # create figure ONCE and reuse
    fig = plt.figure(figsize=(6, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")
    ax.grid(False)
    ax.view_init(elev=15, azim=65)
    ax.set_xlim3d(x_range)
    ax.set_ylim3d(y_range)
    ax.set_zlim3d(z_range)
    ax.set_xlabel("X (Side)")
    ax.set_ylabel("Z (Forward)")
    ax.set_zlabel("Y (Height)")
    ax.set_title(title)

    if probe:
        print(f"\n[probe] Matplotlib camera + scene state for '{title}':")
        probe_camera_state(ax)

    # create line artists ONCE
    lines = [
        ax.plot([], [], [], color=colors[i % len(colors)], marker="o", ms=2, lw=2)[0]
        for i in range(len(T2M_KINEMATIC_CHAIN))
    ]

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    target = str(save_path) if save_path else io.BytesIO()

    writer_kwargs = {
        "fps": fps,
        "codec": "libx264",
        "output_params": ["-preset", "ultrafast", "-crf", "28"],
    }
    if save_path:
        writer = imageio.get_writer(target, format="FFMPEG", **writer_kwargs)
    else:
        writer = imageio.get_writer(target, format="mp4", **writer_kwargs)

    for frame_idx in range(len(motion)):
        if follow_root:
            root = motion[frame_idx, 0, :]
            ax.set_xlim3d([root[0] - radius, root[0] + radius])
            ax.set_ylim3d([root[2] - radius, root[2] + radius])

        # update artist data only — no new objects created
        for i, c_indices in enumerate(T2M_KINEMATIC_CHAIN):
            joints = motion[frame_idx, c_indices, :]
            lines[i].set_data(joints[:, 0], joints[:, 2])
            lines[i].set_3d_properties(joints[:, 1])

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(img)

    writer.close()
    plt.close(fig)

    if save_path:
        print(f"Saved animation to {save_path}")
        return save_path

    target.seek(0)
    b64 = base64.b64encode(target.read()).decode()
    return HTML(f'<video controls width="600"><source src="data:video/mp4;base64,{b64}"></video>')


def visualize_motion(
    joint_positions: np.ndarray,
    title: str = "Motion Visualization",
    save_path: Optional[Path] = None,
    fps: float = 20,
    skip_frames: int = 1,
    radius: float = 1,
    notebook: bool = True,
    probe: bool = False,
    backend: str = "matplotlib",
) -> Any:
    """
    Visualize motion from joint positions.

    Args:
        joint_positions: Joint positions (nframe, 22, 3)
        title: Plot title
        save_path: Optional path to save visualization
        fps: Frames per second
        skip_frames: Skip every N frames (reduces rendering time)
        radius: Radius of the viewing box
        notebook: Whether to return visualization for notebook display
        probe: If True, print camera + scene state
        backend: Visualization backend - only "matplotlib" is supported
    """
    if backend != "matplotlib":
        print(f"Backend '{backend}' is not supported. Using matplotlib.")
    fps = fps / skip_frames
    motion_subsampled = joint_positions[::skip_frames]
    html = plot_3d_motion(motion_subsampled, radius=radius, fps=fps, title=title, probe=probe)
    return html


def compare_motions(
    generated_joints: np.ndarray,
    ground_truth_joints: np.ndarray,
    save_path: Optional[Path] = None,
    backend: str = "matplotlib",
) -> None:
    """
    Compare generated motion with ground truth.
    """
    visualize_motion(
        generated_joints,
        title="Generated vs Ground Truth",
        save_path=save_path,
        backend=backend,
    )
