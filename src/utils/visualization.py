"""
Motion visualization utilities.

Provides 3D animation and comparison visualization for motion sequences.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from typing import Optional, Any
from utils.motion_utils import T2M_KINEMATIC_CHAIN


def probe_camera_state(ax) -> dict:
    """
    Print and return matplotlib 3D camera + scene state for use when
    configuring a Plotly backend to match this view.

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
    max_range = max(xr, yr, zr)

    state = dict(
        elev=elev,
        azim=azim,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        x_range=xr,
        y_range=yr,
        z_range=zr,
        # Plotly aspectratio dict derived from data ranges
        plotly_aspectratio=dict(
            x=xr / max_range,
            y=yr / max_range,
            z=zr / max_range,
        ),
        # Plotly camera eye derived from elev/azim
        plotly_camera_eye=dict(
            x=float(1.75 * np.cos(np.deg2rad(elev)) * np.cos(np.deg2rad(azim))),
            y=float(1.75 * np.cos(np.deg2rad(elev)) * np.sin(np.deg2rad(azim))),
            z=float(1.75 * np.sin(np.deg2rad(elev))),
        ),
    )

    print("=" * 50)
    print(f"  elev       : {elev:.2f} deg")
    print(f"  azim       : {azim:.2f} deg")
    print(f"  xlim       : [{xlim[0]:.3f}, {xlim[1]:.3f}]  range={xr:.3f}")
    print(f"  ylim       : [{ylim[0]:.3f}, {ylim[1]:.3f}]  range={yr:.3f}")
    print(f"  zlim       : [{zlim[0]:.3f}, {zlim[1]:.3f}]  range={zr:.3f}")
    print(f"  plotly aspectratio : {state['plotly_aspectratio']}")
    print(f"  plotly camera eye  : {state['plotly_camera_eye']}")
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
    import imageio
    import io
    import base64
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
    return HTML(
        f'<video controls width="600"><source src="data:video/mp4;base64,{b64}"></video>'
    )


def plot_3d_motion_plotly(
    motion: np.ndarray,
    fps: float = 20,
    radius: float = 1.0,
    title: str = "Motion Visualization",
    follow_root: bool = False,
):
    """
    Create an optimized 3D animation of motion joint positions using Plotly,
    specifically tuned for Jupyter notebooks and matching the Matplotlib probe.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Please install plotly: pip install plotly")

    n_frames = len(motion)

    pos_min = motion.min(axis=(0, 1))
    pos_max = motion.max(axis=(0, 1))

    # Set matching ranges based on matplotlib logic
    if follow_root:
        root0 = motion[0, 0, :]
        x_range = [root0[0] - radius, root0[0] + radius]
        y_range = [root0[2] - radius, root0[2] + radius]
        z_range = [pos_min[1], pos_max[1] + 0.5]
    else:
        x_range = [pos_min[0] - radius, pos_max[0] + radius]
        y_range = [pos_min[2] - radius, pos_max[2] + radius]
        z_range = [pos_min[1], pos_max[1] + 0.5]

    xr = x_range[1] - x_range[0]
    yr = y_range[1] - y_range[0]
    zr = z_range[1] - z_range[0]
    max_range = max(xr, yr, zr)

    # Exactly matching the probed aspect ratio
    aspectratio = dict(
        x=xr / max_range,
        y=yr / max_range,
        z=zr / max_range,
    )

    # Exactly matching the probed camera eye (elev=15, azim=65, dist=1.75)
    elev = 15.0
    azim = 65.0
    dist = 1.75
    camera_eye = dict(
        x=dist * np.cos(np.deg2rad(elev)) * np.cos(np.deg2rad(azim)),
        y=dist * np.cos(np.deg2rad(elev)) * np.sin(np.deg2rad(azim)),
        z=dist * np.sin(np.deg2rad(elev)),
    )

    colors = ["#2980b9", "#c0392b", "#27ae60", "#f39c12", "#8e44ad"]

    # Pre-build frames efficiently by updating only coordinate data
    initial_data = []
    for i, c_indices in enumerate(T2M_KINEMATIC_CHAIN):
        joints = motion[0, c_indices, :]
        initial_data.append(
            go.Scatter3d(
                x=joints[:, 0],
                y=joints[:, 2],
                z=joints[:, 1],
                mode="lines+markers",
                marker=dict(size=2.5, color=colors[i]),
                line=dict(width=3, color=colors[i]),
                name=f"Chain {i}",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    frames = []
    for frame_idx in range(n_frames):
        frame_data = []
        for i, c_indices in enumerate(T2M_KINEMATIC_CHAIN):
            joints = motion[frame_idx, c_indices, :]
            frame_data.append(
                go.Scatter3d(x=joints[:, 0], y=joints[:, 2], z=joints[:, 1])
            )

        layout_update = {}
        if follow_root:
            root = motion[frame_idx, 0, :]
            layout_update = dict(
                scene=dict(
                    xaxis=dict(range=[root[0] - radius, root[0] + radius]),
                    yaxis=dict(range=[root[2] - radius, root[2] + radius]),
                )
            )

        frames.append(
            go.Frame(data=frame_data, name=str(frame_idx), layout=layout_update)
        )

    fig = go.Figure(data=initial_data, frames=frames)

    fig.update_layout(
        title=title,
        width=800,
        height=800,
        scene=dict(
            xaxis=dict(
                title="X (Side)",
                range=x_range,
                autorange=False,
                showbackground=True,
                backgroundcolor="white",
                gridcolor="lightgray",
                zerolinecolor="gray",
            ),
            yaxis=dict(
                title="Z (Forward)",
                range=y_range,
                autorange=False,
                showbackground=True,
                backgroundcolor="white",
                gridcolor="lightgray",
                zerolinecolor="gray",
            ),
            zaxis=dict(
                title="Y (Height)",
                range=z_range,
                autorange=False,
                showbackground=True,
                backgroundcolor="white",
                gridcolor="lightgray",
                zerolinecolor="gray",
            ),
            aspectmode="manual",
            aspectratio=aspectratio,
            camera=dict(
                eye=camera_eye,
                up=dict(x=0, y=0, z=1),
                projection=dict(type="orthographic"),
            ),
        ),
        # in a scope where n_frames and fps are defined
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                direction="left",
                x=0.0,
                y=0,
                xanchor="left",
                yanchor="top",
                buttons=[
                    # Play / Pause toggle
                    dict(
                        label="▶/⏸",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=1000 / fps, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0, easing="linear"),
                            ),
                        ],
                    ),
                    # Restart from beginning
                    dict(
                        label="⟲",
                        method="animate",
                        args=[
                            [str(0)],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=12),
                    prefix="Frame: ",
                    visible=True,
                    xanchor="right",
                ),
                transition=dict(duration=0, easing="linear"),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[
                            [str(k)],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                        label=str(k),
                        method="animate",
                    )
                    for k in range(n_frames)
                ],
            )
        ],
        margin=dict(l=0, r=20, t=40, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


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
        probe: If True, print camera + scene state (for Plotly parity work)
        backend: Visualization backend - "plotly" (default) or "matplotlib"
    """
    fps = fps / skip_frames
    motion_subsampled = joint_positions[::skip_frames]

    if backend == "plotly":
        try:
            fig = plot_3d_motion_plotly(
                motion_subsampled,
                fps=fps,
                radius=radius,
                title=title,
            )

            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                html_path = save_path.with_suffix(".html")
                fig.write_html(str(html_path))
                print(f"Saved interactive animation to {html_path}")

            if notebook:
                return fig
            return fig

        except ImportError:
            print("Plotly not available, falling back to matplotlib...")
            backend = "matplotlib"

    if backend == "matplotlib":
        html = plot_3d_motion(
            motion_subsampled, radius=radius, fps=fps, title=title, probe=probe
        )

        # if save_path:
        #     # imageio needs a real file for saving; re-render to disk
        #     save_path.parent.mkdir(parents=True, exist_ok=True)
        #     import imageio
        #     # render frames again to save_path directly
        #     # simplest: call a thin wrapper that writes to file
        #     _plot_3d_motion_to_file(
        #         motion_subsampled, save_path, radius=radius, fps=fps, title=title
        #     )
        #     print(f"Saved animation to {save_path}")

        return html


def compare_motions(
    generated_joints: np.ndarray,
    ground_truth_joints: np.ndarray,
    save_path: Optional[Path] = None,
    backend: str = "plotly",
) -> None:
    """
    Compare generated motion with ground truth.
    """
    # Simply call the visualization logic with the chosen backend
    visualize_motion(
        generated_joints,
        title="Generated vs Ground Truth",
        save_path=save_path,
        backend=backend,
    )
