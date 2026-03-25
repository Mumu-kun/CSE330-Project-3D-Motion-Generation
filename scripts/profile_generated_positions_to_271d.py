"""Profile generated_positions_to_271d and report internal hotspots.

Usage example:
    python -u scripts/profile_generated_positions_to_271d.py --device cpu --batch-size 2 --iterations 300 --warmup 50
"""

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from typing import Dict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.motion_utils import (  # noqa: E402
    _compute_ik,
    _forward_kinematics,
    generated_positions_to_271d,
    get_dataset_config,
    qrot,
    quaternion_to_cont6d,
)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _new_timer() -> Dict[str, float]:
    return {
        "root_features": 0.0,
        "ik_and_rot6d": 0.0,
        "ric_transform": 0.0,
        "local_vel_and_contacts": 0.0,
        "final_assembly": 0.0,
        "total_reconstructed": 0.0,
    }


def _profile_reconstructed_path(
    new_positions: torch.Tensor,
    prev_positions: torch.Tensor,
    iterations: int,
    warmup: int,
    feet_thre: float,
    use_fk_for_ric: bool,
) -> Dict[str, float]:
    cfg = get_dataset_config("t2m")
    raw_offsets = cfg["raw_offsets"].to(
        device=new_positions.device, dtype=new_positions.dtype
    )
    kinematic_chain = cfg["kinematic_chain"]
    face_joint_indx = cfg["face_joint_indx"]
    fid_l = cfg["fid_l"]
    fid_r = cfg["fid_r"]

    B = new_positions.shape[0]
    device = new_positions.device
    dtype = new_positions.dtype

    timers = _new_timer()

    total_steps = warmup + iterations

    with torch.no_grad():
        for i in range(total_steps):
            measure = i >= warmup

            t0 = time.perf_counter()
            new_root_pos = new_positions[:, 0].clone()
            root_height_y = new_root_pos[:, 1:2]
            root_vel_x = new_root_pos[:, 0:1] - prev_positions[:, 0, 0:1]
            root_vel_z = new_root_pos[:, 2:3] - prev_positions[:, 0, 2:3]
            root_features = torch.cat([root_height_y, root_vel_x, root_vel_z], dim=-1)
            _sync_if_cuda(device)
            t1 = time.perf_counter()

            quaternions = _compute_ik(
                new_positions, raw_offsets, kinematic_chain, face_joint_indx
            )
            rotations_6d = quaternion_to_cont6d(quaternions)
            root_quat = quaternions[:, 0]
            _sync_if_cuda(device)
            t2 = time.perf_counter()

            if use_fk_for_ric:
                ric_source = _forward_kinematics(
                    rotations_6d, new_root_pos, raw_offsets, kinematic_chain
                )
            else:
                ric_source = new_positions
            ric = ric_source - ric_source[:, 0:1]
            root_quat_expanded = root_quat.unsqueeze(1).expand(-1, 22, -1)
            ric = qrot(root_quat_expanded, ric)
            _sync_if_cuda(device)
            t3 = time.perf_counter()

            pos_delta = new_positions - prev_positions
            local_vel = qrot(root_quat_expanded, pos_delta)

            vel_l = new_positions[:, fid_l] - prev_positions[:, fid_l]
            vel_r = new_positions[:, fid_r] - prev_positions[:, fid_r]
            feet_l = (torch.sum(vel_l**2, dim=-1) < feet_thre).float()
            feet_r = (torch.sum(vel_r**2, dim=-1) < feet_thre).float()
            foot_contacts = torch.cat([feet_l, feet_r], dim=-1)
            _sync_if_cuda(device)
            t4 = time.perf_counter()

            _ = torch.cat(
                [
                    root_features,
                    ric.reshape(B, -1),
                    rotations_6d.reshape(B, -1),
                    local_vel.reshape(B, -1),
                    foot_contacts,
                ],
                dim=-1,
            )
            _sync_if_cuda(device)
            t5 = time.perf_counter()

            if measure:
                timers["root_features"] += (t1 - t0) * 1000.0
                timers["ik_and_rot6d"] += (t2 - t1) * 1000.0
                timers["ric_transform"] += (t3 - t2) * 1000.0
                timers["local_vel_and_contacts"] += (t4 - t3) * 1000.0
                timers["final_assembly"] += (t5 - t4) * 1000.0
                timers["total_reconstructed"] += (t5 - t0) * 1000.0

    return timers


def _profile_public_function(
    new_positions: torch.Tensor,
    prev_positions: torch.Tensor,
    iterations: int,
    warmup: int,
    feet_thre: float,
    use_fk_for_ric: bool,
) -> float:
    device = new_positions.device
    total_ms = 0.0

    with torch.no_grad():
        for i in range(warmup + iterations):
            t0 = time.perf_counter()
            _ = generated_positions_to_271d(
                new_positions=new_positions,
                prev_positions=prev_positions,
                dataset_type="t2m",
                feet_thre=feet_thre,
                use_fk_for_ric=use_fk_for_ric,
                normalizer=None,
            )
            _sync_if_cuda(device)
            t1 = time.perf_counter()
            if i >= warmup:
                total_ms += (t1 - t0) * 1000.0

    return total_ms


def _cprofile_public_function(
    new_positions: torch.Tensor,
    prev_positions: torch.Tensor,
    iterations: int,
    feet_thre: float,
    use_fk_for_ric: bool,
) -> str:
    profiler = cProfile.Profile()
    profiler.enable()

    with torch.no_grad():
        for _ in range(iterations):
            _ = generated_positions_to_271d(
                new_positions=new_positions,
                prev_positions=prev_positions,
                dataset_type="t2m",
                feet_thre=feet_thre,
                use_fk_for_ric=use_fk_for_ric,
                normalizer=None,
            )

    profiler.disable()

    output = io.StringIO()
    stats = pstats.Stats(profiler, stream=output).sort_stats("cumtime")
    stats.print_stats(25)
    return output.getvalue()


def _print_breakdown(
    timers: Dict[str, float], total_public_ms: float, iterations: int
) -> None:
    reconstructed_total = timers["total_reconstructed"]
    print("\n=== Block Timing Breakdown (reconstructed path) ===")
    print("Block                          total_ms    mean_ms   share")

    ordered_keys = [
        "root_features",
        "ik_and_rot6d",
        "ric_transform",
        "local_vel_and_contacts",
        "final_assembly",
    ]

    for key in ordered_keys:
        total_ms = timers[key]
        mean_ms = total_ms / iterations
        share = (
            (100.0 * total_ms / reconstructed_total) if reconstructed_total > 0 else 0.0
        )
        print(f"{key:28s} {total_ms:10.3f} {mean_ms:9.4f} {share:6.2f}%")

    print(
        f"{'total_reconstructed':28s} {reconstructed_total:10.3f} {reconstructed_total/iterations:9.4f} {100.00:6.2f}%"
    )

    print("\n=== End-to-End Public Function Timing ===")
    print(
        f"generated_positions_to_271d total_ms={total_public_ms:.3f}, mean_ms={total_public_ms/iterations:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile generated_positions_to_271d hotspots"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Execution device"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--iterations", type=int, default=300, help="Measured iterations"
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--feet-thre", type=float, default=0.002, help="Foot contact threshold"
    )
    parser.add_argument(
        "--use-fk-for-ric", action="store_true", help="Enable FK path for RIC"
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.float32

    new_positions = torch.randn(args.batch_size, 22, 3, device=device, dtype=dtype)
    prev_positions = torch.randn(args.batch_size, 22, 3, device=device, dtype=dtype)

    with torch.no_grad():
        frame_public, root_public = generated_positions_to_271d(
            new_positions=new_positions,
            prev_positions=prev_positions,
            dataset_type="t2m",
            feet_thre=args.feet_thre,
            use_fk_for_ric=args.use_fk_for_ric,
            normalizer=None,
        )

    print("=== Profiling Configuration ===")
    print(f"device={device}, dtype={dtype}, batch_size={args.batch_size}")
    print(
        f"iterations={args.iterations}, warmup={args.warmup}, use_fk_for_ric={args.use_fk_for_ric}"
    )
    print(
        f"output_frame_shape={tuple(frame_public.shape)}, output_root_shape={tuple(root_public.shape)}"
    )

    timers = _profile_reconstructed_path(
        new_positions=new_positions,
        prev_positions=prev_positions,
        iterations=args.iterations,
        warmup=args.warmup,
        feet_thre=args.feet_thre,
        use_fk_for_ric=args.use_fk_for_ric,
    )

    total_public_ms = _profile_public_function(
        new_positions=new_positions,
        prev_positions=prev_positions,
        iterations=args.iterations,
        warmup=args.warmup,
        feet_thre=args.feet_thre,
        use_fk_for_ric=args.use_fk_for_ric,
    )

    _print_breakdown(timers, total_public_ms, args.iterations)

    print("\n=== cProfile Top 25 (cumtime) ===")
    print(
        _cprofile_public_function(
            new_positions=new_positions,
            prev_positions=prev_positions,
            iterations=max(20, min(120, args.iterations)),
            feet_thre=args.feet_thre,
            use_fk_for_ric=args.use_fk_for_ric,
        )
    )


if __name__ == "__main__":
    main()
