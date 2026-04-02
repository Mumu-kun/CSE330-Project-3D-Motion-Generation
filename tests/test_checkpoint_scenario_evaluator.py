import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from utils.checkpoint_eval import build_default_scenarios, load_checkpoint_config, select_sample_ids_by_length_quantiles

import evaluate_checkpoint_scenarios


CHECKPOINT_PATH = ROOT / "tests" / "checkpoints" / "latest5.pt"
DATASET_PATH = ROOT / "tests" / "dataset" / "humanml3d-subset-mini"


def test_checkpoint_config_and_scenario_defaults() -> None:
    config = load_checkpoint_config(CHECKPOINT_PATH)
    scenarios = {scenario.name: scenario for scenario in build_default_scenarios(config)}

    assert config.horizon == 40
    assert config.num_inference_steps == 20
    assert scenarios["rollout_default"].seed_length == config.horizon
    assert scenarios["rollout_default"].num_steps == config.num_inference_steps
    assert scenarios["rollout_short_context"].seed_length == max(1, config.horizon // 2)
    assert scenarios["rollout_high_steps"].num_steps == max(
        config.num_inference_steps * 2,
        config.num_inference_steps + 10,
    )


def test_deterministic_sample_selection_on_mini_test_split() -> None:
    selected = select_sample_ids_by_length_quantiles(DATASET_PATH, split="test", num_samples=3)
    assert selected == ["000110", "000171", "000078"]


def test_render_report_flags_rollout_drift() -> None:
    summary = {
        "checkpoint": {"path": "dummy.pt"},
        "split": "test",
        "selected_samples": [{"sample_id": "000110"}],
        "config": {"horizon": 40, "num_inference_steps": 20, "use_fk": False},
        "scenario_aggregates": {
            "teacher_forced_one_step": {
                "joint_l2_mean": 0.02,
                "root_l2_mean": 0.01,
                "nonroot_joint_l2_mean": 0.03,
            },
            "mixed_teacher_force_rollout": {
                "joint_l2_mean": 0.08,
                "root_l2_mean": 0.04,
                "nonroot_joint_l2_mean": 0.09,
            },
            "rollout_default": {
                "joint_l2_mean": 1.5,
                "root_l2_mean": 2.0,
                "nonroot_joint_l2_mean": 1.0,
            },
            "rollout_short_context": {
                "joint_l2_mean": 1.8,
                "root_l2_mean": 2.2,
                "nonroot_joint_l2_mean": 1.3,
            },
            "rollout_high_steps": {
                "joint_l2_mean": 1.55,
                "root_l2_mean": 2.0,
                "nonroot_joint_l2_mean": 1.05,
            },
        },
        "scenario_top_joints": {
            "rollout_default": [
                {"joint_index": 0, "joint_name": "root", "mean_l2": 2.0},
                {"joint_index": 10, "joint_name": "right_foot", "mean_l2": 1.6},
            ]
        },
    }

    report = evaluate_checkpoint_scenarios.render_report(summary)
    assert "Rollout drift dominates" in report


def test_run_checkpoint_scenario_evaluation_writes_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "checkpoint_eval"
    summary = evaluate_checkpoint_scenarios.run_checkpoint_scenario_evaluation(
        checkpoint_path=CHECKPOINT_PATH,
        dataset_path=DATASET_PATH,
        output_dir=output_dir,
        sample_ids=["000110"],
        device="cpu",
        seed=0,
        render_videos=False,
    )

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "scenario_metrics.csv").exists()
    assert (output_dir / "per_frame_metrics.csv").exists()
    assert (output_dir / "per_joint_metrics.csv").exists()
    assert (output_dir / "report.md").exists()

    parsed_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    scenario_aggregates = parsed_summary["scenario_aggregates"]

    assert scenario_aggregates["teacher_forced_one_step"]["joint_l2_mean"] < scenario_aggregates["rollout_default"]["joint_l2_mean"]
    assert "rollout_high_steps" in scenario_aggregates
    assert scenario_aggregates["rollout_high_steps"]["num_steps"] > scenario_aggregates["rollout_default"]["num_steps"]
    assert summary["scenario_aggregates"]["rollout_default"]["joint_l2_mean"] == scenario_aggregates["rollout_default"]["joint_l2_mean"]
