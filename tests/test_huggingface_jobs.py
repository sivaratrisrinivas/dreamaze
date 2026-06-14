import json
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

from dreamaze.huggingface_jobs import (
    BEST_GPU_HARDWARE_FLAVOR,
    HuggingFaceJobConfig,
    build_huggingface_job_command,
    load_huggingface_job_config,
)
from dreamaze.huggingface_jobs_cli import run_huggingface_jobs_cli


def test_huggingface_job_config_builds_remote_ready_dry_run_command(tmp_path):
    config_path = tmp_path / "hf-job.json"
    config_path.write_text(
        json.dumps(
            {
                "hardware_flavor": "t4-small",
                "timeout": "45m",
                "python": "3.12",
                "repo": "Srini410/dreamaze-job-runs",
                "namespace": "Srini410",
                "dataset_preset": "tiny",
                "dataset_dir": "/tmp/dreamaze/dataset",
                "checkpoint_dir": "/tmp/dreamaze/checkpoints",
                "evaluation_output": "/tmp/dreamaze/evaluation.json",
                "batch_size": 4,
                "sampling_steps": 6,
                "max_train_steps": 7,
                "checkpoint_every_steps": 2,
                "learning_rate": 0.025,
                "positive_loss_weight": 8.0,
                "endpoint_loss_weight": 32.0,
                "mask_bce_loss_weight": 1.5,
                "mask_dice_loss_weight": 0.75,
                "wall_loss_weight": 2.5,
                "path_continuity_loss_weight": 1.25,
                "off_path_loss_weight": 3.5,
                "training_seed": 123,
                "evaluation_seed": 456,
                "retry_count": 3,
                "eval_split": "test",
                "device": "cuda",
                "precision": "float16",
                "num_workers": 2,
                "output_repo": "Srini410/dreamaze-artifacts",
                "output_repo_type": "dataset",
                "output_path_in_repo": "runs/smoke",
                "model_output_repo": "Srini410/dreamaze-solver",
                "model_output_path_in_repo": "checkpoints/smoke",
                "maze_family": "kruskal",
                "env": {"HF_XET_HIGH_PERFORMANCE": "1"},
            }
        )
    )

    config = load_huggingface_job_config(config_path)
    command = build_huggingface_job_command(config)

    assert config == HuggingFaceJobConfig(
        hardware_flavor="t4-small",
        timeout="45m",
        python="3.12",
        repo="Srini410/dreamaze-job-runs",
        namespace="Srini410",
        dataset_preset="tiny",
        dataset_dir="/tmp/dreamaze/dataset",
        checkpoint_dir="/tmp/dreamaze/checkpoints",
        evaluation_output="/tmp/dreamaze/evaluation.json",
        batch_size=4,
        sampling_steps=6,
        max_train_steps=7,
        checkpoint_every_steps=2,
        learning_rate=0.025,
        positive_loss_weight=8.0,
        endpoint_loss_weight=32.0,
        mask_bce_loss_weight=1.5,
        mask_dice_loss_weight=0.75,
        wall_loss_weight=2.5,
        path_continuity_loss_weight=1.25,
        off_path_loss_weight=3.5,
        training_seed=123,
        evaluation_seed=456,
        retry_count=3,
        eval_split="test",
        device="cuda",
        precision="float16",
        num_workers=2,
        output_repo="Srini410/dreamaze-artifacts",
        output_repo_type="dataset",
        output_path_in_repo="runs/smoke",
        model_output_repo="Srini410/dreamaze-solver",
        model_output_path_in_repo="checkpoints/smoke",
        maze_family="kruskal",
        env={"HF_XET_HIGH_PERFORMANCE": "1"},
    )
    assert command[:4] == ["hf", "jobs", "uv", "run"]
    assert command.index("--flavor") < command.index("jobs/dreamaze_hf_job.py")
    assert command.index("--timeout") < command.index("jobs/dreamaze_hf_job.py")
    assert "--flavor" in command
    assert "t4-small" in command
    assert "--timeout" in command
    assert "45m" in command
    assert "--secrets" in command
    assert "HF_TOKEN" in command
    assert "--batch-size" in command
    assert "4" in command
    assert "--sampling-steps" in command
    assert "6" in command
    assert "--checkpoint-every-steps" in command
    assert "2" in command
    assert "--positive-loss-weight" in command
    assert "8.0" in command
    assert "--endpoint-loss-weight" in command
    assert "32.0" in command
    assert "--mask-bce-loss-weight" in command
    assert "1.5" in command
    assert "--mask-dice-loss-weight" in command
    assert "0.75" in command
    assert "--wall-loss-weight" in command
    assert "2.5" in command
    assert "--path-continuity-loss-weight" in command
    assert "1.25" in command
    assert "--off-path-loss-weight" in command
    assert "3.5" in command
    assert "--maze-family" in command
    assert "kruskal" in command
    assert "--output-repo" in command
    assert "Srini410/dreamaze-artifacts" in command
    assert "--model-output-repo" in command
    assert "Srini410/dreamaze-solver" in command


def test_huggingface_job_config_can_select_best_gpu_profile(tmp_path):
    config_path = tmp_path / "hf-job.json"
    config_path.write_text(
        json.dumps(
            {
                "compute_profile": "best_gpu",
                "output_repo": "Srini410/dreamaze-artifacts",
            }
        )
    )

    config = load_huggingface_job_config(config_path)
    command = build_huggingface_job_command(config)

    assert config.hardware_flavor == BEST_GPU_HARDWARE_FLAVOR
    assert config.dataset_preset == "first"
    assert config.device == "cuda"
    assert config.precision == "float16"
    assert config.batch_size == 16
    assert config.sampling_steps == 32
    assert config.max_train_steps == 1000
    assert config.checkpoint_every_steps == 200
    assert config.timeout == "2h"
    assert "--flavor" in command
    assert BEST_GPU_HARDWARE_FLAVOR in command
    assert "--device" in command
    assert "cuda" in command


def test_huggingface_job_config_can_select_larger_gpu_profile(tmp_path):
    config_path = tmp_path / "hf-job.json"
    config_path.write_text(
        json.dumps(
            {
                "compute_profile": "larger_gpu",
                "output_repo": "Srini410/dreamaze-artifacts",
                "model_output_repo": "Srini410/dreamaze-solver",
            }
        )
    )

    config = load_huggingface_job_config(config_path)
    command = build_huggingface_job_command(config)

    assert config.hardware_flavor == BEST_GPU_HARDWARE_FLAVOR
    assert config.dataset_preset == "larger"
    assert config.timeout == "8h"
    assert config.max_train_steps == 10_000
    assert config.checkpoint_every_steps == 1_000
    assert config.retry_count == 0
    assert "--dataset-preset" in command
    assert "larger" in command
    assert "--model-output-repo" in command
    assert "Srini410/dreamaze-solver" in command


def test_checked_in_larger_gpu_job_config_includes_structure_losses():
    config_path = _REPO_ROOT / "configs" / "hf-job-larger-gpu.json"
    config = load_huggingface_job_config(config_path)
    command = build_huggingface_job_command(config)

    assert config.dataset_preset == "larger"
    assert config.maze_family == "kruskal"
    assert config.path_continuity_loss_weight == 1.0
    assert config.off_path_loss_weight == 3.0
    assert config.wall_loss_weight == 1.0
    assert config.mask_bce_loss_weight == 1.0
    assert config.mask_dice_loss_weight == 1.0
    assert "--path-continuity-loss-weight" in command
    assert "1.0" in command
    assert "--off-path-loss-weight" in command
    assert "3.0" in command


def test_checked_in_tiny_overfit_gpu_job_config_matches_structure_loss_stack():
    config_path = _REPO_ROOT / "configs" / "hf-job-tiny-overfit-gpu.json"
    config = load_huggingface_job_config(config_path)

    assert config.dataset_preset == "tiny"
    assert config.maze_family == "kruskal"
    assert config.path_continuity_loss_weight == 1.0
    assert config.off_path_loss_weight == 3.0
    assert config.wall_loss_weight == 1.0
    assert config.mask_bce_loss_weight == 1.0
    assert config.mask_dice_loss_weight == 1.0


def test_huggingface_job_cli_dry_run_prints_command_without_launching(tmp_path, capsys):
    config_path = tmp_path / "hf-job.json"
    config_path.write_text(
        json.dumps(
            {
                "hardware_flavor": "cpu-basic",
                "timeout": "10m",
                "dataset_preset": "tiny",
                "batch_size": 1,
                "sampling_steps": 2,
                "max_train_steps": 1,
                "checkpoint_every_steps": 1,
            }
        )
    )

    exit_code = run_huggingface_jobs_cli(["--config", str(config_path), "--dry-run"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Hugging Face Jobs dry run" in output
    assert "hf jobs uv run" in output
    assert "--flavor cpu-basic" in output
    assert "--dataset-preset tiny" in output


def test_huggingface_job_script_runs_tiny_build_train_evaluate_workflow(
    tmp_path, monkeypatch
):
    _requires_diffusers_runtime()
    script_path = "jobs/dreamaze_hf_job.py"
    spec = importlib.util.spec_from_file_location("dreamaze_hf_job", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dataset_dir = tmp_path / "dataset"
    checkpoint_dir = tmp_path / "checkpoints"
    evaluation_output = tmp_path / "evaluation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            script_path,
            "--dataset-preset",
            "tiny",
            "--dataset-dir",
            str(dataset_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--evaluation-output",
            str(evaluation_output),
            "--batch-size",
            "1",
            "--sampling-steps",
            "2",
            "--max-train-steps",
            "1",
            "--checkpoint-every-steps",
            "1",
            "--learning-rate",
            "0.01",
            "--positive-loss-weight",
            "8.0",
            "--endpoint-loss-weight",
            "32.0",
            "--mask-bce-loss-weight",
            "1.5",
            "--mask-dice-loss-weight",
            "0.75",
            "--wall-loss-weight",
            "2.5",
            "--path-continuity-loss-weight",
            "1.25",
            "--off-path-loss-weight",
            "3.5",
            "--training-seed",
            "7",
            "--evaluation-seed",
            "11",
            "--retry-count",
            "1",
            "--eval-split",
            "validation",
            "--device",
            "cpu",
            "--precision",
            "float32",
            "--num-workers",
            "0",
            "--maze-family",
            "kruskal",
        ],
    )

    exit_code = module.main()

    assert exit_code == 0
    assert (dataset_dir / "manifest.json").exists()
    assert (checkpoint_dir / "checkpoint-step-000001" / "metadata.json").exists()
    report = json.loads(evaluation_output.read_text())
    assert report["dataset_split"] == "validation"
    assert report["official_score"] == "single_sample_success"
    assert "structure_stats" in report
    assert "connected_component_mean" in report.get("structure_stats", {})
    summary = json.loads((tmp_path / "hf-job-summary.json").read_text())
    assert summary["maze_family"] == "kruskal"
    assert summary["positive_loss_weight"] == 8.0
    assert summary["endpoint_loss_weight"] == 32.0
    assert summary["mask_bce_loss_weight"] == 1.5
    assert summary["mask_dice_loss_weight"] == 0.75
    assert summary["wall_loss_weight"] == 2.5
    assert summary["path_continuity_loss_weight"] == 1.25
    assert summary["off_path_loss_weight"] == 3.5


def _requires_diffusers_runtime() -> None:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
    except ImportError as error:
        pytest.skip(f"Diffusers runtime is unavailable locally: {error}")
