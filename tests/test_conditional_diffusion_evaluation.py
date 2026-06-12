import json

import pytest

from dreamaze.dataset import (
    DatasetConfig,
    DatasetSplitName,
    write_dataset_artifacts,
)
from dreamaze.evaluation import (
    EvaluationConfig,
    evaluate_conditional_diffusion_solver,
    load_evaluation_config,
)
from dreamaze.evaluation_cli import run_evaluation_cli
from dreamaze.training import TrainingConfig, train_conditional_diffusion_solver
from dreamaze.training import DIFFUSERS_MODEL_TYPE


def test_evaluation_reports_single_sample_success_and_diagnostics(tmp_path):
    _requires_diffusers_runtime()
    dataset_dir = tmp_path / "dataset"
    checkpoint_dir = tmp_path / "checkpoints"
    write_dataset_artifacts(
        config=DatasetConfig(
            width=4,
            height=4,
            split_sizes={
                DatasetSplitName.TRAIN: 2,
                DatasetSplitName.VALIDATION: 2,
                DatasetSplitName.TEST: 0,
            },
            seed_ranges={
                DatasetSplitName.TRAIN: range(10, 20),
                DatasetSplitName.VALIDATION: range(110, 120),
                DatasetSplitName.TEST: range(210, 211),
            },
            shard_size=1,
        ),
        output_dir=dataset_dir,
    )
    training_result = train_conditional_diffusion_solver(
        TrainingConfig(
            dataset_dir=dataset_dir,
            checkpoint_dir=checkpoint_dir,
            split="train",
            batch_size=1,
            sampling_steps=2,
            max_train_steps=1,
            checkpoint_every_steps=1,
            seed=7,
        )
    )

    result = evaluate_conditional_diffusion_solver(
        EvaluationConfig(
            dataset_dir=dataset_dir,
            checkpoint_path=training_result.checkpoints[-1],
            split="validation",
            sampling_steps=2,
            retry_count=1,
            seed=11,
        )
    )

    assert result.dataset_split == "validation"
    assert result.checkpoint_path == training_result.checkpoints[-1]
    assert result.sampling_steps == 2
    assert result.evaluated_examples == 2
    assert 0.0 <= result.single_sample_success.valid_solution_rate <= 1.0
    assert result.retry_success is not None
    assert 0.0 <= result.retry_success.valid_solution_rate <= 1.0
    assert result.retry_success.valid_solution_rate >= (
        result.single_sample_success.valid_solution_rate
    )
    assert result.retry_success.excluded_from_official_score is True
    assert 0.0 <= result.mask_overlap <= 1.0
    assert result.sampled_tensor_stats["total_cells"] == 162
    assert result.sampled_tensor_stats["raw_min"] <= result.sampled_tensor_stats["raw_max"]
    assert (
        0.0
        <= result.sampled_tensor_stats["fraction_at_or_above_threshold"]
        <= 1.0
    )
    assert sum(result.failure_reason_counts.values()) == (
        result.evaluated_examples - result.single_sample_success.valid_count
    )

    report = json.loads(result.to_json())
    assert report["dataset_split"] == "validation"
    assert report["checkpoint"]["model_type"] == DIFFUSERS_MODEL_TYPE
    assert len(report["checkpoint"]["sha256"]) == 64
    assert report["sampling"]["retry_count"] == 1
    assert report["sampling"]["seed"] == 11
    assert report["official_score"] == "single_sample_success"
    assert "sampled_tensor_stats" in report
    assert "raw_mean" in report["sampled_tensor_stats"]
    assert report["retry_success"]["excluded_from_official_score"] is True


def test_evaluation_cli_writes_json_report_from_config_file(tmp_path, capsys):
    _requires_diffusers_runtime()
    dataset_dir = tmp_path / "dataset"
    checkpoint_dir = tmp_path / "checkpoints"
    report_path = tmp_path / "evaluation.json"
    config_path = tmp_path / "evaluation-config.json"
    write_dataset_artifacts(
        config=DatasetConfig(
            width=4,
            height=4,
            split_sizes={
                DatasetSplitName.TRAIN: 1,
                DatasetSplitName.VALIDATION: 1,
                DatasetSplitName.TEST: 0,
            },
            seed_ranges={
                DatasetSplitName.TRAIN: range(20, 30),
                DatasetSplitName.VALIDATION: range(120, 130),
                DatasetSplitName.TEST: range(220, 221),
            },
        ),
        output_dir=dataset_dir,
    )
    training_result = train_conditional_diffusion_solver(
        TrainingConfig(
            dataset_dir=dataset_dir,
            checkpoint_dir=checkpoint_dir,
            split="train",
            batch_size=1,
            sampling_steps=2,
            max_train_steps=1,
            checkpoint_every_steps=1,
        )
    )
    config_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "checkpoint_path": str(training_result.checkpoints[-1]),
                "split": "validation",
                "sampling_steps": 2,
                "retry_count": 1,
                "seed": 5,
                "report_path": str(report_path),
                "device": "cpu",
                "precision": "float32",
            }
        )
    )

    config = load_evaluation_config(config_path)
    exit_code = run_evaluation_cli(["--config", str(config_path)])

    assert config.report_path == report_path
    assert exit_code == 0
    report = json.loads(report_path.read_text())
    assert report["dataset_split"] == "validation"
    assert report["checkpoint"]["path"] == str(training_result.checkpoints[-1])
    assert report["official_score"] == "single_sample_success"
    assert "failure_reason_counts" in report
    assert "sampled_tensor_stats" in report
    cli_output = capsys.readouterr().out
    assert "Conditional Diffusion Solver evaluation complete" in cli_output
    assert "Official score: Single-Sample Success" in cli_output


def _requires_diffusers_runtime() -> None:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
    except ImportError as error:
        pytest.skip(f"Diffusers runtime is unavailable locally: {error}")
