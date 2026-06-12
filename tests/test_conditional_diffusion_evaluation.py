import json

import pytest

from dreamaze.dataset import (
    DatasetConfig,
    DatasetSplitName,
    write_dataset_artifacts,
)
from dreamaze.evaluation import (
    ConditionalDiffusionSamplingExample,
    EvaluationConfig,
    evaluate_conditional_diffusion_solver,
    load_evaluation_config,
    _endpoint_inclusion_payload,
    _mask_includes_cell,
    _mask_overlap_excluding_cells,
    _sampling_condition_tensor,
)
from dreamaze.evaluation_cli import run_evaluation_cli
from dreamaze.training import (
    TrainingConfig,
    TrainingExampleArrays,
    train_conditional_diffusion_solver,
    _batch_tensors,
)
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
    assert 0.0 <= result.endpoint_inclusion["start_cell_inclusion_rate"] <= 1.0
    assert 0.0 <= result.endpoint_inclusion["goal_cell_inclusion_rate"] <= 1.0
    assert (
        0.0
        <= result.endpoint_inclusion["both_endpoints_inclusion_rate"]
        <= 1.0
    )
    assert (
        0.0
        <= result.endpoint_inclusion["mask_overlap_excluding_endpoints"]
        <= 1.0
    )
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
    assert "endpoint_inclusion" in report
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
    assert "endpoint_inclusion" in report
    assert "sampled_tensor_stats" in report
    cli_output = capsys.readouterr().out
    assert "Conditional Diffusion Solver evaluation complete" in cli_output
    assert "Official score: Single-Sample Success" in cli_output
    assert "Start Cell inclusion:" in cli_output
    assert "Goal Cell inclusion:" in cli_output
    assert "Both endpoints inclusion:" in cli_output
    assert "Mask overlap excluding endpoints:" in cli_output


def test_endpoint_inclusion_payload_reports_separate_endpoint_rates():
    payload = _endpoint_inclusion_payload(
        evaluated_examples=4,
        start_cell_included_count=3,
        goal_cell_included_count=2,
        both_endpoints_included_count=1,
        body_mask_overlaps=[0.25, 0.75],
    )

    assert payload == {
        "start_cell_included_count": 3,
        "goal_cell_included_count": 2,
        "both_endpoints_included_count": 1,
        "start_cell_inclusion_rate": 0.75,
        "goal_cell_inclusion_rate": 0.5,
        "both_endpoints_inclusion_rate": 0.25,
        "mask_overlap_excluding_endpoints": 0.5,
    }


def test_mask_overlap_excluding_cells_reports_body_overlap_without_endpoints():
    proposed_mask = (
        (True, True, False),
        (False, True, False),
        (False, False, True),
    )
    label_mask = (
        (1, 1, 0),
        (0, 0, 0),
        (0, 1, 1),
    )

    assert _mask_overlap_excluding_cells(
        proposed_mask=proposed_mask,
        label_mask=label_mask,
        excluded_cells={(0, 0), (2, 2)},
    ) == pytest.approx(1 / 3)


def test_mask_includes_cell_rejects_unmarked_or_out_of_bounds_cells():
    mask = (
        (False, True),
        (False, False),
    )

    assert _mask_includes_cell(mask, (0, 1)) is True
    assert _mask_includes_cell(mask, (1, 1)) is False
    assert _mask_includes_cell(mask, (-1, 0)) is False
    assert _mask_includes_cell(mask, (2, 0)) is False
    assert _mask_includes_cell(mask, (0, 2)) is False


def test_sampling_condition_tensor_matches_training_condition_tensor():
    example = TrainingExampleArrays(
        maze_condition=(
            (0, 0, 0, 0, 0),
            (0, 1, 1, 1, 0),
            (0, 0, 0, 1, 0),
            (0, 1, 1, 1, 0),
            (0, 0, 0, 0, 0),
        ),
        solution_mask=(
            (0, 0, 0, 0, 0),
            (0, 0, 1, 1, 0),
            (0, 0, 0, 1, 0),
            (0, 1, 1, 1, 0),
            (0, 0, 0, 0, 0),
        ),
        start_cell=(0, 1),
        goal_cell=(1, 0),
    )
    torch = _FakeTorch()

    training_condition, _ = _batch_tensors(
        torch=torch,
        batch=(example,),
        device="cpu",
        dtype="float32",
        sample_size=(8, 8),
    )
    sampling_condition = _sampling_condition_tensor(
        torch=torch,
        example=ConditionalDiffusionSamplingExample(
            maze_condition=example.maze_condition,
            start_cell=example.start_cell,
            goal_cell=example.goal_cell,
        ),
        device="cpu",
        dtype="float32",
        sample_size=(8, 8),
    )

    assert sampling_condition.values == training_condition.values


class _FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = _shape(values)


class _FakeTorch:
    float32 = "float32"

    class nn:
        class functional:
            @staticmethod
            def pad(tensor, padding):
                left, right, top, bottom = padding
                assert left == 0
                assert top == 0
                padded_batches = []
                for batch in tensor.values:
                    padded_channels = []
                    for channel in batch:
                        padded_rows = [row + [0.0] * right for row in channel]
                        width = len(padded_rows[0]) if padded_rows else right
                        padded_rows.extend([[0.0] * width for _ in range(bottom)])
                        padded_channels.append(padded_rows)
                    padded_batches.append(padded_channels)
                return _FakeTensor(padded_batches)

    @staticmethod
    def tensor(values, *, dtype, device):
        assert dtype == "float32"
        assert device == "cpu"
        return _FakeTensor(values)


def _shape(values):
    shape = []
    current = values
    while isinstance(current, list):
        shape.append(len(current))
        current = current[0] if current else []
    return tuple(shape)


def _requires_diffusers_runtime() -> None:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
    except ImportError as error:
        pytest.skip(f"Diffusers runtime is unavailable locally: {error}")
