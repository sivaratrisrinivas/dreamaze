import os

import pytest

from dreamaze.dataset import (
    DatasetConfig,
    DatasetSplitName,
    write_dataset_artifacts,
)
from dreamaze.evaluation import EvaluationConfig, evaluate_conditional_diffusion_solver
from dreamaze.training import TrainingConfig, train_conditional_diffusion_solver


pytestmark = pytest.mark.skipif(
    os.environ.get("DREAMAZE_RUN_OVERFIT_DIAGNOSTIC") != "1",
    reason=(
        "Tiny overfit diagnostic is opt-in; set "
        "DREAMAZE_RUN_OVERFIT_DIAGNOSTIC=1 to run it."
    ),
)


def test_tiny_overfit_diagnostic_generates_non_empty_mask_with_overlap(tmp_path):
    _requires_diffusers_runtime()
    dataset_dir = tmp_path / "dataset"
    checkpoint_dir = tmp_path / "checkpoints"
    write_dataset_artifacts(
        config=DatasetConfig(
            width=4,
            height=4,
            split_sizes={
                DatasetSplitName.TRAIN: 1,
                DatasetSplitName.VALIDATION: 0,
                DatasetSplitName.TEST: 0,
            },
            seed_ranges={
                DatasetSplitName.TRAIN: range(10, 20),
                DatasetSplitName.VALIDATION: range(110, 111),
                DatasetSplitName.TEST: range(210, 211),
            },
            shard_size=1,
            minimum_path_length=2,
        ),
        output_dir=dataset_dir,
    )
    training_result = train_conditional_diffusion_solver(
        TrainingConfig(
            dataset_dir=dataset_dir,
            checkpoint_dir=checkpoint_dir,
            split="train",
            batch_size=1,
            sampling_steps=8,
            max_train_steps=200,
            checkpoint_every_steps=200,
            learning_rate=0.001,
            seed=17,
            device="cpu",
            precision="float32",
        )
    )

    result = evaluate_conditional_diffusion_solver(
        EvaluationConfig(
            dataset_dir=dataset_dir,
            checkpoint_path=training_result.checkpoints[-1],
            split="train",
            sampling_steps=8,
            retry_count=0,
            seed=23,
            device="cpu",
            precision="float32",
        )
    )

    assert result.sampled_tensor_stats["marked_count_mean"] > 0.0
    assert result.sampled_tensor_stats["fraction_at_or_above_threshold"] > 0.0
    assert result.mask_overlap > 0.0
    assert "connected_component_mean" in result.structure_stats
    assert "wall_crossing_count_mean" in result.structure_stats
    assert result.structure_stats["marked_count_mean"] >= 0


def _requires_diffusers_runtime() -> None:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
    except ImportError as error:
        pytest.skip(f"Diffusers runtime is unavailable locally: {error}")
