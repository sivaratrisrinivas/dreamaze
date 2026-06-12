import json
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Mapping

from dreamaze.dataset import (
    DatasetSplitName,
    load_dataset_artifact_manifest,
    load_dataset_artifact_shard,
)
from dreamaze.training import _predict, _rendered_cell
from dreamaze.validation import validate_solution_mask


@dataclass(frozen=True)
class EvaluationConfig:
    dataset_dir: str | Path
    checkpoint_path: str | Path
    split: str = "validation"
    sampling_steps: int = 8
    retry_count: int = 0
    seed: int = 0
    report_path: str | Path | None = None


@dataclass(frozen=True)
class EvaluationMetric:
    valid_count: int
    evaluated_examples: int
    valid_solution_rate: float
    excluded_from_official_score: bool = False

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "valid_count": self.valid_count,
            "evaluated_examples": self.evaluated_examples,
            "valid_solution_rate": self.valid_solution_rate,
            "excluded_from_official_score": self.excluded_from_official_score,
        }


@dataclass(frozen=True)
class EvaluationResult:
    dataset_split: str
    checkpoint_path: Path
    checkpoint_model_type: str
    checkpoint_training_step: int
    checkpoint_sha256: str
    sampling_steps: int
    retry_count: int
    seed: int
    evaluated_examples: int
    single_sample_success: EvaluationMetric
    retry_success: EvaluationMetric | None
    mask_overlap: float
    failure_reason_counts: Mapping[str, int]

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n"

    def to_payload(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "dataset_split": self.dataset_split,
            "checkpoint": {
                "path": str(self.checkpoint_path),
                "model_type": self.checkpoint_model_type,
                "training_step": self.checkpoint_training_step,
                "sha256": self.checkpoint_sha256,
            },
            "sampling": {
                "sampling_steps": self.sampling_steps,
                "retry_count": self.retry_count,
                "seed": self.seed,
            },
            "official_score": "single_sample_success",
            "single_sample_success": self.single_sample_success.to_payload(),
            "mask_overlap": self.mask_overlap,
            "failure_reason_counts": dict(self.failure_reason_counts),
        }
        if self.retry_success is not None:
            payload["retry_success"] = self.retry_success.to_payload()
        return payload


@dataclass(frozen=True)
class ConditionalDiffusionSamplingExample:
    maze_condition: tuple[tuple[int, ...], ...]
    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]


@dataclass(frozen=True)
class _EvaluationExampleArrays:
    maze_condition: tuple[tuple[int, ...], ...]
    solution_mask: tuple[tuple[int, ...], ...]
    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]


def evaluate_conditional_diffusion_solver(
    config: EvaluationConfig,
) -> EvaluationResult:
    _validate_evaluation_config(config)
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint = json.loads(checkpoint_path.read_text())
    weights = checkpoint["weights"]
    examples = _load_evaluation_examples(config)
    rng = Random(config.seed)

    single_sample_valid_count = 0
    retry_valid_count = 0
    failure_reasons: Counter[str] = Counter()
    mask_overlaps: list[float] = []

    for example in examples:
        first_mask = _sample_solution_mask(
            example=example,
            weights=weights,
            sampling_steps=config.sampling_steps,
            rng=rng,
        )
        first_result = validate_solution_mask(
            grid_maze=_bool_grid(example.maze_condition),
            solution_mask=first_mask,
            start_cell=_rendered_cell(example.start_cell),
            goal_cell=_rendered_cell(example.goal_cell),
        )
        mask_overlaps.append(
            _mask_overlap(proposed_mask=first_mask, label_mask=example.solution_mask)
        )

        if first_result.valid:
            single_sample_valid_count += 1
            retry_valid_count += 1
            continue

        failure_reasons[str(first_result.reason)] += 1
        if _retry_finds_valid_solution(
            example=example,
            weights=weights,
            sampling_steps=config.sampling_steps,
            retry_count=config.retry_count,
            rng=rng,
        ):
            retry_valid_count += 1

    evaluated_examples = len(examples)
    single_sample_success = EvaluationMetric(
        valid_count=single_sample_valid_count,
        evaluated_examples=evaluated_examples,
        valid_solution_rate=single_sample_valid_count / evaluated_examples,
    )
    retry_success = None
    if config.retry_count > 0:
        retry_success = EvaluationMetric(
            valid_count=retry_valid_count,
            evaluated_examples=evaluated_examples,
            valid_solution_rate=retry_valid_count / evaluated_examples,
            excluded_from_official_score=True,
        )

    return EvaluationResult(
        dataset_split=config.split,
        checkpoint_path=checkpoint_path,
        checkpoint_model_type=checkpoint["model_type"],
        checkpoint_training_step=checkpoint["training_step"],
        checkpoint_sha256=_sha256_file(checkpoint_path),
        sampling_steps=config.sampling_steps,
        retry_count=config.retry_count,
        seed=config.seed,
        evaluated_examples=evaluated_examples,
        single_sample_success=single_sample_success,
        retry_success=retry_success,
        mask_overlap=sum(mask_overlaps) / evaluated_examples,
        failure_reason_counts=dict(sorted(failure_reasons.items())),
    )


def sample_conditional_diffusion_solution_mask(
    *,
    example: ConditionalDiffusionSamplingExample,
    weights: Mapping[str, float],
    sampling_steps: int,
    rng: Random,
) -> tuple[tuple[bool, ...], ...]:
    if sampling_steps < 1:
        raise ValueError("Conditional Diffusion Solver sampling steps must be positive")
    return _sample_solution_mask(
        example=_EvaluationExampleArrays(
            maze_condition=example.maze_condition,
            solution_mask=(),
            start_cell=example.start_cell,
            goal_cell=example.goal_cell,
        ),
        weights=weights,
        sampling_steps=sampling_steps,
        rng=rng,
    )


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    payload = json.loads(Path(path).read_text())
    config = EvaluationConfig(
        dataset_dir=Path(payload["dataset_dir"]),
        checkpoint_path=Path(payload["checkpoint_path"]),
        split=payload.get("split", "validation"),
        sampling_steps=payload["sampling_steps"],
        retry_count=payload.get("retry_count", 0),
        seed=payload.get("seed", 0),
        report_path=(
            Path(payload["report_path"]) if payload.get("report_path") else None
        ),
    )
    _validate_evaluation_config(config)
    return config


def _retry_finds_valid_solution(
    *,
    example: _EvaluationExampleArrays,
    weights: Mapping[str, float],
    sampling_steps: int,
    retry_count: int,
    rng: Random,
) -> bool:
    for _ in range(retry_count):
        mask = _sample_solution_mask(
            example=example,
            weights=weights,
            sampling_steps=sampling_steps,
            rng=rng,
        )
        result = validate_solution_mask(
            grid_maze=_bool_grid(example.maze_condition),
            solution_mask=mask,
            start_cell=_rendered_cell(example.start_cell),
            goal_cell=_rendered_cell(example.goal_cell),
        )
        if result.valid:
            return True
    return False


def _sample_solution_mask(
    *,
    example: _EvaluationExampleArrays,
    weights: Mapping[str, float],
    sampling_steps: int,
    rng: Random,
) -> tuple[tuple[bool, ...], ...]:
    current_mask = tuple(
        tuple(rng.choice((0.0, 1.0)) for _ in row) for row in example.maze_condition
    )
    start_cell = _rendered_cell(example.start_cell)
    goal_cell = _rendered_cell(example.goal_cell)

    for timestep in range(sampling_steps, 0, -1):
        timestep_feature = timestep / sampling_steps
        next_mask: list[tuple[float, ...]] = []
        for row_index, row in enumerate(example.maze_condition):
            next_row: list[float] = []
            for column_index, is_open in enumerate(row):
                features = {
                    "bias": 1.0,
                    "maze_open": float(is_open),
                    "start": float((row_index, column_index) == start_cell),
                    "goal": float((row_index, column_index) == goal_cell),
                    "noisy_mask": current_mask[row_index][column_index],
                    "timestep": timestep_feature,
                }
                next_row.append(
                    1.0 if _predict(weights, features) >= rng.random() else 0.0
                )
            next_mask.append(tuple(next_row))
        current_mask = tuple(next_mask)

    return tuple(tuple(bool(value) for value in row) for row in current_mask)


def _load_evaluation_examples(
    config: EvaluationConfig,
) -> tuple[_EvaluationExampleArrays, ...]:
    dataset_dir = Path(config.dataset_dir)
    manifest = load_dataset_artifact_manifest(dataset_dir / "manifest.json")
    requested_split = DatasetSplitName(config.split)
    examples: list[_EvaluationExampleArrays] = []

    for shard in manifest.shards:
        if shard.split != requested_split:
            continue
        payload = load_dataset_artifact_shard(dataset_dir / shard.name)
        examples.extend(_examples_from_shard_payload(payload))

    if not examples:
        raise ValueError(f"No Dataset Artifact examples found for {config.split} split")

    return tuple(examples)


def _examples_from_shard_payload(
    payload: Mapping[str, Any]
) -> tuple[_EvaluationExampleArrays, ...]:
    return tuple(
        _EvaluationExampleArrays(
            maze_condition=_grid(payload["maze_condition"][index]),
            solution_mask=_grid(payload["solution_mask"][index]),
            start_cell=tuple(payload["start_cell"][index]),
            goal_cell=tuple(payload["goal_cell"][index]),
        )
        for index in range(len(payload["maze_condition"]))
    )


def _mask_overlap(
    *,
    proposed_mask: tuple[tuple[bool, ...], ...],
    label_mask: tuple[tuple[int, ...], ...],
) -> float:
    proposed_cells = _marked_cells(proposed_mask)
    label_cells = _marked_cells(label_mask)
    union = proposed_cells | label_cells
    if not union:
        return 1.0
    return len(proposed_cells & label_cells) / len(union)


def _marked_cells(mask: tuple[tuple[bool | int, ...], ...]) -> set[tuple[int, int]]:
    return {
        (row_index, column_index)
        for row_index, row in enumerate(mask)
        for column_index, is_marked in enumerate(row)
        if is_marked
    }


def _validate_evaluation_config(config: EvaluationConfig) -> None:
    DatasetSplitName(config.split)
    if config.sampling_steps < 1:
        raise ValueError("Evaluation sampling steps must be positive")
    if config.retry_count < 0:
        raise ValueError("Evaluation retry count cannot be negative")


def _bool_grid(rows: tuple[tuple[int, ...], ...]) -> tuple[tuple[bool, ...], ...]:
    return tuple(tuple(bool(value) for value in row) for row in rows)


def _grid(rows: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
