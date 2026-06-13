import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from dreamaze.dataset import (
    DatasetSplitName,
    load_dataset_artifact_manifest,
    load_dataset_artifact_shard,
)
from dreamaze.training import (
    DIFFUSERS_MODEL_TYPE,
    TrainingExampleArrays,
    _ceil_multiple,
    _condition_channels,
    _pad_to_sample_size,
    _rendered_cell,
    _torch_device,
    _torch_dtype,
    load_checkpoint_metadata,
)
from dreamaze.validation import validate_solution_mask

_THRESHOLD_CALIBRATION_CANDIDATES = (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75)


@dataclass(frozen=True)
class EvaluationConfig:
    dataset_dir: str | Path
    checkpoint_path: str | Path
    split: str = "validation"
    sampling_steps: int = 32
    retry_count: int = 0
    seed: int = 0
    report_path: str | Path | None = None
    device: str = "cpu"
    precision: str = "float32"


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
    endpoint_inclusion: Mapping[str, float | int]
    sampled_tensor_stats: Mapping[str, float | int]
    endpoint_raw_values: Mapping[str, float | int]
    endpoint_raw_value_examples: tuple[Mapping[str, Any], ...]
    failure_reason_counts: Mapping[str, int]
    threshold_calibration: tuple[Mapping[str, Any], ...]

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
            "endpoint_inclusion": dict(self.endpoint_inclusion),
            "sampled_tensor_stats": dict(self.sampled_tensor_stats),
            "endpoint_raw_values": dict(self.endpoint_raw_values),
            "endpoint_raw_value_examples": [
                dict(item) for item in self.endpoint_raw_value_examples
            ],
            "failure_reason_counts": dict(self.failure_reason_counts),
            "threshold_calibration": [
                dict(item) for item in self.threshold_calibration
            ],
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
class SampledTensorStats:
    raw_min: float
    raw_max: float
    raw_mean: float
    fraction_at_or_above_threshold: float
    marked_count: int
    total_cells: int

    def to_payload(self) -> Mapping[str, float | int]:
        return {
            "raw_min": self.raw_min,
            "raw_max": self.raw_max,
            "raw_mean": self.raw_mean,
            "fraction_at_or_above_threshold": self.fraction_at_or_above_threshold,
            "marked_count": self.marked_count,
            "total_cells": self.total_cells,
        }


@dataclass(frozen=True)
class SampledSolutionMask:
    mask: tuple[tuple[bool, ...], ...]
    tensor_stats: SampledTensorStats
    raw_values: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class EndpointRawValueDiagnostics:
    start_cell_raw_value: float
    goal_cell_raw_value: float
    label_cell_raw_mean: float
    label_cell_raw_min: float
    label_cell_raw_max: float
    non_label_cell_raw_mean: float
    non_label_cell_raw_min: float
    non_label_cell_raw_max: float
    start_cell_descending_rank: int
    goal_cell_descending_rank: int
    start_cell_percentile: float
    goal_cell_percentile: float


@dataclass(frozen=True)
class _LoadedDiffusersSolver:
    torch: Any
    model: Any
    scheduler: Any
    device: Any
    dtype: Any


def evaluate_conditional_diffusion_solver(
    config: EvaluationConfig,
) -> EvaluationResult:
    _validate_evaluation_config(config)
    checkpoint_path = Path(config.checkpoint_path)
    metadata = load_checkpoint_metadata(checkpoint_path)
    _validate_checkpoint_metadata(metadata)
    solver = load_diffusers_solver_checkpoint(
        checkpoint_path=checkpoint_path,
        device=config.device,
        precision=config.precision,
    )
    examples = _load_evaluation_examples(config)

    single_sample_valid_count = 0
    retry_valid_count = 0
    failure_reasons: Counter[str] = Counter()
    mask_overlaps: list[float] = []
    body_mask_overlaps: list[float] = []
    start_cell_included_count = 0
    goal_cell_included_count = 0
    both_endpoints_included_count = 0
    sampled_tensor_stats: list[SampledTensorStats] = []
    endpoint_raw_value_diagnostics: list[EndpointRawValueDiagnostics] = []
    endpoint_raw_value_examples: list[Mapping[str, Any]] = []
    threshold_calibration_examples: list[tuple[TrainingExampleArrays, SampledSolutionMask]] = []

    for index, example in enumerate(examples):
        first_sample = sample_conditional_diffusion_solution_mask_with_stats(
            example=_sampling_example_from_arrays(example),
            solver=solver,
            sampling_steps=config.sampling_steps,
            seed=config.seed + index,
        )
        first_mask = first_sample.mask
        threshold_calibration_examples.append((example, first_sample))
        sampled_tensor_stats.append(first_sample.tensor_stats)
        rendered_start_cell = _rendered_cell(example.start_cell)
        rendered_goal_cell = _rendered_cell(example.goal_cell)
        endpoint_diagnostics = _endpoint_raw_value_diagnostics(
            raw_values=first_sample.raw_values,
            label_mask=example.solution_mask,
            start_cell=rendered_start_cell,
            goal_cell=rendered_goal_cell,
        )
        endpoint_raw_value_diagnostics.append(endpoint_diagnostics)
        first_result = validate_solution_mask(
            grid_maze=_bool_grid(example.maze_condition),
            solution_mask=first_mask,
            start_cell=rendered_start_cell,
            goal_cell=rendered_goal_cell,
        )
        mask_overlaps.append(
            _mask_overlap(proposed_mask=first_mask, label_mask=example.solution_mask)
        )
        start_cell_included = _mask_includes_cell(first_mask, rendered_start_cell)
        goal_cell_included = _mask_includes_cell(first_mask, rendered_goal_cell)
        if start_cell_included:
            start_cell_included_count += 1
        if goal_cell_included:
            goal_cell_included_count += 1
        if start_cell_included and goal_cell_included:
            both_endpoints_included_count += 1
        endpoint_raw_value_examples.append(
            _endpoint_raw_value_example_payload(
                example_index=index,
                validation_failure_reason=(
                    None if first_result.reason is None else str(first_result.reason)
                ),
                start_cell_included=start_cell_included,
                goal_cell_included=goal_cell_included,
                diagnostics=endpoint_diagnostics,
                tensor_stats_marked_count=first_sample.tensor_stats.marked_count,
                tensor_stats_fraction_at_or_above_threshold=(
                    first_sample.tensor_stats.fraction_at_or_above_threshold
                ),
            )
        )
        body_mask_overlaps.append(
            _mask_overlap_excluding_cells(
                proposed_mask=first_mask,
                label_mask=example.solution_mask,
                excluded_cells={rendered_start_cell, rendered_goal_cell},
            )
        )

        if first_result.valid:
            single_sample_valid_count += 1
            retry_valid_count += 1
            continue

        failure_reasons[str(first_result.reason)] += 1
        if _retry_finds_valid_solution(
            example=example,
            solver=solver,
            sampling_steps=config.sampling_steps,
            retry_count=config.retry_count,
            seed=config.seed + 10_000 + index,
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
        checkpoint_model_type=metadata["model_type"],
        checkpoint_training_step=metadata["training_step"],
        checkpoint_sha256=_sha256_path(checkpoint_path),
        sampling_steps=config.sampling_steps,
        retry_count=config.retry_count,
        seed=config.seed,
        evaluated_examples=evaluated_examples,
        single_sample_success=single_sample_success,
        retry_success=retry_success,
        mask_overlap=sum(mask_overlaps) / evaluated_examples,
        endpoint_inclusion=_endpoint_inclusion_payload(
            evaluated_examples=evaluated_examples,
            start_cell_included_count=start_cell_included_count,
            goal_cell_included_count=goal_cell_included_count,
            both_endpoints_included_count=both_endpoints_included_count,
            body_mask_overlaps=body_mask_overlaps,
        ),
        sampled_tensor_stats=_aggregate_sampled_tensor_stats(sampled_tensor_stats),
        endpoint_raw_values=_aggregate_endpoint_raw_value_diagnostics(
            endpoint_raw_value_diagnostics
        ),
        endpoint_raw_value_examples=tuple(endpoint_raw_value_examples),
        failure_reason_counts=dict(sorted(failure_reasons.items())),
        threshold_calibration=_threshold_calibration_payloads(
            threshold_calibration_examples
        ),
    )


def load_diffusers_solver_checkpoint(
    *, checkpoint_path: str | Path, device: str = "cpu", precision: str = "float32"
) -> _LoadedDiffusersSolver:
    try:
        import torch
        from diffusers import DDPMScheduler, UNet2DModel
    except ImportError as error:
        raise RuntimeError(
            "Dreamaze Runtime Solving requires torch and diffusers. "
            "Run the Proof Demo on a configured Hugging Face environment."
        ) from error

    checkpoint = Path(checkpoint_path)
    metadata = load_checkpoint_metadata(checkpoint)
    _validate_checkpoint_metadata(metadata)
    torch_device = _torch_device(torch, device)
    dtype = _torch_dtype(torch, precision)
    model = UNet2DModel.from_pretrained(checkpoint / "unet").to(
        device=torch_device, dtype=dtype
    )
    scheduler = DDPMScheduler.from_pretrained(checkpoint / "scheduler")
    model.eval()
    return _LoadedDiffusersSolver(
        torch=torch,
        model=model,
        scheduler=scheduler,
        device=torch_device,
        dtype=dtype,
    )


def sample_conditional_diffusion_solution_mask(
    *,
    example: ConditionalDiffusionSamplingExample,
    solver: _LoadedDiffusersSolver,
    sampling_steps: int,
    seed: int,
) -> tuple[tuple[bool, ...], ...]:
    return sample_conditional_diffusion_solution_mask_with_stats(
        example=example,
        solver=solver,
        sampling_steps=sampling_steps,
        seed=seed,
    ).mask


def sample_conditional_diffusion_solution_mask_with_stats(
    *,
    example: ConditionalDiffusionSamplingExample,
    solver: _LoadedDiffusersSolver,
    sampling_steps: int,
    seed: int,
) -> SampledSolutionMask:
    trajectory = list(
        _iter_conditional_diffusion_solution_mask_samples(
            example=example,
            solver=solver,
            sampling_steps=sampling_steps,
            seed=seed,
        )
    )
    return trajectory[-1]


def sample_conditional_diffusion_solution_mask_trajectory(
    *,
    example: ConditionalDiffusionSamplingExample,
    solver: _LoadedDiffusersSolver,
    sampling_steps: int,
    seed: int,
) -> list[tuple[tuple[bool, ...], ...]]:
    return list(
        iter_conditional_diffusion_solution_mask_trajectory(
            example=example,
            solver=solver,
            sampling_steps=sampling_steps,
            seed=seed,
        )
    )


def iter_conditional_diffusion_solution_mask_trajectory(
    *,
    example: ConditionalDiffusionSamplingExample,
    solver: _LoadedDiffusersSolver,
    sampling_steps: int,
    seed: int,
) -> Iterator[tuple[tuple[bool, ...], ...]]:
    for sample in _iter_conditional_diffusion_solution_mask_samples(
        example=example,
        solver=solver,
        sampling_steps=sampling_steps,
        seed=seed,
    ):
        yield sample.mask


def _iter_conditional_diffusion_solution_mask_samples(
    *,
    example: ConditionalDiffusionSamplingExample,
    solver: _LoadedDiffusersSolver,
    sampling_steps: int,
    seed: int,
) -> Iterator[SampledSolutionMask]:
    if sampling_steps < 1:
        raise ValueError("Conditional Diffusion Solver sampling steps must be positive")

    torch = solver.torch
    original_rows = len(example.maze_condition)
    original_columns = len(example.maze_condition[0])
    sample_size = (
        _ceil_multiple(original_rows, 8),
        _ceil_multiple(original_columns, 8),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    condition = _sampling_condition_tensor(
        torch=torch,
        example=example,
        device=solver.device,
        dtype=solver.dtype,
        sample_size=sample_size,
    )
    noisy_mask = torch.randn(
        (1, 1, sample_size[0], sample_size[1]),
        generator=generator,
        dtype=solver.dtype,
    ).to(solver.device)

    yield _tensor_to_mask_sample(noisy_mask, original_rows, original_columns)
    solver.scheduler.set_timesteps(sampling_steps, device=solver.device)

    with torch.no_grad():
        for timestep in solver.scheduler.timesteps:
            model_input = torch.cat([noisy_mask, condition], dim=1)
            predicted_noise = solver.model(model_input, timestep).sample
            noisy_mask = solver.scheduler.step(
                predicted_noise,
                timestep,
                noisy_mask,
                generator=generator,
            ).prev_sample
            yield _tensor_to_mask_sample(noisy_mask, original_rows, original_columns)


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
        device=payload.get("device", "cpu"),
        precision=payload.get("precision", "float32"),
    )
    _validate_evaluation_config(config)
    return config


def _retry_finds_valid_solution(
    *,
    example: TrainingExampleArrays,
    solver: _LoadedDiffusersSolver,
    sampling_steps: int,
    retry_count: int,
    seed: int,
) -> bool:
    for attempt in range(retry_count):
        mask = sample_conditional_diffusion_solution_mask(
            example=_sampling_example_from_arrays(example),
            solver=solver,
            sampling_steps=sampling_steps,
            seed=seed + attempt,
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


def _load_evaluation_examples(
    config: EvaluationConfig,
) -> tuple[TrainingExampleArrays, ...]:
    dataset_dir = Path(config.dataset_dir)
    manifest = load_dataset_artifact_manifest(dataset_dir / "manifest.json")
    requested_split = DatasetSplitName(config.split)
    examples: list[TrainingExampleArrays] = []

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
) -> tuple[TrainingExampleArrays, ...]:
    return tuple(
        TrainingExampleArrays(
            maze_condition=_grid(payload["maze_condition"][index]),
            solution_mask=_grid(payload["solution_mask"][index]),
            start_cell=tuple(payload["start_cell"][index]),
            goal_cell=tuple(payload["goal_cell"][index]),
        )
        for index in range(len(payload["maze_condition"]))
    )


def _sampling_example_from_arrays(
    example: TrainingExampleArrays,
) -> ConditionalDiffusionSamplingExample:
    return ConditionalDiffusionSamplingExample(
        maze_condition=example.maze_condition,
        start_cell=example.start_cell,
        goal_cell=example.goal_cell,
    )


def _sampling_condition_tensor(*, torch, example, device, dtype, sample_size):
    arrays = TrainingExampleArrays(
        maze_condition=example.maze_condition,
        solution_mask=(),
        start_cell=example.start_cell,
        goal_cell=example.goal_cell,
    )
    condition = torch.tensor(
        [_condition_channels(arrays)],
        dtype=dtype,
        device=device,
    )
    return _pad_to_sample_size(torch, condition, sample_size)


def _tensor_to_mask_sample(tensor, rows: int, columns: int) -> SampledSolutionMask:
    cropped = tensor.detach().float().cpu()[0, 0, :rows, :columns]
    raw_values = tuple(
        tuple(float(value) for value in row.tolist())
        for row in cropped
    )
    mask = tuple(
        tuple(bool(value >= 0.0) for value in row.tolist())
        for row in cropped
    )
    marked_count = sum(1 for row in mask for value in row if value)
    total_cells = rows * columns
    return SampledSolutionMask(
        mask=mask,
        raw_values=raw_values,
        tensor_stats=SampledTensorStats(
            raw_min=float(cropped.min()),
            raw_max=float(cropped.max()),
            raw_mean=float(cropped.mean()),
            fraction_at_or_above_threshold=marked_count / total_cells,
            marked_count=marked_count,
            total_cells=total_cells,
        ),
    )


def _tensor_to_mask(tensor, rows: int, columns: int) -> tuple[tuple[bool, ...], ...]:
    return _tensor_to_mask_sample(tensor, rows, columns).mask


def _aggregate_sampled_tensor_stats(
    stats: list[SampledTensorStats],
) -> Mapping[str, float | int]:
    if not stats:
        return {
            "raw_min": 0.0,
            "raw_max": 0.0,
            "raw_mean": 0.0,
            "fraction_at_or_above_threshold": 0.0,
            "marked_count_mean": 0.0,
            "total_cells": 0,
        }
    return {
        "raw_min": min(item.raw_min for item in stats),
        "raw_max": max(item.raw_max for item in stats),
        "raw_mean": sum(item.raw_mean for item in stats) / len(stats),
        "fraction_at_or_above_threshold": sum(
            item.fraction_at_or_above_threshold for item in stats
        )
        / len(stats),
        "marked_count_mean": sum(item.marked_count for item in stats) / len(stats),
        "total_cells": sum(item.total_cells for item in stats),
    }


def _endpoint_raw_value_diagnostics(
    *,
    raw_values: tuple[tuple[float, ...], ...],
    label_mask: tuple[tuple[int, ...], ...],
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
) -> EndpointRawValueDiagnostics:
    label_values: list[float] = []
    non_label_values: list[float] = []
    all_values: list[float] = []

    for row_index, row in enumerate(raw_values):
        for column_index, raw_value in enumerate(row):
            value = float(raw_value)
            all_values.append(value)
            if label_mask[row_index][column_index]:
                label_values.append(value)
            else:
                non_label_values.append(value)

    start_value = _raw_value_at_cell(raw_values, start_cell)
    goal_value = _raw_value_at_cell(raw_values, goal_cell)
    return EndpointRawValueDiagnostics(
        start_cell_raw_value=start_value,
        goal_cell_raw_value=goal_value,
        label_cell_raw_mean=_mean(label_values),
        label_cell_raw_min=min(label_values) if label_values else 0.0,
        label_cell_raw_max=max(label_values) if label_values else 0.0,
        non_label_cell_raw_mean=_mean(non_label_values),
        non_label_cell_raw_min=min(non_label_values) if non_label_values else 0.0,
        non_label_cell_raw_max=max(non_label_values) if non_label_values else 0.0,
        start_cell_descending_rank=_descending_rank(all_values, start_value),
        goal_cell_descending_rank=_descending_rank(all_values, goal_value),
        start_cell_percentile=_percentile(all_values, start_value),
        goal_cell_percentile=_percentile(all_values, goal_value),
    )


def _aggregate_endpoint_raw_value_diagnostics(
    diagnostics: list[EndpointRawValueDiagnostics],
) -> Mapping[str, float | int]:
    if not diagnostics:
        return {
            "start_cell_raw_mean": 0.0,
            "start_cell_raw_min": 0.0,
            "start_cell_raw_max": 0.0,
            "goal_cell_raw_mean": 0.0,
            "goal_cell_raw_min": 0.0,
            "goal_cell_raw_max": 0.0,
            "label_cell_raw_mean": 0.0,
            "label_cell_raw_min": 0.0,
            "label_cell_raw_max": 0.0,
            "non_label_cell_raw_mean": 0.0,
            "non_label_cell_raw_min": 0.0,
            "non_label_cell_raw_max": 0.0,
            "start_cell_descending_rank_mean": 0.0,
            "goal_cell_descending_rank_mean": 0.0,
            "start_cell_percentile_mean": 0.0,
            "goal_cell_percentile_mean": 0.0,
        }

    return {
        "start_cell_raw_mean": _mean(
            [item.start_cell_raw_value for item in diagnostics]
        ),
        "start_cell_raw_min": min(item.start_cell_raw_value for item in diagnostics),
        "start_cell_raw_max": max(item.start_cell_raw_value for item in diagnostics),
        "goal_cell_raw_mean": _mean(
            [item.goal_cell_raw_value for item in diagnostics]
        ),
        "goal_cell_raw_min": min(item.goal_cell_raw_value for item in diagnostics),
        "goal_cell_raw_max": max(item.goal_cell_raw_value for item in diagnostics),
        "label_cell_raw_mean": _mean(
            [item.label_cell_raw_mean for item in diagnostics]
        ),
        "label_cell_raw_min": min(item.label_cell_raw_min for item in diagnostics),
        "label_cell_raw_max": max(item.label_cell_raw_max for item in diagnostics),
        "non_label_cell_raw_mean": _mean(
            [item.non_label_cell_raw_mean for item in diagnostics]
        ),
        "non_label_cell_raw_min": min(
            item.non_label_cell_raw_min for item in diagnostics
        ),
        "non_label_cell_raw_max": max(
            item.non_label_cell_raw_max for item in diagnostics
        ),
        "start_cell_descending_rank_mean": _mean(
            [item.start_cell_descending_rank for item in diagnostics]
        ),
        "goal_cell_descending_rank_mean": _mean(
            [item.goal_cell_descending_rank for item in diagnostics]
        ),
        "start_cell_percentile_mean": _mean(
            [item.start_cell_percentile for item in diagnostics]
        ),
        "goal_cell_percentile_mean": _mean(
            [item.goal_cell_percentile for item in diagnostics]
        ),
    }


def _endpoint_raw_value_example_payload(
    *,
    example_index: int,
    validation_failure_reason: str | None,
    start_cell_included: bool,
    goal_cell_included: bool,
    diagnostics: EndpointRawValueDiagnostics,
    tensor_stats_marked_count: int,
    tensor_stats_fraction_at_or_above_threshold: float,
) -> Mapping[str, float | int | bool | str | None]:
    return {
        "example_index": example_index,
        "validation_failure_reason": validation_failure_reason,
        "start_cell_included": start_cell_included,
        "goal_cell_included": goal_cell_included,
        "start_cell_raw_value": diagnostics.start_cell_raw_value,
        "goal_cell_raw_value": diagnostics.goal_cell_raw_value,
        "start_cell_descending_rank": diagnostics.start_cell_descending_rank,
        "goal_cell_descending_rank": diagnostics.goal_cell_descending_rank,
        "start_cell_percentile": diagnostics.start_cell_percentile,
        "goal_cell_percentile": diagnostics.goal_cell_percentile,
        "label_cell_raw_mean": diagnostics.label_cell_raw_mean,
        "non_label_cell_raw_mean": diagnostics.non_label_cell_raw_mean,
        "marked_count": tensor_stats_marked_count,
        "fraction_at_or_above_threshold": (
            tensor_stats_fraction_at_or_above_threshold
        ),
    }


def _raw_value_at_cell(
    raw_values: tuple[tuple[float, ...], ...], cell: tuple[int, int]
) -> float:
    row, column = cell
    return float(raw_values[row][column])


def _descending_rank(values: list[float], value: float) -> int:
    return 1 + sum(1 for candidate in values if candidate > value)


def _percentile(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    return sum(1 for candidate in values if candidate <= value) / len(values)


def _mean(values: list[float | int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _threshold_calibration_payloads(
    examples: list[tuple[TrainingExampleArrays, SampledSolutionMask]],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        _threshold_calibration_payload(
            examples=examples,
            threshold=threshold,
        )
        for threshold in _THRESHOLD_CALIBRATION_CANDIDATES
    )


def _threshold_calibration_payload(
    *,
    examples: list[tuple[TrainingExampleArrays, SampledSolutionMask]],
    threshold: float,
) -> Mapping[str, Any]:
    valid_count = 0
    failure_reasons: Counter[str] = Counter()
    mask_overlaps: list[float] = []
    marked_counts: list[int] = []
    start_cell_included_count = 0
    goal_cell_included_count = 0
    both_endpoints_included_count = 0

    for example, sample in examples:
        mask = _mask_from_raw_values(sample.raw_values, threshold)
        rendered_start_cell = _rendered_cell(example.start_cell)
        rendered_goal_cell = _rendered_cell(example.goal_cell)
        result = validate_solution_mask(
            grid_maze=_bool_grid(example.maze_condition),
            solution_mask=mask,
            start_cell=rendered_start_cell,
            goal_cell=rendered_goal_cell,
        )
        if result.valid:
            valid_count += 1
        else:
            failure_reasons[str(result.reason)] += 1
        mask_overlaps.append(
            _mask_overlap(proposed_mask=mask, label_mask=example.solution_mask)
        )
        marked_counts.append(sum(1 for row in mask for value in row if value))
        start_cell_included = _mask_includes_cell(mask, rendered_start_cell)
        goal_cell_included = _mask_includes_cell(mask, rendered_goal_cell)
        if start_cell_included:
            start_cell_included_count += 1
        if goal_cell_included:
            goal_cell_included_count += 1
        if start_cell_included and goal_cell_included:
            both_endpoints_included_count += 1

    evaluated_examples = len(examples)
    total_cells = sum(
        len(sample.raw_values) * len(sample.raw_values[0])
        for _, sample in examples
    )
    marked_count = sum(marked_counts)
    return {
        "threshold": threshold,
        "valid_count": valid_count,
        "evaluated_examples": evaluated_examples,
        "valid_solution_rate": (
            valid_count / evaluated_examples if evaluated_examples else 0.0
        ),
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "mask_overlap": _mean(mask_overlaps),
        "marked_count_mean": _mean(marked_counts),
        "fraction_at_or_above_threshold": (
            marked_count / total_cells if total_cells else 0.0
        ),
        "start_cell_inclusion_rate": (
            start_cell_included_count / evaluated_examples
            if evaluated_examples
            else 0.0
        ),
        "goal_cell_inclusion_rate": (
            goal_cell_included_count / evaluated_examples
            if evaluated_examples
            else 0.0
        ),
        "both_endpoints_inclusion_rate": (
            both_endpoints_included_count / evaluated_examples
            if evaluated_examples
            else 0.0
        ),
    }


def _mask_from_raw_values(
    raw_values: tuple[tuple[float, ...], ...], threshold: float
) -> tuple[tuple[bool, ...], ...]:
    return tuple(
        tuple(value >= threshold for value in row)
        for row in raw_values
    )


def _mask_overlap(
    *,
    proposed_mask: tuple[tuple[bool, ...], ...],
    label_mask: tuple[tuple[int, ...], ...],
) -> float:
    return _mask_overlap_excluding_cells(
        proposed_mask=proposed_mask,
        label_mask=label_mask,
        excluded_cells=set(),
    )


def _mask_overlap_excluding_cells(
    *,
    proposed_mask: tuple[tuple[bool, ...], ...],
    label_mask: tuple[tuple[int, ...], ...],
    excluded_cells: set[tuple[int, int]],
) -> float:
    proposed_cells = _marked_cells(proposed_mask)
    label_cells = _marked_cells(label_mask)
    proposed_cells -= excluded_cells
    label_cells -= excluded_cells
    union = proposed_cells | label_cells
    if not union:
        return 1.0
    return len(proposed_cells & label_cells) / len(union)


def _mask_includes_cell(
    mask: tuple[tuple[bool, ...], ...], cell: tuple[int, int]
) -> bool:
    row, column = cell
    if row < 0 or column < 0:
        return False
    if row >= len(mask) or column >= len(mask[row]):
        return False
    return bool(mask[row][column])


def _endpoint_inclusion_payload(
    *,
    evaluated_examples: int,
    start_cell_included_count: int,
    goal_cell_included_count: int,
    both_endpoints_included_count: int,
    body_mask_overlaps: list[float],
) -> Mapping[str, float | int]:
    if evaluated_examples == 0:
        return {
            "start_cell_included_count": 0,
            "goal_cell_included_count": 0,
            "both_endpoints_included_count": 0,
            "start_cell_inclusion_rate": 0.0,
            "goal_cell_inclusion_rate": 0.0,
            "both_endpoints_inclusion_rate": 0.0,
            "mask_overlap_excluding_endpoints": 0.0,
        }
    return {
        "start_cell_included_count": start_cell_included_count,
        "goal_cell_included_count": goal_cell_included_count,
        "both_endpoints_included_count": both_endpoints_included_count,
        "start_cell_inclusion_rate": start_cell_included_count / evaluated_examples,
        "goal_cell_inclusion_rate": goal_cell_included_count / evaluated_examples,
        "both_endpoints_inclusion_rate": (
            both_endpoints_included_count / evaluated_examples
        ),
        "mask_overlap_excluding_endpoints": (
            sum(body_mask_overlaps) / len(body_mask_overlaps)
            if body_mask_overlaps
            else 0.0
        ),
    }


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
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("Evaluation device must be cpu or cuda")
    if config.precision not in {"float32", "float16", "bfloat16"}:
        raise ValueError("Evaluation precision must be float32, float16, or bfloat16")


def _validate_checkpoint_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("model_type") != DIFFUSERS_MODEL_TYPE:
        raise ValueError("Checkpoint is not a Dreamaze Diffusers Conditional Diffusion Solver")


def _bool_grid(rows: tuple[tuple[int, ...], ...]) -> tuple[tuple[bool, ...], ...]:
    return tuple(tuple(bool(value) for value in row) for row in rows)


def _grid(rows: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in rows)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for file_path in files:
        digest.update(str(file_path.relative_to(path) if path.is_dir() else file_path.name).encode())
        with file_path.open("rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
