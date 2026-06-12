import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Mapping, Sequence

from dreamaze.dataset import (
    BoolGrid,
    Cell,
    MazeFamily,
    TrainingExample,
    TrainingExampleConfig,
    build_training_example,
)
from dreamaze.evaluation import (
    ConditionalDiffusionSamplingExample,
    sample_conditional_diffusion_solution_mask,
)
from dreamaze.validation import SolutionValidationResult, validate_solution_mask


@dataclass(frozen=True)
class ProofDemoConfig:
    checkpoint_path: str | Path | None = None
    width: int = 16
    height: int = 16
    maze_seed: int = 0
    maze_family: MazeFamily = MazeFamily.KRUSKAL
    sampling_steps: int = 8
    retry_count: int = 0
    seed: int = 0
    debug_reveal: bool = False


@dataclass(frozen=True)
class ProofDemoRetryResult:
    valid: bool
    attempts: int
    excluded_from_official_score: bool = True


@dataclass(frozen=True)
class ProofDemoResult:
    rendered_maze_svg: str
    generated_solution_mask_svg: str
    start_cell: Cell
    goal_cell: Cell
    validation_status: str
    validation_reason: str | None
    single_sample_success: SolutionValidationResult
    official_score: str = "Single-Sample Success"
    retry_success: ProofDemoRetryResult | None = None
    training_label_svg: str | None = None
    difference_svg: str | None = None


def run_proof_demo(config: ProofDemoConfig) -> ProofDemoResult:
    _validate_proof_demo_config(config)
    example = build_training_example(
        seed=config.maze_seed,
        config=TrainingExampleConfig(
            width=config.width,
            height=config.height,
            maze_family=config.maze_family,
            split="demo",
        ),
    )
    weights = _load_checkpoint_weights(config.checkpoint_path)
    generated_mask = sample_conditional_diffusion_solution_mask(
        example=_sampling_example_from_training_example(example),
        weights=weights,
        sampling_steps=config.sampling_steps,
        rng=Random(config.seed),
    )
    validation = validate_solution_mask(
        grid_maze=example.maze_condition.rendered_maze,
        solution_mask=generated_mask,
        start_cell=example.rendered_start_cell,
        goal_cell=example.rendered_goal_cell,
    )
    retry_success = _run_sampling_retries(
        example=example,
        weights=weights,
        sampling_steps=config.sampling_steps,
        retry_count=config.retry_count,
        seed=config.seed,
    )

    return ProofDemoResult(
        rendered_maze_svg=_maze_svg(
            rendered_maze=example.maze_condition.rendered_maze,
            start_cell=example.rendered_start_cell,
            goal_cell=example.rendered_goal_cell,
        ),
        generated_solution_mask_svg=_mask_svg(
            rendered_maze=example.maze_condition.rendered_maze,
            solution_mask=generated_mask,
            start_cell=example.rendered_start_cell,
            goal_cell=example.rendered_goal_cell,
        ),
        start_cell=example.start_cell,
        goal_cell=example.goal_cell,
        validation_status="Valid Solution" if validation.valid else "Invalid Solution",
        validation_reason=None if validation.reason is None else validation.reason.value,
        single_sample_success=validation,
        retry_success=retry_success,
        training_label_svg=(
            _mask_svg(
                rendered_maze=example.maze_condition.rendered_maze,
                solution_mask=example.training_label,
                start_cell=example.rendered_start_cell,
                goal_cell=example.rendered_goal_cell,
            )
            if config.debug_reveal
            else None
        ),
        difference_svg=(
            _difference_svg(
                rendered_maze=example.maze_condition.rendered_maze,
                generated_mask=generated_mask,
                label_mask=example.training_label,
                start_cell=example.rendered_start_cell,
                goal_cell=example.rendered_goal_cell,
            )
            if config.debug_reveal
            else None
        ),
    )


def _load_checkpoint_weights(checkpoint_path: str | Path | None) -> Mapping[str, float]:
    if checkpoint_path is None:
        return _tiny_fixture_weights()

    payload = json.loads(Path(checkpoint_path).read_text())
    if payload.get("model_type") != "custom_conditional_diffusion_solver":
        raise ValueError("Checkpoint is not a custom Conditional Diffusion Solver")
    return payload["weights"]


def _run_sampling_retries(
    *,
    example: TrainingExample,
    weights: Mapping[str, float],
    sampling_steps: int,
    retry_count: int,
    seed: int,
) -> ProofDemoRetryResult | None:
    if retry_count == 0:
        return None

    rng = Random(seed + 1)
    sampling_example = _sampling_example_from_training_example(example)
    for attempt in range(1, retry_count + 1):
        retry_mask = sample_conditional_diffusion_solution_mask(
            example=sampling_example,
            weights=weights,
            sampling_steps=sampling_steps,
            rng=rng,
        )
        retry_validation = validate_solution_mask(
            grid_maze=example.maze_condition.rendered_maze,
            solution_mask=retry_mask,
            start_cell=example.rendered_start_cell,
            goal_cell=example.rendered_goal_cell,
        )
        if retry_validation.valid:
            return ProofDemoRetryResult(valid=True, attempts=attempt)
    return ProofDemoRetryResult(valid=False, attempts=retry_count)


def _tiny_fixture_weights() -> Mapping[str, float]:
    return {
        "bias": -2.0,
        "maze_open": 1.5,
        "start": 1.0,
        "goal": 1.0,
        "noisy_mask": 0.5,
        "timestep": 0.25,
    }


def _sampling_example_from_training_example(
    example: TrainingExample,
) -> ConditionalDiffusionSamplingExample:
    return ConditionalDiffusionSamplingExample(
        maze_condition=_int_grid(example.maze_condition.rendered_maze),
        start_cell=example.start_cell,
        goal_cell=example.goal_cell,
    )


def _maze_svg(*, rendered_maze: BoolGrid, start_cell: Cell, goal_cell: Cell) -> str:
    return _grid_svg(
        rendered_maze=rendered_maze,
        solution_mask=None,
        label_mask=None,
        start_cell=start_cell,
        goal_cell=goal_cell,
    )


def _mask_svg(
    *,
    rendered_maze: BoolGrid,
    solution_mask: Sequence[Sequence[bool]],
    start_cell: Cell,
    goal_cell: Cell,
) -> str:
    return _grid_svg(
        rendered_maze=rendered_maze,
        solution_mask=solution_mask,
        label_mask=None,
        start_cell=start_cell,
        goal_cell=goal_cell,
    )


def _grid_svg(
    *,
    rendered_maze: BoolGrid,
    solution_mask: Sequence[Sequence[bool]] | None,
    label_mask: Sequence[Sequence[bool]] | None,
    start_cell: Cell,
    goal_cell: Cell,
) -> str:
    cell_size = 12
    rows = len(rendered_maze)
    columns = len(rendered_maze[0])
    parts = [
        (
            f'<svg viewBox="0 0 {columns * cell_size} {rows * cell_size}" '
            'xmlns="http://www.w3.org/2000/svg" role="img">'
        )
    ]

    for row_index, row in enumerate(rendered_maze):
        for column_index, is_open in enumerate(row):
            fill = "#f8fafc" if is_open else "#111827"
            if solution_mask is not None and solution_mask[row_index][column_index]:
                fill = "#2563eb"
            if label_mask is not None and label_mask[row_index][column_index]:
                fill = "#f59e0b"
            if (row_index, column_index) == start_cell:
                fill = "#16a34a"
            elif (row_index, column_index) == goal_cell:
                fill = "#dc2626"
            parts.append(
                (
                    f'<rect x="{column_index * cell_size}" '
                    f'y="{row_index * cell_size}" width="{cell_size}" '
                    f'height="{cell_size}" fill="{fill}"/>'
                )
            )

    parts.append("</svg>")
    return "".join(parts)


def _difference_svg(
    *,
    rendered_maze: BoolGrid,
    generated_mask: Sequence[Sequence[bool]],
    label_mask: Sequence[Sequence[bool]],
    start_cell: Cell,
    goal_cell: Cell,
) -> str:
    cell_size = 12
    rows = len(rendered_maze)
    columns = len(rendered_maze[0])
    parts = [
        (
            f'<svg viewBox="0 0 {columns * cell_size} {rows * cell_size}" '
            'xmlns="http://www.w3.org/2000/svg" role="img">'
        )
    ]
    for row_index, row in enumerate(rendered_maze):
        for column_index, is_open in enumerate(row):
            generated = bool(generated_mask[row_index][column_index])
            label = bool(label_mask[row_index][column_index])
            fill = "#111827" if not is_open else "#f8fafc"
            if generated and label:
                fill = "#16a34a"
            elif generated and not label:
                fill = "#dc2626"
            elif label and not generated:
                fill = "#f59e0b"
            if (row_index, column_index) == start_cell:
                fill = "#14532d"
            elif (row_index, column_index) == goal_cell:
                fill = "#7f1d1d"
            parts.append(
                (
                    f'<rect x="{column_index * cell_size}" '
                    f'y="{row_index * cell_size}" width="{cell_size}" '
                    f'height="{cell_size}" fill="{fill}"/>'
                )
            )
    parts.append("</svg>")
    return "".join(parts)


def _int_grid(grid: Sequence[Sequence[bool]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(1 if value else 0 for value in row) for row in grid)


def _validate_proof_demo_config(config: ProofDemoConfig) -> None:
    if config.width < 2 or config.height < 2:
        raise ValueError("Proof Demo maze dimensions must be at least 2 by 2")
    if config.sampling_steps < 1:
        raise ValueError("Proof Demo sampling steps must be positive")
    if config.retry_count < 0:
        raise ValueError("Proof Demo retry count cannot be negative")
