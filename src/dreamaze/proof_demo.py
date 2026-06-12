from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
    load_diffusers_solver_checkpoint,
    sample_conditional_diffusion_solution_mask,
    sample_conditional_diffusion_solution_mask_trajectory,
)
from dreamaze.validation import SolutionValidationResult, validate_solution_mask


@dataclass(frozen=True)
class ProofDemoConfig:
    checkpoint_path: str | Path
    width: int = 16
    height: int = 16
    maze_seed: int = 0
    maze_family: MazeFamily = MazeFamily.KRUSKAL
    sampling_steps: int = 8
    retry_count: int = 0
    seed: int = 0
    debug_reveal: bool = False
    capture_trajectory: bool = False


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
    rendered_start_cell: Cell
    rendered_goal_cell: Cell
    rendered_maze: BoolGrid
    validation_status: str
    validation_reason: str | None
    single_sample_success: SolutionValidationResult
    official_score: str = "Single-Sample Success"
    retry_success: ProofDemoRetryResult | None = None
    training_label_svg: str | None = None
    difference_svg: str | None = None
    diffusion_intermediates: list[tuple[tuple[bool, ...], ...]] | None = None


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
    solver = load_diffusers_solver_checkpoint(checkpoint_path=config.checkpoint_path)
    sampling_example = _sampling_example_from_training_example(example)
    if config.capture_trajectory:
        intermediates = sample_conditional_diffusion_solution_mask_trajectory(
            example=sampling_example,
            solver=solver,
            sampling_steps=config.sampling_steps,
            seed=config.seed,
        )
        generated_mask = intermediates[-1]
    else:
        generated_mask = sample_conditional_diffusion_solution_mask(
            example=sampling_example,
            solver=solver,
            sampling_steps=config.sampling_steps,
            seed=config.seed,
        )
        intermediates = None

    validation = validate_solution_mask(
        grid_maze=example.maze_condition.rendered_maze,
        solution_mask=generated_mask,
        start_cell=example.rendered_start_cell,
        goal_cell=example.rendered_goal_cell,
    )
    retry_success = _run_sampling_retries(
        example=example,
        solver=solver,
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
        rendered_start_cell=example.rendered_start_cell,
        rendered_goal_cell=example.rendered_goal_cell,
        rendered_maze=example.maze_condition.rendered_maze,
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
        diffusion_intermediates=intermediates,
    )


def _run_sampling_retries(
    *,
    example: TrainingExample,
    solver,
    sampling_steps: int,
    retry_count: int,
    seed: int,
) -> ProofDemoRetryResult | None:
    if retry_count == 0:
        return None

    sampling_example = _sampling_example_from_training_example(example)
    for attempt in range(1, retry_count + 1):
        retry_mask = sample_conditional_diffusion_solution_mask(
            example=sampling_example,
            solver=solver,
            sampling_steps=sampling_steps,
            seed=seed + attempt,
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


# --- Real-time diffusion visualization (HTML/CSS/JS player for Proof Demo) ---


def build_diffusion_viz_html(
    result: ProofDemoResult,
    *,
    maze_family: MazeFamily | str = "demo",
    maze_seed: int = 0,
    sampling_steps_used: int = 8,
) -> str:
    """Build a self-contained intuitive HTML/CSS/JS player showing the maze
    and the Conditional Diffusion Solver's denoising trajectory in real time.

    The player auto-plays the refinement steps at a human-visible pace after
    the server-side sampling (which is the actual Runtime Solving). No config
    exposed to the end user.
    """
    if result.diffusion_intermediates and getattr(result, "rendered_maze", None) is not None:
        fam = maze_family
        if not isinstance(fam, MazeFamily):
            try:
                fam = MazeFamily(str(fam))
            except Exception:
                fam = MazeFamily.KRUSKAL
        return _build_automated_player_html(
            rendered_maze=result.rendered_maze,
            rendered_start_cell=result.rendered_start_cell,
            rendered_goal_cell=result.rendered_goal_cell,
            intermediates=result.diffusion_intermediates,
            validation_status=result.validation_status,
            validation_reason=result.validation_reason,
            maze_family=fam,
            maze_seed=maze_seed,
            sampling_steps_used=sampling_steps_used,
        )

    # Fallback for results without trajectory captured
    return (
        f'<div class="dreamaze-fallback">'
        f'{result.rendered_maze_svg}'
        f'{result.generated_solution_mask_svg}'
        f'<div class="verdict" style="font-weight:600;margin:4px 0">{result.validation_status}</div>'
        f'{"<div style=\"color:#b91c1c;font-size:0.85em\">Reason: "+result.validation_reason+"</div>" if result.validation_reason else ""}'
        f"</div>"
    )

def _build_automated_player_html(
    *,
    rendered_maze: BoolGrid,
    rendered_start_cell: Cell,
    rendered_goal_cell: Cell,
    intermediates: list[tuple[tuple[bool, ...], ...]],
    validation_status: str,
    validation_reason: str | None,
    maze_family: MazeFamily,
    maze_seed: int,
    sampling_steps_used: int,
) -> str:
    """Produce the full custom HTML/CSS/JS for the intuitive one-button demo result.

    Uses embedded JS to animate the diffusion trajectory client-side after the
    (instant) server sampling. This gives a clear "watch the model solve" experience
    at a comfortable visual speed without prolonging GPU function time.
    """
    import json as _json

    rows = len(rendered_maze)
    cols = len(rendered_maze[0]) if rows else 0
    cell = 9
    width_px = cols * cell
    height_px = rows * cell

    # Serialize data for JS
    maze_open = [[1 if bool(cell) else 0 for cell in row] for row in rendered_maze]
    frames = [
        [[1 if bool(cell) else 0 for cell in row] for row in m]
        for m in intermediates
    ]
    vid = f"dmz{int(_json.dumps([maze_seed, sampling_steps_used, len(intermediates)]).__hash__()) & 0xfffffff}"

    # Build initial SVG skeleton (structure + S/G only). JS will mutate solution fills live.
    svg_parts: list[str] = [
        f'<svg id="{vid}-grid" viewBox="0 0 {width_px} {height_px}" '
        f'width="{width_px}" height="{height_px}" '
        'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Maze with live diffusion solution animation">'
    ]
    for ri, row in enumerate(rendered_maze):
        for ci, is_open in enumerate(row):
            r = rendered_start_cell
            g = rendered_goal_cell
            is_s = (ri, ci) == r
            is_g = (ri, ci) == g
            if is_s:
                fill = "#16a34a"
            elif is_g:
                fill = "#dc2626"
            elif not is_open:
                fill = "#111827"
            else:
                fill = "#f8fafc"
            svg_parts.append(
                f'<rect id="{vid}-c-{ri}-{ci}" '
                f'x="{ci * cell}" y="{ri * cell}" '
                f'width="{cell}" height="{cell}" fill="{fill}" />'
            )
    svg_parts.append("</svg>")
    base_svg = "".join(svg_parts)

    # Data blobs (avoid too large, but 33x33 x 17 ~ 18k numbers, fine as json)
    maze_json = _json.dumps(maze_open)
    frames_json = _json.dumps(frames)
    s_json = _json.dumps(list(rendered_start_cell))
    g_json = _json.dumps(list(rendered_goal_cell))
    family_str = maze_family.value if hasattr(maze_family, "value") else str(maze_family)
    reason_html = (
        f'<div class="reason" style="color:#b91c1c;font-size:0.85em;margin-top:2px">'
        f'Validation reason: {validation_reason}</div>'
        if validation_reason
        else ""
    )
    verdict_class = "valid" if "Valid" in validation_status else "invalid"
    verdict_color = "#166534" if "Valid" in validation_status else "#991b1b"

    html = f"""<div id="{vid}-wrap" style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:640px;margin:0 auto">
  <style>
    #{vid}-wrap .dm-header {{font-size:13px;opacity:.75;margin-bottom:6px;letter-spacing:.3px}}
    #{vid}-wrap .dm-grid-wrap {{border:1px solid #e5e7eb;border-radius:8px;padding:8px;background:#fff;display:inline-block;box-shadow:0 1px 2px rgb(0 0 0 / 0.04)}}
    #{vid}-wrap .dm-controls {{display:flex;gap:8px;align-items:center;margin:8px 0 4px;font-size:12px}}
    #{vid}-wrap button.dm-replay {{font-size:11px;padding:2px 8px;border-radius:4px;border:1px solid #cbd5e1;background:#f8fafc;cursor:pointer}}
    #{vid}-wrap button.dm-replay:hover {{background:#f1f5f9}}
    #{vid}-wrap .dm-verdict {{display:inline-block;padding:2px 10px;border-radius:9999px;font-weight:600;font-size:0.95em;margin-top:4px}}
    #{vid}-wrap .dm-verdict.valid {{background:#dcfce7;color:#166534}}
    #{vid}-wrap .dm-verdict.invalid {{background:#fee2e2;color:#991b1b}}
    #{vid}-wrap .dm-legend {{font-size:10px;opacity:.6;margin-top:6px;display:flex;gap:12px;flex-wrap:wrap}}
    #{vid}-wrap .dm-legend span {{display:inline-flex;align-items:center;gap:4px}}
    #{vid}-wrap .dm-legend i {{width:10px;height:10px;border:1px solid #cbd5e1}}
  </style>
  <div class="dm-header">Maze (16\times16 Grid) + Conditional Diffusion Solver trajectory — {family_str} seed {maze_seed}</div>
  <div class="dm-grid-wrap">{base_svg}</div>
  <div class="dm-controls">
    <span>Refining path: <strong id="{vid}-step">0</strong> / {len(intermediates)-1}</span>
    <button class="dm-replay" onclick="window.__dmzReplay_{vid}()">⟳ Replay solving steps</button>
  </div>
  <div class="dm-verdict {verdict_class}" style="background:{verdict_color}22;color:{verdict_color}">{validation_status}</div>
  {reason_html}
  <div class="dm-legend">
    <span><i style="background:#16a34a"></i> start</span>
    <span><i style="background:#dc2626"></i> goal</span>
    <span><i style="background:#2563eb"></i> model path claim (evolves)</span>
    <span><i style="background:#f8fafc;border-color:#111827"></i> open passage</span>
    <span><i style="background:#111827"></i> wall</span>
  </div>
  <div style="font-size:9px;opacity:.5;margin-top:4px">Single-Sample Success (official). The diffusion model proposes every cell of the Solution Mask itself.</div>
<script>
(function() {{
  const VID = '{vid}';
  const FRAMES = {frames_json};
  const MAZE = {maze_json};
  const START = {s_json};
  const GOAL = {g_json};
  const TOTAL = FRAMES.length - 1;
  let step = 0;
  let timer = null;
  const STEP_MS = 210; // visible real-time pace for the denoising process

  function $(id) {{ return document.getElementById(id); }}
  function rect(r, c) {{ return $('{vid}-c-' + r + '-' + c); }}

  function apply(s) {{
    const el = $('{vid}-step');
    if (el) el.textContent = String(s);
    const frame = FRAMES[s] || FRAMES[FRAMES.length-1];
    const R = MAZE.length, C = MAZE[0].length;
    for (let r = 0; r < R; r++) {{
      for (let c = 0; c < C; c++) {{
        const node = rect(r, c);
        if (!node) continue;
        const isS = (r === START[0] && c === START[1]);
        const isG = (r === GOAL[0] && c === GOAL[1]);
        const isSol = !!(frame && frame[r] && frame[r][c]);
        const isOpen = !!(MAZE[r] && MAZE[r][c]);
        let fill;
        if (isS) fill = '#16a34a';
        else if (isG) fill = '#dc2626';
        else if (isSol) fill = '#2563eb';
        else if (isOpen) fill = '#f8fafc';
        else fill = '#111827';
        node.setAttribute('fill', fill);
      }}
    }}
  }}

  function playNext() {{
    if (step >= TOTAL) return;
    step += 1;
    apply(step);
    if (step < TOTAL) {{
      timer = setTimeout(playNext, STEP_MS);
    }}
  }}

  function startAuto() {{
    step = 0;
    apply(0);
    if (timer) clearTimeout(timer);
    timer = setTimeout(playNext, 280);
  }}

  window.__dmzReplay_{vid} = function() {{
    if (timer) clearTimeout(timer);
    startAuto();
  }};

  // Boot
  apply(0);
  // Auto-start the diffusion "thinking" animation shortly after mount for immediate "real time" feel
  setTimeout(startAuto, 120);
}})();
</script>
</div>"""
    return html
