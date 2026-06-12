import os
from pathlib import Path
import sys
from typing import Any

_SPACE_SRC_PATH = Path(__file__).resolve().parent / "src"
if _SPACE_SRC_PATH.exists():
    sys.path.insert(0, str(_SPACE_SRC_PATH))

try:
    import spaces
except ImportError:
    class _LocalSpaces:
        @staticmethod
        def GPU(*_args: Any, **_kwargs: Any) -> Any:
            def decorator(fn: Any) -> Any:
                return fn

            return decorator

    spaces = _LocalSpaces()

from dreamaze.dataset import MazeFamily
from dreamaze.proof_demo import ProofDemoConfig, run_proof_demo


def run_demo(
    *,
    maze_family: str,
    maze_seed: int,
    sampling_steps: int,
    retry_count: int,
    debug_reveal: bool,
) -> dict[str, str]:
    checkpoint_path = os.environ.get("DREAMAZE_CHECKPOINT_PATH")
    result = run_proof_demo(
        ProofDemoConfig(
            checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
            maze_family=MazeFamily(maze_family),
            maze_seed=int(maze_seed),
            sampling_steps=int(sampling_steps),
            retry_count=int(retry_count),
            debug_reveal=debug_reveal,
        )
    )
    reason = result.validation_reason or "none"
    retry_summary = "Sampling Retry: disabled"
    if result.retry_success is not None:
        retry_state = "valid" if result.retry_success.valid else "invalid"
        retry_summary = (
            f"Sampling Retry: {retry_state} after "
            f"{result.retry_success.attempts} attempt(s), excluded from official score"
        )

    return {
        "rendered_maze": result.rendered_maze_svg,
        "generated_solution_mask": result.generated_solution_mask_svg,
        "validation_status": result.validation_status,
        "validation_reason": reason,
        "sampling_summary": f"{result.official_score}: {result.validation_status}. {retry_summary}.",
        "training_label": result.training_label_svg or "",
        "difference": result.difference_svg or "",
    }


def _run_demo_for_gradio(
    maze_family: str,
    maze_seed: int,
    sampling_steps: int,
    retry_count: int,
    debug_reveal: bool,
) -> tuple[str, str, str, str, str, str, str]:
    outputs = run_demo(
        maze_family=maze_family,
        maze_seed=maze_seed,
        sampling_steps=sampling_steps,
        retry_count=retry_count,
        debug_reveal=debug_reveal,
    )
    return (
        outputs["rendered_maze"],
        outputs["generated_solution_mask"],
        outputs["validation_status"],
        outputs["validation_reason"],
        outputs["sampling_summary"],
        outputs["training_label"],
        outputs["difference"],
    )


_run_demo_for_gradio = spaces.GPU(duration=30)(_run_demo_for_gradio)


def build_app() -> Any:
    import gradio as gr

    with gr.Blocks(title="Dreamaze Proof Demo") as demo:
        gr.Markdown("# Dreamaze Proof Demo")
        with gr.Row():
            with gr.Column(scale=1):
                maze_family = gr.Dropdown(
                    choices=[MazeFamily.KRUSKAL.value, MazeFamily.WILSON.value],
                    value=MazeFamily.KRUSKAL.value,
                    label="Maze Family",
                )
                maze_seed = gr.Number(value=0, precision=0, label="Maze Seed")
                sampling_steps = gr.Slider(
                    minimum=1,
                    maximum=32,
                    value=8,
                    step=1,
                    label="Sampling Steps",
                )
                retry_count = gr.Slider(
                    minimum=0,
                    maximum=5,
                    value=0,
                    step=1,
                    label="Sampling Retry Count",
                )
                debug_reveal = gr.Checkbox(value=False, label="Debug Reveal")
                run_button = gr.Button("Run Solver", variant="primary")
            with gr.Column(scale=2):
                validation_status = gr.Textbox(
                    label="Valid Solution Status", interactive=False
                )
                validation_reason = gr.Textbox(
                    label="Validation Reason", interactive=False
                )
                sampling_summary = gr.Textbox(
                    label="Sampling Summary", interactive=False
                )

        with gr.Row():
            rendered_maze = gr.HTML(label="Rendered Maze")
            generated_solution_mask = gr.HTML(label="Generated Solution Mask")
        with gr.Row():
            training_label = gr.HTML(label="Training Label")
            difference = gr.HTML(label="Debug Difference")

        run_button.click(
            fn=_run_demo_for_gradio,
            inputs=[
                maze_family,
                maze_seed,
                sampling_steps,
                retry_count,
                debug_reveal,
            ],
            outputs=[
                rendered_maze,
                generated_solution_mask,
                validation_status,
                validation_reason,
                sampling_summary,
                training_label,
                difference,
            ],
            api_name="run_solver",
        )
        demo.load(
            fn=_run_demo_for_gradio,
            inputs=[
                maze_family,
                maze_seed,
                sampling_steps,
                retry_count,
                debug_reveal,
            ],
            outputs=[
                rendered_maze,
                generated_solution_mask,
                validation_status,
                validation_reason,
                sampling_summary,
                training_label,
                difference,
            ],
        )
    return demo


if __name__ == "__main__":
    build_app().launch()
