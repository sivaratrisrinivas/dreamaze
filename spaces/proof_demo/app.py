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

import random as _random
import time

from dreamaze.dataset import MazeFamily
from dreamaze.proof_demo import ProofDemoConfig, build_diffusion_viz_html, run_proof_demo


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


def _safe_gpu_decorate(fn, duration=30):
    try:
        return spaces.GPU(duration=duration)(fn)
    except Exception:
        # In test / local envs the spaces stub or package may not provide GPU.
        # The plain fn is still callable for the python-level run_demo / run_automated_demo.
        return fn


_run_demo_for_gradio = _safe_gpu_decorate(_run_demo_for_gradio)


# --- New automated path: zero config, one button, real-time diffusion viz ---


def run_automated_demo() -> str:
    """Pick a fresh Grid Maze, run the Conditional Diffusion Solver (single sample),
    capture the full denoising trajectory, validate strictly, and return a
    self-contained HTML/CSS/JS player that animates the solving steps live in
    the browser for an intuitive "watch it think" experience.
    """
    checkpoint_path = os.environ.get("DREAMAZE_CHECKPOINT_PATH")
    ckpt = Path(checkpoint_path) if checkpoint_path else None

    # Always automate: fresh maze + fixed high-quality sampling steps for the viz.
    # Using time + random gives a different Training Example (and stochastic sample)
    # on every play / page load. This demonstrates genuine Runtime Solving.
    tseed = int(time.time() * 1000) % 100000
    rng = _random.Random(tseed)
    family = rng.choice([MazeFamily.KRUSKAL, MazeFamily.WILSON])
    # Reasonable seed range keeps generated mazes interesting without trivial cases.
    maze_seed = rng.randrange(2000, 88000)

    STEPS = 16  # Enough frames for a satisfying visible refinement animation.

    result = run_proof_demo(
        ProofDemoConfig(
            checkpoint_path=ckpt,
            maze_family=family,
            maze_seed=maze_seed,
            sampling_steps=STEPS,
            retry_count=0,
            debug_reveal=False,
            capture_trajectory=True,
        )
    )

    # The player is the entire "result" the user sees. It is fully self-contained.
    player_html = build_diffusion_viz_html(
        result,
        maze_family=family,
        maze_seed=maze_seed,
        sampling_steps_used=STEPS,
    )
    return player_html


_run_automated_for_gradio = _safe_gpu_decorate(run_automated_demo)


def build_app() -> Any:
    """Minimal, intuitive Proof Demo UI.

    The user sees only a clear call-to-action button and the result area.
    All configuration (family, seed, sampling steps, retries, debug) is fully
    automated inside run_automated_demo. The result uses custom HTML/CSS/JS
    (allowed per requirements) to show the diffusion trajectory animating
    in real time.
    """
    import gradio as gr

    # Light custom CSS to make the primary action obvious and the demo feel clean.
    custom_css = """
    .dm-play-button button {
        font-size: 1.05rem !important;
        padding: 14px 32px !important;
        border-radius: 8px !important;
    }
    .dm-result .gr-html { border: none !important; background: transparent !important; }
    """

    with gr.Blocks(title="Dreamaze Proof Demo", css=custom_css) as demo:
        gr.Markdown(
            "# Dreamaze\n"
            "**A tiny Conditional Diffusion Solver learns to solve perfect Grid Mazes.**\n\n"
            "Click the button to generate a fresh maze and watch the model solve it. "
            "The Solution Mask is produced entirely by the learned model (no classical pathfinding at runtime). "
            "Graph Validation then decides if it is a Valid Solution."
        )

        with gr.Row():
            with gr.Column():
                # The ONLY interactive element the end-user needs.
                play_btn = gr.Button(
                    "▶ Solve New Maze — Watch the diffusion solver",
                    variant="primary",
                    elem_classes=["dm-play-button"],
                )

        # The entire result (maze + animated real-time solving + verdict) lives here.
        # The returned HTML block is rich, self-contained, and intuitive.
        result_area = gr.HTML(
            value="<div style='padding:12px;color:#64748b;font-size:0.95em'>Click the button above to run the Conditional Diffusion Solver and see the real-time denoising trajectory.</div>",
            elem_classes=["dm-result"],
        )

        play_btn.click(
            fn=_run_automated_for_gradio,
            inputs=[],
            outputs=[result_area],
            api_name="solve_new_maze",
        )

        # Populate an initial result on load so the page immediately shows the button + a result.
        # (Still just the automated path — user never sees or sets any config.)
        demo.load(
            fn=_run_automated_for_gradio,
            inputs=[],
            outputs=[result_area],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
