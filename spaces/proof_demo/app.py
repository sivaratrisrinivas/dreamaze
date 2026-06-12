import os
from pathlib import Path
import random
import sys
import time
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
from dreamaze.proof_demo import ProofDemoConfig, build_diffusion_viz_html, run_proof_demo


_CHECKPOINT_ENV = os.environ.get("DREAMAZE_CHECKPOINT_PATH")
if not _CHECKPOINT_ENV:
    raise RuntimeError("DREAMAZE_CHECKPOINT_PATH must point to a Trained Solver Checkpoint")
CHECKPOINT_PATH = Path(_CHECKPOINT_ENV).expanduser()
if not CHECKPOINT_PATH.exists():
    raise RuntimeError(f"Trained Solver Checkpoint does not exist: {CHECKPOINT_PATH}")


def _safe_gpu_decorate(fn, duration=60):
    try:
        return spaces.GPU(duration=duration)(fn)
    except Exception:
        return fn


def run_automated_demo() -> str:
    tseed = int(time.time() * 1000) % 100000
    rng = random.Random(tseed)
    maze_family = rng.choice([MazeFamily.KRUSKAL, MazeFamily.WILSON])
    maze_seed = rng.randrange(2000, 88000)
    sampling_steps = 32

    result = run_proof_demo(
        ProofDemoConfig(
            checkpoint_path=CHECKPOINT_PATH,
            maze_family=maze_family,
            maze_seed=maze_seed,
            sampling_steps=sampling_steps,
            retry_count=0,
            debug_reveal=False,
            capture_trajectory=True,
            seed=rng.randrange(0, 1_000_000),
        )
    )
    return build_diffusion_viz_html(
        result,
        maze_family=maze_family,
        maze_seed=maze_seed,
        sampling_steps_used=sampling_steps,
    )


_run_automated_for_gradio = _safe_gpu_decorate(run_automated_demo)


def build_app() -> Any:
    import gradio as gr

    custom_css = """
    .dm-shell { max-width: 760px; margin: 0 auto; }
    .dm-play-button button {
        font-size: 1.05rem !important;
        padding: 14px 32px !important;
        border-radius: 8px !important;
    }
    .dm-result .gr-html { border: none !important; background: transparent !important; }
    """

    with gr.Blocks(title="Dreamaze Proof Demo", css=custom_css) as demo:
        with gr.Column(elem_classes=["dm-shell"]):
            play_btn = gr.Button(
                "Solve New Maze",
                variant="primary",
                elem_classes=["dm-play-button"],
            )
            result_area = gr.HTML(elem_classes=["dm-result"])

        play_btn.click(
            fn=_run_automated_for_gradio,
            inputs=[],
            outputs=[result_area],
            api_name="solve_new_maze",
        )
        demo.load(
            fn=_run_automated_for_gradio,
            inputs=[],
            outputs=[result_area],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
