import os
from pathlib import Path
import random
import sys
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_SPACE_ROOT = Path(__file__).resolve().parent
_SPACE_SRC_PATH = _SPACE_ROOT / "src"
if _SPACE_SRC_PATH.exists():
    sys.path.insert(0, str(_SPACE_SRC_PATH))

from dreamaze.dataset import MazeFamily
from dreamaze.proof_demo import ProofDemoConfig, build_diffusion_viz_html, run_proof_demo


_CHECKPOINT_ENV = os.environ.get("DREAMAZE_CHECKPOINT_PATH")
_CHECKPOINT_REPO_ID = os.environ.get("DREAMAZE_CHECKPOINT_REPO_ID")
_CHECKPOINT_REPO_PATH = os.environ.get("DREAMAZE_CHECKPOINT_REPO_PATH")
_CHECKPOINT_REVISION = os.environ.get("DREAMAZE_CHECKPOINT_REVISION")
_ALLOW_SMOKE_MODE = os.environ.get("DREAMAZE_ALLOW_SMOKE_MODE") == "1"


_CHECKPOINT_PATH_CACHE: Path | None = None


def _resolve_checkpoint_path() -> Path | None:
    global _CHECKPOINT_PATH_CACHE
    if _CHECKPOINT_PATH_CACHE is not None:
        return _CHECKPOINT_PATH_CACHE
    if _CHECKPOINT_ENV:
        _CHECKPOINT_PATH_CACHE = Path(_CHECKPOINT_ENV).expanduser()
        return _CHECKPOINT_PATH_CACHE
    if _CHECKPOINT_REPO_ID and _CHECKPOINT_REPO_PATH:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "DREAMAZE_CHECKPOINT_REPO_ID requires huggingface_hub in the Space runtime"
            ) from error
        _CHECKPOINT_PATH_CACHE = Path(
            snapshot_download(
                repo_id=_CHECKPOINT_REPO_ID,
                revision=_CHECKPOINT_REVISION,
                allow_patterns=f"{_CHECKPOINT_REPO_PATH.rstrip('/')}/*",
            )
        ) / _CHECKPOINT_REPO_PATH
        return _CHECKPOINT_PATH_CACHE
    if _ALLOW_SMOKE_MODE:
        return None
    raise RuntimeError(
        "Set DREAMAZE_CHECKPOINT_PATH or DREAMAZE_CHECKPOINT_REPO_ID + "
        "DREAMAZE_CHECKPOINT_REPO_PATH to a Trained Solver Checkpoint"
    )


if not (_CHECKPOINT_ENV or (_CHECKPOINT_REPO_ID and _CHECKPOINT_REPO_PATH) or _ALLOW_SMOKE_MODE):
    raise RuntimeError(
        "Set DREAMAZE_CHECKPOINT_PATH or DREAMAZE_CHECKPOINT_REPO_ID + "
        "DREAMAZE_CHECKPOINT_REPO_PATH to a Trained Solver Checkpoint"
    )


def run_automated_demo() -> str:
    checkpoint_path = _resolve_checkpoint_path()
    if checkpoint_path is None:
        return (
            '<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;'
            'max-width:620px;margin:0 auto;padding:18px;background:#fff;border:1px solid #d9e0e5;'
            'border-radius:8px;color:#172026">'
            '<div style="font-size:12px;font-weight:700;color:#0b5f59;text-transform:uppercase">'
            "UI smoke mode</div>"
            '<h2 style="margin:8px 0 10px;font-size:22px;letter-spacing:0">The app shell is working</h2>'
            '<p style="margin:0 0 10px;color:#61717d;line-height:1.45">'
            "A real maze solve needs a trained Dreamaze Diffusers checkpoint. "
            "Deploy with a Hugging Face checkpoint repo path when you have one:"
            "</p>"
            '<code style="display:block;white-space:normal;background:#eef2f4;'
            'border:1px solid #d9e0e5;border-radius:6px;padding:10px;color:#172026">'
            "DREAMAZE_CHECKPOINT_REPO_ID=Srini410/dreamaze-solver "
            "DREAMAZE_CHECKPOINT_REPO_PATH=checkpoints/run/checkpoint-step-xxxxxx ./run.sh"
            "</code>"
            "</div>"
        )
    if not checkpoint_path.exists():
        raise RuntimeError(f"Trained Solver Checkpoint does not exist: {checkpoint_path}")

    tseed = int(time.time() * 1000) % 100000
    rng = random.Random(tseed)
    maze_family = rng.choice([MazeFamily.KRUSKAL, MazeFamily.WILSON])
    maze_seed = rng.randrange(2000, 88000)
    sampling_steps = 32

    result = run_proof_demo(
        ProofDemoConfig(
            checkpoint_path=checkpoint_path,
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


_run_automated_for_http = run_automated_demo

app = FastAPI(title="Dreamaze Proof Demo")
app.mount("/static", StaticFiles(directory=_SPACE_ROOT / "static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_SPACE_ROOT / "static" / "index.html")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/solve_new_maze")
def solve_new_maze() -> dict[str, str]:
    return {"html": _run_automated_for_http()}
