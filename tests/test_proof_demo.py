import importlib.util
import sys
import types
from dataclasses import replace
from pathlib import Path

from dreamaze.proof_demo import (
    ProofDemoResult,
    build_diffusion_viz_html,
    iter_proof_demo_stream_events,
    run_proof_demo,
)
from dreamaze.validation import SolutionValidationResult


_SPACE_APP_PATH = Path(__file__).parents[1] / "spaces" / "proof_demo" / "app.py"


def test_proof_demo_requires_a_trained_solver_checkpoint():
    try:
        run_proof_demo(
            _minimal_config(checkpoint_path=Path("/tmp/missing-dreamaze-checkpoint"))
        )
    except FileNotFoundError:
        pass
    except RuntimeError as error:
        assert "torch and diffusers" in str(error)
    else:
        raise AssertionError("Proof Demo ran without a Trained Solver Checkpoint")


def test_diffusion_viz_html_animates_captured_trajectory():
    result = _proof_demo_result()

    html = build_diffusion_viz_html(
        result,
        maze_family="kruskal",
        maze_seed=123,
        sampling_steps_used=2,
    )

    assert "Conditional Diffusion Solver trajectory" in html
    assert "FRAMES" in html
    assert "STEP_MS = 750" in html
    assert "fixture" not in html.lower()


def test_diffusion_viz_html_falls_back_without_trajectory():
    result = replace(_proof_demo_result(), diffusion_intermediates=None)

    html = build_diffusion_viz_html(result)

    assert "dreamaze-fallback" in html
    assert "Valid Solution" in html


def test_proof_demo_stream_events_emit_init_frames_and_done(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint-step-000001"
    checkpoint_dir.mkdir()
    masks = [
        ((False, False, False), (False, True, False), (False, False, False)),
        ((False, True, False), (False, True, False), (False, True, False)),
    ]

    monkeypatch.setattr(
        "dreamaze.proof_demo.load_diffusers_solver_checkpoint",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "dreamaze.proof_demo.iter_conditional_diffusion_solution_mask_trajectory",
        lambda **_kwargs: iter(masks),
    )

    events = list(
        iter_proof_demo_stream_events(_minimal_config(checkpoint_path=checkpoint_dir))
    )

    assert events[0]["type"] == "init"
    assert events[0]["totalSteps"] == 2
    assert events[1] == {
        "type": "frame",
        "step": 0,
        "mask": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    }
    assert events[2]["type"] == "frame"
    assert events[-1]["type"] == "done"
    assert events[-1]["validationStatus"] in {"Valid Solution", "Invalid Solution"}


def test_space_app_requires_checkpoint_at_startup(monkeypatch):
    monkeypatch.delenv("DREAMAZE_CHECKPOINT_PATH", raising=False)
    monkeypatch.delenv("DREAMAZE_ALLOW_SMOKE_MODE", raising=False)

    try:
        _load_space_app("dreamaze_space_missing_checkpoint")
    except RuntimeError as error:
        assert "DREAMAZE_CHECKPOINT_PATH" in str(error)
    else:
        raise AssertionError("Space app started without a Trained Solver Checkpoint")


def test_space_app_can_start_in_explicit_smoke_mode_without_checkpoint(monkeypatch):
    monkeypatch.delenv("DREAMAZE_CHECKPOINT_PATH", raising=False)
    monkeypatch.setenv("DREAMAZE_ALLOW_SMOKE_MODE", "1")

    module = _load_space_app("dreamaze_space_smoke_mode")

    assert module.healthz() == {"status": "ok"}
    assert "UI smoke mode" in module.run_automated_demo()


def test_space_app_can_resolve_checkpoint_from_hugging_face_repo_env(
    monkeypatch, tmp_path
):
    checkpoint_root = tmp_path / "hub-cache"
    checkpoint_dir = checkpoint_root / "checkpoints" / "run" / "checkpoint-step-000001"
    (checkpoint_dir / "unet").mkdir(parents=True)
    (checkpoint_dir / "scheduler").mkdir()
    (checkpoint_dir / "metadata.json").write_text("{}")

    fake_hub = types.SimpleNamespace(
        snapshot_download=lambda **_kwargs: str(checkpoint_root)
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.delenv("DREAMAZE_CHECKPOINT_PATH", raising=False)
    monkeypatch.delenv("DREAMAZE_ALLOW_SMOKE_MODE", raising=False)
    monkeypatch.setenv("DREAMAZE_CHECKPOINT_REPO_ID", "Srini410/dreamaze-solver")
    monkeypatch.setenv(
        "DREAMAZE_CHECKPOINT_REPO_PATH", "checkpoints/run/checkpoint-step-000001"
    )

    module = _load_space_app("dreamaze_space_repo_checkpoint")

    assert module._resolve_checkpoint_path() == checkpoint_dir


def test_space_app_exposes_only_solve_new_maze_handler(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint-step-000001"
    checkpoint_dir.mkdir()
    monkeypatch.setenv("DREAMAZE_CHECKPOINT_PATH", str(checkpoint_dir))
    module = _load_space_app("dreamaze_space_one_button")

    assert hasattr(module, "run_automated_demo")
    assert hasattr(module, "solve_new_maze")
    assert hasattr(module, "app")
    assert not hasattr(module, "run_demo")
    assert not hasattr(module, "_run_demo_for_gradio")
    assert not hasattr(module, "build_app")


def test_space_app_serves_html_css_js_ui(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint-step-000001"
    checkpoint_dir.mkdir()
    monkeypatch.setenv("DREAMAZE_CHECKPOINT_PATH", str(checkpoint_dir))
    module = _load_space_app("dreamaze_space_static_ui")

    index = module.index()
    static_dir = _SPACE_APP_PATH.parent / "static"

    assert Path(index.path) == static_dir / "index.html"
    assert (static_dir / "styles.css").exists()
    assert (static_dir / "app.js").exists()


def test_space_app_solve_endpoint_returns_visualization_html(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint-step-000001"
    checkpoint_dir.mkdir()
    monkeypatch.setenv("DREAMAZE_CHECKPOINT_PATH", str(checkpoint_dir))
    module = _load_space_app("dreamaze_space_solve_endpoint")
    monkeypatch.setattr(module, "_run_automated_for_http", lambda: "<div>solved</div>")

    assert module.solve_new_maze() == {"html": "<div>solved</div>"}


def test_space_app_stream_endpoint_returns_event_stream(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint-step-000001"
    checkpoint_dir.mkdir()
    monkeypatch.setenv("DREAMAZE_CHECKPOINT_PATH", str(checkpoint_dir))
    module = _load_space_app("dreamaze_space_stream_endpoint")

    response = module.solve_new_maze_stream()

    assert response.media_type == "text/event-stream"


def test_space_app_stream_events_are_sse_encoded(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint-step-000001"
    checkpoint_dir.mkdir()
    monkeypatch.setenv("DREAMAZE_CHECKPOINT_PATH", str(checkpoint_dir))
    module = _load_space_app("dreamaze_space_stream_sse")
    monkeypatch.setattr(
        module,
        "_stream_automated_for_http",
        lambda: iter([{"type": "frame", "step": 1, "mask": [[1]]}]),
    )

    assert list(module._stream_sse_events()) == [
        'data: {"type":"frame","step":1,"mask":[[1]]}\n\n'
    ]


def _load_space_app(module_name: str):
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, _SPACE_APP_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_config(*, checkpoint_path: Path):
    from dreamaze.proof_demo import ProofDemoConfig

    return ProofDemoConfig(
        checkpoint_path=checkpoint_path,
        width=4,
        height=4,
        maze_seed=123,
        sampling_steps=2,
        seed=7,
    )


def _proof_demo_result() -> ProofDemoResult:
    rendered_maze = (
        (False, False, False),
        (False, True, False),
        (False, False, False),
    )
    mask = (
        (False, False, False),
        (False, True, False),
        (False, False, False),
    )
    svg = '<svg viewBox="0 0 3 3"></svg>'
    return ProofDemoResult(
        rendered_maze_svg=svg,
        generated_solution_mask_svg=svg,
        start_cell=(0, 0),
        goal_cell=(0, 0),
        rendered_start_cell=(1, 1),
        rendered_goal_cell=(1, 1),
        rendered_maze=rendered_maze,
        validation_status="Valid Solution",
        validation_reason=None,
        single_sample_success=SolutionValidationResult(valid=True),
        diffusion_intermediates=[mask, mask, mask],
    )
