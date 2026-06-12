import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

from dreamaze.proof_demo import (
    ProofDemoResult,
    build_diffusion_viz_html,
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
    assert "Single-Sample Success" in html
    assert "FRAMES" in html
    assert "Replay solving steps" in html
    assert "fixture" not in html.lower()


def test_diffusion_viz_html_falls_back_without_trajectory():
    result = replace(_proof_demo_result(), diffusion_intermediates=None)

    html = build_diffusion_viz_html(result)

    assert "dreamaze-fallback" in html
    assert "Valid Solution" in html


def test_space_app_requires_checkpoint_at_startup(monkeypatch):
    monkeypatch.delenv("DREAMAZE_CHECKPOINT_PATH", raising=False)

    try:
        _load_space_app("dreamaze_space_missing_checkpoint")
    except RuntimeError as error:
        assert "DREAMAZE_CHECKPOINT_PATH" in str(error)
    else:
        raise AssertionError("Space app started without a Trained Solver Checkpoint")


def test_space_app_exposes_only_solve_new_maze_handler(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint-step-000001"
    checkpoint_dir.mkdir()
    monkeypatch.setenv("DREAMAZE_CHECKPOINT_PATH", str(checkpoint_dir))
    module = _load_space_app("dreamaze_space_one_button")

    assert hasattr(module, "run_automated_demo")
    assert not hasattr(module, "run_demo")
    assert not hasattr(module, "_run_demo_for_gradio")


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
