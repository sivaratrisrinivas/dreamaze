import importlib.util
from pathlib import Path

from dreamaze.proof_demo import ProofDemoConfig, run_proof_demo


_SPACE_APP_PATH = Path(__file__).parents[1] / "spaces" / "proof_demo" / "app.py"
_SPACE_APP_SPEC = importlib.util.spec_from_file_location(
    "dreamaze_proof_demo_space_app", _SPACE_APP_PATH
)
assert _SPACE_APP_SPEC is not None
assert _SPACE_APP_SPEC.loader is not None
_SPACE_APP = importlib.util.module_from_spec(_SPACE_APP_SPEC)
_SPACE_APP_SPEC.loader.exec_module(_SPACE_APP)


def test_proof_demo_fixture_displays_maze_mask_and_validation_status():
    result = run_proof_demo(
        ProofDemoConfig(
            width=4,
            height=4,
            maze_seed=123,
            sampling_steps=2,
            seed=7,
        )
    )

    assert result.start_cell != result.goal_cell
    assert result.rendered_maze_svg.startswith("<svg")
    assert result.generated_solution_mask_svg.startswith("<svg")
    assert result.single_sample_success.valid in {True, False}
    assert result.validation_status in {"Valid Solution", "Invalid Solution"}
    if not result.single_sample_success.valid:
        assert result.validation_reason is not None


def test_proof_demo_debug_reveal_adds_training_label_and_difference_views():
    hidden = run_proof_demo(
        ProofDemoConfig(width=4, height=4, maze_seed=123, debug_reveal=False)
    )
    revealed = run_proof_demo(
        ProofDemoConfig(width=4, height=4, maze_seed=123, debug_reveal=True)
    )

    assert hidden.training_label_svg is None
    assert hidden.difference_svg is None
    assert revealed.training_label_svg is not None
    assert revealed.difference_svg is not None
    assert revealed.generated_solution_mask_svg != revealed.training_label_svg


def test_proof_demo_reports_retry_success_separately_from_single_sample():
    result = run_proof_demo(
        ProofDemoConfig(width=4, height=4, maze_seed=123, retry_count=2)
    )

    assert result.retry_success is not None
    assert result.retry_success.excluded_from_official_score is True
    assert result.official_score == "Single-Sample Success"


def test_space_demo_handler_returns_browser_facing_outputs():
    outputs = _SPACE_APP.run_demo(
        maze_family="kruskal",
        maze_seed=123,
        sampling_steps=2,
        retry_count=1,
        debug_reveal=True,
    )

    assert outputs["rendered_maze"].startswith("<svg")
    assert outputs["generated_solution_mask"].startswith("<svg")
    assert outputs["validation_status"] in {"Valid Solution", "Invalid Solution"}
    assert "Single-Sample Success" in outputs["sampling_summary"]
    assert outputs["training_label"].startswith("<svg")
    assert outputs["difference"].startswith("<svg")
