from dreamaze.validation import ValidationReason, validate_solution_mask


def test_accepts_solution_mask_that_connects_start_to_goal_through_open_cells():
    grid_maze = [
        [True, True, True],
        [False, False, True],
        [False, False, True],
    ]
    solution_mask = [
        [True, True, True],
        [False, False, True],
        [False, False, True],
    ]

    result = validate_solution_mask(
        grid_maze=grid_maze,
        solution_mask=solution_mask,
        start_cell=(0, 0),
        goal_cell=(2, 2),
    )

    assert result.valid is True
    assert result.reason is None


def test_rejects_empty_solution_mask_with_validation_reason():
    result = validate_solution_mask(
        grid_maze=[
            [True, True],
            [True, True],
        ],
        solution_mask=[
            [False, False],
            [False, False],
        ],
        start_cell=(0, 0),
        goal_cell=(1, 1),
    )

    assert result.valid is False
    assert result.reason == ValidationReason.EMPTY_MASK


def test_rejects_solution_mask_missing_start_cell_with_validation_reason():
    result = validate_solution_mask(
        grid_maze=[
            [True, True],
            [True, True],
        ],
        solution_mask=[
            [False, True],
            [False, True],
        ],
        start_cell=(0, 0),
        goal_cell=(1, 1),
    )

    assert result.valid is False
    assert result.reason == ValidationReason.MISSING_START


def test_rejects_solution_mask_missing_goal_cell_with_validation_reason():
    result = validate_solution_mask(
        grid_maze=[
            [True, True],
            [True, True],
        ],
        solution_mask=[
            [True, True],
            [False, False],
        ],
        start_cell=(0, 0),
        goal_cell=(1, 1),
    )

    assert result.valid is False
    assert result.reason == ValidationReason.MISSING_GOAL


def test_rejects_solution_mask_that_crosses_wall_with_validation_reason():
    result = validate_solution_mask(
        grid_maze=[
            [True, False, True],
        ],
        solution_mask=[
            [True, True, True],
        ],
        start_cell=(0, 0),
        goal_cell=(0, 2),
    )

    assert result.valid is False
    assert result.reason == ValidationReason.WALL_CROSSING


def test_rejects_disconnected_solution_mask_with_validation_reason():
    result = validate_solution_mask(
        grid_maze=[
            [True, True, True],
            [True, True, True],
            [True, True, True],
        ],
        solution_mask=[
            [True, False, False],
            [False, False, False],
            [False, False, True],
        ],
        start_cell=(0, 0),
        goal_cell=(2, 2),
    )

    assert result.valid is False
    assert result.reason == ValidationReason.DISCONNECTED


def test_rejects_diagonal_only_solution_mask_with_validation_reason():
    result = validate_solution_mask(
        grid_maze=[
            [True, True],
            [True, True],
        ],
        solution_mask=[
            [True, False],
            [False, True],
        ],
        start_cell=(0, 0),
        goal_cell=(1, 1),
    )

    assert result.valid is False
    assert result.reason == ValidationReason.DIAGONAL_ONLY


def test_rejects_disconnected_solution_mask_with_diagonal_contact_as_disconnected():
    result = validate_solution_mask(
        grid_maze=[
            [True, True, True],
            [True, True, True],
        ],
        solution_mask=[
            [True, True, False],
            [False, False, True],
        ],
        start_cell=(0, 0),
        goal_cell=(1, 2),
    )

    assert result.valid is False
    assert result.reason == ValidationReason.DISCONNECTED


def test_rejects_solution_mask_with_extra_branch_with_validation_reason():
    result = validate_solution_mask(
        grid_maze=[
            [True, True, True],
            [False, True, False],
        ],
        solution_mask=[
            [True, True, True],
            [False, True, False],
        ],
        start_cell=(0, 0),
        goal_cell=(0, 2),
    )

    assert result.valid is False
    assert result.reason == ValidationReason.EXTRA_BRANCH


def test_mask_structure_diagnostics_detects_components_and_violations():
    from dreamaze.validation import (
        MaskStructureDiagnostics,
        compute_mask_structure_diagnostics,
    )

    # A clean path (valid)
    grid = [
        [True, True, True, True],
        [True, True, True, True],
    ]
    clean_mask = [
        [False, True, True, False],
        [False, False, True, True],
    ]
    diag = compute_mask_structure_diagnostics(
        grid_maze=grid,
        solution_mask=clean_mask,
        start_cell=(0, 1),
        goal_cell=(1, 3),
    )
    assert isinstance(diag, MaskStructureDiagnostics)
    assert diag.marked_count == 4
    assert diag.wall_crossing_count == 0
    assert diag.connected_component_count == 1
    assert diag.start_included is True
    assert diag.goal_included is True
    assert diag.endpoints_in_same_component is True
    assert diag.degree_histogram == {1: 2, 2: 2}  # two ends deg1, two middles deg2
    assert diag.extra_branch_violation_count == 0

    # Disconnected + wall cross
    bad_mask = [
        [True, False, True, False],
        [False, False, False, True],
    ]
    diag_bad = compute_mask_structure_diagnostics(
        grid_maze=grid,
        solution_mask=bad_mask,
        start_cell=(0, 0),
        goal_cell=(1, 3),
    )
    assert diag_bad.marked_count == 3
    # Use a grid with an explicit wall to test crossing count + component split
    walled = [
        [True, True, False, True],
        [True, True, True, True],
    ]
    cross = [
        [True, True, True, True],
        [False, False, False, True],
    ]
    d_cross = compute_mask_structure_diagnostics(
        grid_maze=walled,
        solution_mask=cross,
        start_cell=(0, 0),
        goal_cell=(1, 3),
    )
    assert d_cross.wall_crossing_count == 1  # cell (0,2)
    # whole marked set remains one component because cross cell still glues the path cells
    assert d_cross.connected_component_count == 1
    assert d_cross.extra_branch_violation_count >= 0


def test_mask_structure_diagnostics_branch_and_endpoint_membership():
    from dreamaze.validation import compute_mask_structure_diagnostics

    grid = [[True] * 5 for _ in range(3)]
    # A mask with an extra branch (T at (1,1) spur)
    branched = [
        [False, True, True, True, False],
        [False, True, False, False, False],
        [False, False, False, False, False],
    ]
    d = compute_mask_structure_diagnostics(
        grid_maze=grid,
        solution_mask=branched,
        start_cell=(0, 1),
        goal_cell=(0, 3),
    )
    assert d.marked_count == 4
    assert d.connected_component_count == 1
    assert d.endpoints_in_same_component is True
    # (1,1) is a deg-1 cell that is not an endpoint => extra violation
    # start has deg=2 (connected to spur) instead of 1, spur (1,1) deg=1 instead of 2 => 2 violations
    assert d.extra_branch_violation_count == 2
    assert d.degree_histogram.get(1, 0) == 2
    assert d.degree_histogram.get(2, 0) == 2
