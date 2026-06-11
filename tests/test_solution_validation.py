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
