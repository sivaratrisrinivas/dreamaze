from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

Cell = tuple[int, int]
BoolGrid = Sequence[Sequence[bool]]


class ValidationReason(StrEnum):
    """Reasons a proposed Solution Mask is not a Valid Solution."""

    EMPTY_MASK = "empty_mask"
    MISSING_START = "missing_start"
    MISSING_GOAL = "missing_goal"
    WALL_CROSSING = "wall_crossing"
    DISCONNECTED = "disconnected"
    DIAGONAL_ONLY = "diagonal_only"
    EXTRA_BRANCH = "extra_branch"


@dataclass(frozen=True)
class SolutionValidationResult:
    valid: bool
    reason: ValidationReason | None = None


def validate_solution_mask(
    *,
    grid_maze: BoolGrid,
    solution_mask: BoolGrid,
    start_cell: Cell,
    goal_cell: Cell,
) -> SolutionValidationResult:
    mask_cells = _marked_cells(solution_mask)
    if not mask_cells:
        return SolutionValidationResult(
            valid=False, reason=ValidationReason.EMPTY_MASK
        )
    if start_cell not in mask_cells:
        return SolutionValidationResult(
            valid=False, reason=ValidationReason.MISSING_START
        )
    if goal_cell not in mask_cells:
        return SolutionValidationResult(
            valid=False, reason=ValidationReason.MISSING_GOAL
        )
    if any(not _is_open_cell(grid_maze, cell) for cell in mask_cells):
        return SolutionValidationResult(
            valid=False, reason=ValidationReason.WALL_CROSSING
        )
    connected_cells = _connected_mask_cells(mask_cells, start_cell)
    if goal_cell not in connected_cells or connected_cells != mask_cells:
        if _is_diagonal_only(mask_cells):
            return SolutionValidationResult(
                valid=False, reason=ValidationReason.DIAGONAL_ONLY
            )
        return SolutionValidationResult(
            valid=False, reason=ValidationReason.DISCONNECTED
        )
    if _has_extra_branch(mask_cells, start_cell, goal_cell):
        return SolutionValidationResult(
            valid=False, reason=ValidationReason.EXTRA_BRANCH
        )

    return SolutionValidationResult(valid=True)


def _marked_cells(solution_mask: BoolGrid) -> set[Cell]:
    return {
        (row_index, column_index)
        for row_index, row in enumerate(solution_mask)
        for column_index, is_marked in enumerate(row)
        if is_marked
    }


def _is_open_cell(grid_maze: BoolGrid, cell: Cell) -> bool:
    row, column = cell
    if row < 0 or column < 0:
        return False
    if row >= len(grid_maze) or column >= len(grid_maze[row]):
        return False
    return grid_maze[row][column]


def _connected_mask_cells(mask_cells: set[Cell], start_cell: Cell) -> set[Cell]:
    connected_cells: set[Cell] = set()
    frontier = [start_cell]

    while frontier:
        cell = frontier.pop()
        if cell in connected_cells:
            continue

        connected_cells.add(cell)
        frontier.extend(
            neighbor
            for neighbor in _four_way_neighbors(cell)
            if neighbor in mask_cells and neighbor not in connected_cells
        )

    return connected_cells


def _four_way_neighbors(cell: Cell) -> tuple[Cell, Cell, Cell, Cell]:
    row, column = cell
    return (
        (row - 1, column),
        (row + 1, column),
        (row, column - 1),
        (row, column + 1),
    )


def _is_diagonal_only(mask_cells: set[Cell]) -> bool:
    has_diagonal_contact = False

    for cell in mask_cells:
        if any(neighbor in mask_cells for neighbor in _four_way_neighbors(cell)):
            return False

    for row, column in mask_cells:
        if any(
            neighbor in mask_cells
            for neighbor in (
                (row - 1, column - 1),
                (row - 1, column + 1),
                (row + 1, column - 1),
                (row + 1, column + 1),
            )
        ):
            has_diagonal_contact = True

    return has_diagonal_contact


def _has_extra_branch(
    mask_cells: set[Cell], start_cell: Cell, goal_cell: Cell
) -> bool:
    for cell in mask_cells:
        marked_neighbor_count = sum(
            neighbor in mask_cells for neighbor in _four_way_neighbors(cell)
        )
        expected_neighbor_count = 1 if cell in {start_cell, goal_cell} else 2
        if marked_neighbor_count != expected_neighbor_count:
            return True
    return False
