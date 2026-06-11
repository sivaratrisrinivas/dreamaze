"""Dreamaze package."""

from dreamaze.dataset import (
    CellGraphMaze,
    MazeCondition,
    MazeFamily,
    TrainingExample,
    TrainingExampleConfig,
    build_kruskal_training_example,
    build_training_example,
    build_wilson_training_example,
)
from dreamaze.validation import (
    SolutionValidationResult,
    ValidationReason,
    validate_solution_mask,
)

__all__ = [
    "CellGraphMaze",
    "MazeCondition",
    "MazeFamily",
    "SolutionValidationResult",
    "TrainingExample",
    "TrainingExampleConfig",
    "ValidationReason",
    "build_kruskal_training_example",
    "build_training_example",
    "build_wilson_training_example",
    "validate_solution_mask",
]
