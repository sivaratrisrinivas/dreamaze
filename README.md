# Dreamaze

Dreamaze is an experiment to teach a small AI model to solve mazes.

The first version is intentionally simple: it uses clean grid mazes, asks the model to draw the correct path from start to finish, and then checks whether that path is truly valid.

## What

Dreamaze will build a maze-solving model that:

- receives a maze with a marked start and goal
- predicts the path through the maze
- returns that path as a simple mask
- checks whether the path really reaches the goal without crossing walls

The first mazes are 16 by 16 cell grid mazes made with Kruskal's and Wilson's maze-generation methods.

## Why

The goal is to prove that the model is actually solving the maze, not just making a picture that looks right.

That means:

- the model must produce the answer path itself
- normal pathfinding algorithms can help create training answers
- normal pathfinding algorithms cannot secretly solve the maze during the demo
- success is measured by how often the model gives a real valid path

The first target is for the model to solve at least 80% of new test mazes in one try.

## How

The project will be built in stages.

1. Build a dataset generator.
   It creates perfect mazes, chooses start and goal cells, finds the correct answer path for training, and saves the data.

2. Train a small diffusion model.
   The model learns from the generated mazes and answer paths.

3. Measure the model.
   A checker verifies whether each predicted path is connected, stays inside open maze cells, reaches the goal, and has no extra branches.

4. Build a proof demo.
   The demo shows a maze, the model's predicted path, and whether the prediction is valid.

Training and the proof demo are planned for Hugging Face. Training should run on Hugging Face compute, and the demo should run on Hugging Face Spaces.

## Current Implementation

Dreamaze now includes Graph Validation for proposed Solution Masks, Dataset
Builder support for deterministic Kruskal and Wilson Training Examples plus
deterministic train, validation, and test Dataset Splits, and a small
Conditional Diffusion Solver training tracer bullet. The Dataset Builder can
persist those splits as sharded Dataset Artifacts with a resumable manifest.

The validator checks a submitted mask without creating, filling, repairing, or replacing the path. It accepts a mask only when the marked cells form one continuous 4-Way Movement route from the Start Cell to the Goal Cell through open Grid Maze cells.

Invalid masks return a structured Validation Reason:

- `empty_mask`
- `missing_start`
- `missing_goal`
- `wall_crossing`
- `disconnected`
- `diagonal_only`
- `extra_branch`

The Training Example builder creates a 16 by 16 Cell Graph Maze from a fixed
seed, explicit Maze Family, and config. It currently supports Kruskal and Wilson
Maze Families, chooses a seed-dependent Border Endpoint Pair, computes the
Unique Solution Path as the Training Label, renders the Maze Condition, and
checks cleanly through Solution Validation. The Training Label uses the same
rendered coordinate system as the Rendered Maze so wall-separated neighboring
cells do not look connected to Graph Validation.

Use `build_training_example(...)` with `TrainingExampleConfig.maze_family` to
select a Maze Family, or call the family-specific helpers
`build_kruskal_training_example(...)` and `build_wilson_training_example(...)`.

Dataset Splits are generated with `build_dataset_splits(...)` and
`DatasetConfig`. The config carries the First Maze Size, Maze Families, train /
validation / test split sizes, separate fixed seed ranges, the Border Endpoint
Pair rule, Minimum Path Length, output format, shard size, and Preview Image
setting. Each split uses its own fixed seed range, so generated Training
Examples are deterministic and split membership is stable. Expected Minimum
Path Length rejections are skipped and replaced until the requested split is
full or the rejection limit is hit. Invariant failures, including duplicate
split seeds, invalid Unique Solution Paths, invalid Training Labels, or
rendering mismatches, stop generation loudly.

Dataset Artifacts are written with `write_dataset_artifacts(...)`. The writer
stores compressed split shards, each containing the Maze Condition arrays,
Solution Masks, endpoint positions, Maze Family, split, seed, and metadata
needed for training and audit. It also writes `manifest.json` with the Dataset
Config, split names, split counts, seeds used, shard names, shard counts,
SHA-256 integrity checks, and build status. Re-running against a complete
matching manifest skips regeneration after checking every shard hash; incomplete
manifests or missing shards fail clearly. Optional Preview Images are written
under `previews/` for inspection and are not part of the training artifact
source data.

The Dataset Builder CLI can run the same artifact writer for a small smoke-test
dataset or the First Dataset Size:

```bash
dreamaze-dataset build --preset tiny --output-dir ./artifacts/tiny --preview-images
dreamaze-dataset build --preset first --output-dir ./artifacts/first
```

The `tiny` preset writes 3 training, 1 validation, and 1 test Training Example
with small 4 by 4 Grid Mazes for CI and local checks. The `first` preset uses
the First Dataset Size: 10,000 training, 1,000 validation, and 1,000 test
Training Examples. Re-running the command against a completed matching manifest
resumes by reusing the existing Dataset Artifacts after the writer verifies
shard integrity.

The first training tracer bullet is intentionally small and CPU-friendly. It
loads sharded Dataset Artifacts, conditions on Maze Condition arrays, trains a
custom Conditional Diffusion Solver toward Solution Masks, and writes JSON
checkpoints. It is a smoke path for training infrastructure, not a claim that
Dreamaze has reached the First Success Target.

Create a training config such as:

```json
{
  "dataset_dir": "./artifacts/tiny",
  "checkpoint_dir": "./artifacts/checkpoints",
  "split": "train",
  "batch_size": 2,
  "sampling_steps": 3,
  "max_train_steps": 1,
  "checkpoint_every_steps": 1,
  "learning_rate": 0.05,
  "seed": 123,
  "device": "cpu",
  "precision": "float32",
  "num_workers": 0
}
```

Then run:

```bash
dreamaze-train --config ./training.json
```

Run the test suite with:

```bash
pytest
```

## Not Yet

These ideas are intentionally saved for later:

- drawing a squiggle to create a maze
- illustrated fantasy-style maze art
- large mazes
- impossible mazes with no solution
- polished mobile app experience

The first job is smaller and stricter: prove that the model can solve clean mazes.
