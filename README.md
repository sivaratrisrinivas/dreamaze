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

Dreamaze now includes Graph Validation for proposed Solution Masks.

The validator checks a submitted mask without creating, filling, repairing, or replacing the path. It accepts a mask only when the marked cells form one continuous 4-Way Movement route from the Start Cell to the Goal Cell through open Grid Maze cells.

Invalid masks return a structured Validation Reason:

- `empty_mask`
- `missing_start`
- `missing_goal`
- `wall_crossing`
- `disconnected`
- `diagonal_only`
- `extra_branch`

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
