---
title: Dreamaze Proof Demo
emoji: 🧭
colorFrom: green
colorTo: blue
sdk: docker
short_description: Validate learned maze solution masks
python_version: "3.12"
startup_duration_timeout: 30m
---

# Dreamaze Proof Demo

This Space runs the Dreamaze Conditional Diffusion Solver against a fresh Grid Maze.

**The UI is fully automated and built with plain HTML/CSS/JavaScript**: there is only a "Solve New Maze" button. All settings (maze family, seed, sampling steps, retries, debug) are chosen internally so the visitor sees only the button and the result.

Clicking calls a small FastAPI backend and triggers a live **Runtime Solving** step: the model receives the Rendered Maze + Start Cell + Goal Cell and samples a Solution Mask via its iterative denoising process. The full trajectory of refinement steps is captured and played back in the browser as a smooth real-time animation so you can literally watch the diffusion model solve the maze. The final mask is then checked with strict Graph Validation / Solution Validation — no classical pathfinder is ever used to repair or replace the model's output.

The result area shows the evolving path claim, the final verdict ("Valid Solution" or "Invalid Solution" + reason), and a tiny legend. Everything uses the project's domain language (Conditional Diffusion Solver, Grid Maze, Solution Mask, Single-Sample Success, Graph Validation).

Set `DREAMAZE_CHECKPOINT_PATH` to a trained Dreamaze Diffusers checkpoint directory in the Space. If it is unset or points to a missing checkpoint, the Space fails at startup instead of showing a fixture result.

Hardware is intentionally configured outside this README. This HTML/CSS/JavaScript frontend is packaged as a Docker Space, so use regular paid GPU hardware for the trained Diffusers checkpoint. Hugging Face ZeroGPU is currently limited to Gradio SDK Spaces; use the older Gradio wrapper only if ZeroGPU is the hard requirement.
