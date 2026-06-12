---
title: Dreamaze Proof Demo
emoji: 🧭
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.15.1
app_file: app.py
short_description: Validate learned maze solution masks
python_version: "3.12"
startup_duration_timeout: 30m
---

# Dreamaze Proof Demo

This Space runs the Dreamaze Conditional Diffusion Solver against a generated Grid Maze and displays the Rendered Maze, generated Solution Mask, Valid Solution status, Validation Reason, and optional Debug Reveal.

Set `DREAMAZE_CHECKPOINT_PATH` to a checkpoint path in the Space when a trained checkpoint is available. If it is unset, the Space uses the configured tiny fixture checkpoint for smoke testing.

Hardware is intentionally configured outside this README. Use `cpu-basic` for the tiny fixture, `zero-a10g` for a free queued GPU path on eligible Hugging Face Pro/Team/Enterprise accounts, or paid GPU hardware when latency matters more than creator cost.
