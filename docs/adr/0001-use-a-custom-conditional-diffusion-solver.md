# Use a custom conditional diffusion solver

Dreamaze's first version will use a custom Conditional Diffusion Solver that takes a clean Grid Maze as input and generates a Solution Mask as output. The solver should be trained from scratch on Dreamaze Dataset Artifacts, using Hugging Face Diffusers building blocks such as `UNet2DModel` rather than a pretrained Stable Diffusion image model, because the first goal is measurable maze solving rather than general image generation or visual style.
