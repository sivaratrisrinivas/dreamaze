# Use Hugging Face for compute and proof demo

Dreamaze's first version will target Hugging Face for model training, evaluation, artifact sharing, and the public Proof Demo instead of relying on the user's local GPU or a generic application host. Hugging Face Jobs should run training and evaluation, Hugging Face dataset and model repositories should hold Dataset Artifacts, evaluation reports, and checkpoints, and a public Hugging Face Space should host the Proof Demo.

The first deployment should live under the user's personal Hugging Face namespace so the repo, billing, and permissions stay simple while Dreamaze is still a Research-First Project. The original low-cost preference was ZeroGPU, but the Proof Demo now requires a plain HTML/CSS/JavaScript frontend rather than a Gradio UI. Because Hugging Face ZeroGPU is limited to Gradio SDK Spaces, the HTML frontend should be deployed as a Docker Space on regular paid GPU hardware when the trained Diffusers checkpoint requires CUDA.
