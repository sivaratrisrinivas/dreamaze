# Use Hugging Face for compute and proof demo

Dreamaze's first version will target Hugging Face for model training, evaluation, artifact sharing, and the public Proof Demo instead of relying on the user's local GPU or a generic application host. Hugging Face Jobs should run training and evaluation, Hugging Face dataset and model repositories should hold Dataset Artifacts, evaluation reports, and checkpoints, and a public Hugging Face Space should host the Proof Demo.

The first deployment should live under the user's personal Hugging Face namespace so the repo, billing, and permissions stay simple while Dreamaze is still a Research-First Project. Because the user has Hugging Face Pro, the Proof Demo should prefer ZeroGPU for learned-model Runtime Solving, starting with the default ZeroGPU size for public-demo reliability and moving to larger or paid hardware only when evidence shows it is needed.
