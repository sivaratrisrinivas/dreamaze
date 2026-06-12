# Dreamaze

Dreamaze is a maze-solving project where a diffusion model learns to produce a solution path for generated mazes.

## Language

**Diffusion Maze Solver**:
A model that solves a maze by generating or refining a solution path through the maze. The solver is the learned model itself, not a separate graph-search algorithm.
_Avoid_: DFS solver, pathfinding visualizer

**Conditional Diffusion Solver**:
A Dreamaze-trained diffusion model that receives a **Grid Maze** and generates a **Solution Mask** for that maze.
_Avoid_: Stable Diffusion maze art, fake diffusion animation

**Maze**:
A connected puzzle space with walls, passages, a start, and a goal.
_Avoid_: Dungeon, grid art

**Perfect Maze**:
A connected **Maze** where there is exactly one route between any two open cells.
_Avoid_: Loopy maze, multi-solution maze

**Grid Maze**:
A **Maze** represented as a regular grid of blocked and open cells, with one start cell and one goal cell.
_Avoid_: Illustrated maze, painted maze

**Cell Graph Maze**:
A **Grid Maze** represented as cells connected by passages, where walls are the missing connections between neighboring cells.
_Avoid_: Pixel maze as source of truth, blocked-cell source of truth

**Rendered Maze**:
An image-like view of a **Cell Graph Maze** that shows walls, passages, the **Start Cell**, and the **Goal Cell** for the model or user interface.
_Avoid_: Source maze, maze logic

**4-Way Movement**:
Movement between neighboring **Grid Maze** cells only in the up, down, left, or right directions.
_Avoid_: Diagonal movement, corner cutting

**Start Cell**:
The cell in a **Grid Maze** where the **Solution Path** begins.
_Avoid_: Entry, spawn

**Goal Cell**:
The cell in a **Grid Maze** where the **Solution Path** ends.
_Avoid_: Exit, target

**Border Endpoint Pair**:
The chosen **Start Cell** and **Goal Cell** for a **Grid Maze**, placed on different border cells and preferably far apart.
_Avoid_: Fixed corners, hidden endpoints

**Maze Family**:
The generation style used to create a **Grid Maze**, such as Kruskal or Wilson.
_Avoid_: Theme, art style

**Training Maze Set**:
A collection of **Grid Mazes** used to teach the **Diffusion Maze Solver**, including mazes from both Kruskal and Wilson **Maze Families**.
_Avoid_: Single-generator dataset

**Solvable Training Set**:
A **Training Maze Set** where every **Training Example** has a **Unique Solution Path**.
_Avoid_: Impossible mazes, no-solution task

**Dataset Split**:
A separate train, validation, or test group of **Training Examples** generated from its own fixed seed range.
_Avoid_: Shared-seed split, hand-picked demo set

**First Dataset Size**:
The initial dataset scale for Dreamaze: 10,000 training examples, 1,000 validation examples, and 1,000 test examples.
_Avoid_: Tiny memorization set, giant first dataset

**Training Label**:
The correct **Solution Mask** for a **Grid Maze**, created before training so the model has an answer to learn from.
_Avoid_: Runtime solution, fallback answer

**Training Example**:
One **Grid Maze** with its **Start Cell**, **Goal Cell**, **Training Label**, and **Maze Family**.
_Avoid_: Demo frame, rendered illustration

**Maze Condition**:
The maze information given to the **Conditional Diffusion Solver**, including the **Rendered Maze**, **Start Cell**, and **Goal Cell**.
_Avoid_: Color-coded maze picture, hidden start and goal

**First Maze Size**:
The initial maze size used to prove the project, set to 16 by 16 **Grid Maze** cells rather than 16 by 16 pixels.
_Avoid_: Final maze size, production limit

**First Version**:
The first proof of Dreamaze, focused on teaching the **Diffusion Maze Solver** to produce **Valid Solutions** for clean **Grid Mazes**.
_Avoid_: Full magical demo, illustrated maze product

**Research-First Project**:
A project where dataset quality, model training, and measured **Valid-Solution Rate** define progress before visual polish.
_Avoid_: App-first project, art-first project

**Dataset Builder**:
The first implementation milestone that creates checked **Training Examples** for the **First Dataset Size**.
_Avoid_: First trained model, first web demo

**Dataset Artifact**:
The saved array-based output of the **Dataset Builder**, containing maze conditions, solution masks, endpoint positions, maze family, split, and seed.
_Avoid_: Preview image as source of truth, images-only dataset

**Preview Image**:
A human-readable rendering of a **Training Example** used for inspection, not as the source dataset.
_Avoid_: Dataset artifact, training source

**Proof Demo**:
A simple demonstration that shows a **Rendered Maze**, the generated **Solution Mask**, and whether the mask is a **Valid Solution**.
_Avoid_: Full magical demo, art-first demo

**Debug Reveal**:
A proof-demo mode that shows the **Training Label** and differences between it and the generated **Solution Mask**.
_Avoid_: Default answer reveal, hidden fallback

**Validation Reason**:
The specific reason a generated **Solution Mask** fails **Solution Validation**.
_Avoid_: Generic failure, hidden invalid output

**Solution Path**:
The route through a **Maze** from its start to its goal.
_Avoid_: Trail, DFS path

**Unique Solution Path**:
The only **Solution Path** through a **Perfect Maze** from the **Start Cell** to the **Goal Cell**.
_Avoid_: One possible path, shortest path among many

**Minimum Path Length**:
The shortest allowed **Unique Solution Path** for a **Training Example**, used to avoid trivial mazes.
_Avoid_: Path length balancing, unlimited tiny paths

**Solution Mask**:
An image-like layer that marks the cells or pixels belonging to the **Solution Path** and leaves non-solution areas unmarked.
_Avoid_: Move list, pretty overlay

**Diffusion Trajectory**:
The ordered sequence of intermediate **Solution Masks** produced while the **Conditional Diffusion Solver** refines noise into its final generated mask.
_Avoid_: Fake animation, decorative progress effect

**Real-Time Diffusion Playback**:
A visual playback of the **Diffusion Trajectory** that lets a user watch the **Conditional Diffusion Solver** refine its proposed **Solution Mask** step by step.
_Avoid_: Static result only, pathfinding animation

**Trained Solver Checkpoint**:
A saved **Conditional Diffusion Solver** state produced by Dreamaze training and used for **Runtime Solving**.
_Avoid_: Fixture weights, hand-tuned demo weights

**Valid Solution**:
A **Solution Mask** that connects the **Start Cell** to the **Goal Cell** through open cells in one continuous route with no extra branches.
_Avoid_: Plausible-looking path, decorative path

**Mask Cleanup**:
The process of turning a model's fuzzy **Solution Mask** into a clearer marked/unmarked mask without finding or repairing the route.
_Avoid_: Path repair, fallback solver

**Solution Validation**:
The check that decides whether a **Solution Mask** is a **Valid Solution**.
_Avoid_: Solution generation, pathfinding

**Graph Validation**:
A form of **Solution Validation** that checks connectivity and legality of a proposed **Solution Mask** without creating or changing the path.
_Avoid_: Graph solving, path repair

**Valid-Solution Rate**:
The share of tested **Grid Mazes** where the generated **Solution Mask** is a **Valid Solution**.
_Avoid_: Pixel accuracy as the main score, visual quality score

**Single-Sample Success**:
A **Valid Solution** produced from one generation attempt by the **Diffusion Maze Solver**.
_Avoid_: Retry-assisted success, best-of-many score

**Retry Success**:
A **Valid Solution** produced after one or more **Sampling Retries** for the same **Grid Maze**.
_Avoid_: Official success score

**First Success Target**:
The first proof target for Dreamaze: at least 80% **Valid-Solution Rate** from **Single-Sample Success** on unseen 16 by 16 Kruskal and Wilson **Grid Mazes**.
_Avoid_: Open-ended demo quality, visual-only success

**Runtime Solving**:
The live act of producing a **Solution Mask** for a **Grid Maze** after the model is trained.
_Avoid_: Offline labeling, dataset generation

**Sampling Retry**:
A new attempt by the **Diffusion Maze Solver** to generate a **Solution Mask** for the same **Grid Maze** after an invalid output.
_Avoid_: Fallback solver, algorithmic repair

## Example Dialogue

Developer: "Should we add DFS to solve the maze after generation?"

Domain expert: "No. Dreamaze is a Diffusion Maze Solver, so the learned model must produce the Solution Path."

Developer: "What should the model output?"

Domain expert: "A Solution Mask. We can draw that mask over the Maze for the user."

Developer: "Should the first model solve painted fantasy mazes?"

Domain expert: "No. The first model solves Grid Mazes so we can train and measure it clearly."

Developer: "Should we train only on Kruskal mazes?"

Domain expert: "No. The Training Maze Set mixes Kruskal and Wilson Maze Families so the solver learns more than one maze style."

Developer: "Can the model guess the start and goal?"

Domain expert: "No. The Grid Maze includes a Start Cell and Goal Cell so the Solution Path is clearly defined."

Developer: "Is a pretty path enough?"

Domain expert: "No. A Solution Mask must be a Valid Solution to count as solved."

Developer: "Can we use DFS to fix a broken model output?"

Domain expert: "No. Mask Cleanup may clean the output, and Solution Validation may check it, but neither may solve the maze."

Developer: "What happens if the model gives a bad path?"

Domain expert: "We may use a Sampling Retry, but we do not replace the model's answer with a path from another solver."

Developer: "Can there be multiple correct answers?"

Domain expert: "Not in the first version. The Training Maze Set uses Perfect Mazes with one Unique Solution Path."

Developer: "Should we start with huge mazes?"

Domain expert: "No. The First Maze Size is 16 by 16 cells so the solver can be trained and debugged clearly."

Developer: "Can a normal algorithm make the answer key?"

Domain expert: "Yes, but only as a Training Label. Runtime Solving must come from the Diffusion Maze Solver."

Developer: "What data does one training item need?"

Domain expert: "A Training Example includes the Grid Maze, Start Cell, Goal Cell, Training Label, and Maze Family."

Developer: "Should the first version include the full magical drawing demo?"

Domain expert: "No. The First Version proves the Diffusion Maze Solver can produce Valid Solutions for Grid Mazes."

Developer: "Are we using Stable Diffusion for the first solver?"

Domain expert: "No. The First Version uses a Conditional Diffusion Solver trained for Grid Maze inputs and Solution Mask outputs."

Developer: "Should the model infer the start and goal from colors in a picture?"

Domain expert: "No. The Maze Condition gives the Grid Maze, Start Cell, and Goal Cell clearly."

Developer: "What is the main score?"

Domain expert: "Valid-Solution Rate, because a path only matters if it truly connects the Start Cell to the Goal Cell."

Developer: "When does the first version count as working?"

Domain expert: "When it reaches the First Success Target: at least 80% Valid-Solution Rate on unseen 16 by 16 Kruskal and Wilson Grid Mazes."

Developer: "Can retries count toward the first success target?"

Domain expert: "No. The official score uses Single-Sample Success. Retry Success is useful for demos but not for the First Success Target."

Developer: "Can the checker use graph logic?"

Domain expert: "Yes. Graph Validation may inspect the model's proposed Solution Mask, but it may not create or repair a path."

Developer: "Can a solution include dead-end side branches?"

Domain expert: "No. A Valid Solution is one continuous route from Start Cell to Goal Cell with no extra branches."

Developer: "Can the path move diagonally?"

Domain expert: "No. Grid Mazes use 4-Way Movement."

Developer: "Is the source maze just a wall/open pixel image?"

Domain expert: "No. The source is a Cell Graph Maze, and the model receives a Rendered Maze made from it."

Developer: "Does the model get the hidden cell graph?"

Domain expert: "No. The Cell Graph Maze is the source of truth, but the Conditional Diffusion Solver receives the Rendered Maze."

Developer: "Should every maze start in the top-left and end in the bottom-right?"

Domain expert: "No. Each maze uses a Border Endpoint Pair so the solver does not learn fixed endpoint positions."

Developer: "Can the training set include very short solutions?"

Domain expert: "No. Training Examples must meet a Minimum Path Length so the solver learns meaningful mazes."

Developer: "Should version 1 include impossible mazes?"

Domain expert: "No. Version 1 uses a Solvable Training Set so every example has one correct Solution Mask."

Developer: "Can test mazes come from the same random seeds as training?"

Domain expert: "No. Each Dataset Split uses its own fixed seed range."

Developer: "How large is the first dataset?"

Domain expert: "The First Dataset Size is 10,000 training examples, 1,000 validation examples, and 1,000 test examples."

Developer: "What should the first demo show?"

Domain expert: "A Proof Demo: a Rendered Maze, the model's Solution Mask, and whether Solution Validation accepts it."

Developer: "Is Dreamaze an app-first project?"

Domain expert: "No. Dreamaze is a Research-First Project with a Proof Demo."

Developer: "What should we build first?"

Domain expert: "Build the Dataset Builder first so the model learns from trustworthy Training Examples."

Developer: "Should the dataset be PNG files?"

Domain expert: "No. The Dataset Artifact is array-based, while Preview Images are optional inspection tools."

Developer: "Should users always see the true answer?"

Domain expert: "No. The Proof Demo hides the Training Label by default and exposes it through Debug Reveal."

Developer: "What should the demo do with an invalid model answer?"

Domain expert: "Show the generated Solution Mask and its Validation Reason instead of hiding or replacing it."
