# Contributing a task

This started as a personal passion project, but it would be great if it grows into something larger. Even if it doesn't, this document serves as documentation of how to add a new task to the framework and is addressed to my future self as much as anyone else.

If you'd like to contribute a task from your domain (or just have a suggestion), reach out [here](https://otheiner.github.io/#contact) and I'll be happy to help. The contribution process is straightforward:

## 1) Fork the repository and clone it

```bash
git clone https://github.com/<your-username>/PARAMETR-Bench.git
cd PARAMETR-Bench
```

## 2) Scaffold a new task

```bash
python new_task.py --name <new-task-name> --author "Your Name"
```

This creates a task folder in `tasks` directory with four files to fill in:

- `prompt.md` — the task prompt
- `generate.py` — the data generator producing inputs and ground truth (implement `_generate()`)
- `config.json` — difficulty parameters loaded by `generate.py` (currently: easy, medium, hard)
- `metarubrics.json` — rubric templates that the framework instantiates from generated ground truth


## 3) Implement the task

Each task lives in `tasks/<new-task-name>/` and consists of four files:

- `generate.py` — implement `_generate()`
- `config.json` — define difficulty levels (easy, medium, hard)
- `prompt.md` — write the task prompt in natural language
- `metarubrics.json` — write the grading criteria as templates

Once these are in place, the framework handles rubric instantiation, sandboxing, model invocation, and judging automatically.

### Start from a minimal example

Before writing your own, read the two reference tasks:

- `tasks/_count_circles/` — multimodal (images), tolerance-based grading
- `tasks/_compute_average/` — text-only, computational grading

These are kept deliberately minimal. Follow the same structure: a single `_generate()` method calling small private helpers, using `self.seed`, `self.get_params()`, `self.input_dir`, `self.ground_truth_dir`, and populating `self.ground_truth` with DataFrames.

### Implementation rules

- **Seeding.** Any random number generator is fine, but it must be initialized from `self.seed` (or a value deterministically derived from it). This is what makes task generation reproducible and controllable by the framework.
- **Parameters.** Simulation parameters live in `config.json` and are loaded via `self.get_params()`. See the minimal examples for the expected structure.
- **Input data.** Whatever you generate as model input goes into `self.input_dir`. You can create files and subdirectories freely.
- **Ground truth.** Data used for grading goes into `self.ground_truth` (see below). You can also write auxiliary files to `self.ground_truth_dir` — these aren't used for grading but are useful for human inspection.
- **Ground-truth format.** Anything referenced by a metarubric must be stored as a pandas DataFrame in the `self.ground_truth` dictionary. You can have as many DataFrames as you need. The dictionary key becomes the `source` field in `metarubrics.json`, and the DataFrame's columns become the variables available in metarubric template strings. For example the criterion `Did the model compute {z} for galaxy {galaxy_ID}?` pulls `z` and `galaxy_ID` from the DataFrame columns of the source.

## 4) Validate without API calls

```bash
python run.py --task my_task --validate-only
```
This same check is run by github CI/CD pipeline, so it needs to pass.

## 5) Open a pull request

Any scientific process with a simulatable generating distribution can become a task — physics, mathematics, chemistry, biology, climate science, and beyond. Tasks in the repo starting with underscore are skipped for benchmarking (unless explicitly requested in `run.py` by `--tasks <_task_name>`) but they are minimal working examples that can be used as a reference.
