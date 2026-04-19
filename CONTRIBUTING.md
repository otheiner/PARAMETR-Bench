# Contributing a task

This project started as a personal passion project but it would be great if it grows into something bigger. If you feel motivated to contribute a task from your domain (or you have any suggestion), reach out to me [here](https://otheiner.github.io/#contact) and I will be happy to help. Contributing process is simple:

1) **Fork the repository and scaffold a new task:**

```bash
python new_task.py --name my_task --author "Your Name"
```

2) **Create the task by:**
   - Implementing `_generate()` in `tasks/my_task/generate.py`.
   - Filling in `tasks/my_task/config.json` specifying dificulty levels.
   - Writing prompt in `tasks/my_task/prompt.md`.
   - Writing grading criteria in `tasks/my_task/metarubrics.json`.
     
The  framework handles everything else.

Before starting writing the task, check two minimal working examples. 
    -  `tasks/_count_circles/` — multimodal (images) task with tolerance-based grading
    - `tasks/_compute_average/` — text-only task with computational grading

These tasks are kept minimal on purpose. Follow the same structure — one `_generate()` method calling small private helpers, using `self.seed`, `self.get_params()`, `self.input_dir`, `self.ground_truth_dir`, and populating `self.ground_truth` with DataFrames.

3) **Validate without API calls:**

```bash
python run.py --task my_task --validate-only
```

4) **Open a pull request.**

Any scientific process with a simulatable generating distribution can become a task — physics, mathematics, chemistry, biology, climate science, and beyond. Tasks in the repo starting with underscore are skipped for benchmarking (unless explicitly requested in `run.py` by `--tasks <_task_name>`) but they are minimal working examples that can be used as a reference.
