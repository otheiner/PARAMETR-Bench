## Contributing a task

This project started as a personal passion project but it would be great if it grew into something bigger. If you feel motivated to contribute a task from your domain, reach out to me and I will be happy to help. Process is simple:

1) Fork the repository and scaffold a new task:

```bash
python new_task.py --name my_task --author "Your Name"
```

2) Implement `generate_task()` in `tasks/my_task/generate.py` and fill in `tasks/my_task/config.json` and `tasks/my_task/metarubrics.json`. The framework handles everything else.

3) Validate without API calls:

```bash
python run.py --task my_task --validate-only
```

4) Open a pull request. 

Any scientific process with a simulatable generating distribution can become a task — physics, mathematics, chemistry, biology, climate science, and beyond. Tasks in the repo starting with underscore are skipped for benchmarking (unless explicitly requested in `run.py` by `--tasks <_task_name>`) but they are minimal working examples that can be used as a reference.