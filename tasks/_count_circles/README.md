# Count circles - minimal working example task

This task is, by default, not included when running the benchmark, but it is in the repository to demonstrate the framework on a minimal working example.

The task asks the model to count number of dots in each image and compute average number of dots in the image. This is very simple problem for human which models will struggle with, if the number of dots is slightly higher (`--difficulty hard`). However, if we allow agentic evaluation, moderately capable models will easily write python script that can analyze the image and they will score 100% on the task.

You can check the main logic behind the task implementation and also see how metarubrics (rubric templating) works.

If you want to run this task use flag `--task _count_circles` when running `run.py`. You can do:

```bash
python run.py --models gemini/gemini-3.1-flash-lite-preview \
              --judge  gemini/gemini-2.5-flash \
              --difficulty medium \
              --seeds 0 1
              --task _count_circles
```

Or use agentic evaluation:

```bash
python run.py --models gemini/gemini-3.1-flash-lite-preview \
              --judge  gemini/gemini-2.5-flash \
              --difficulty medium \
              --seeds 0 1
              --task _count_circles
              --agentic 
```
