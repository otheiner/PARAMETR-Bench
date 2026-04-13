# Count circles

This task is by default not included when running benchmark, but it is in the repository to demonstrate the framework on a minimal working example. If you want to run this task use flag `--task _count_circles` when running `run.py`. You can do:

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

This task is simple - it asks the model to count number of dots in each image and compute average number of donts in the image. This is very simple task which models will struggle with, if the number of dots is higher (approx. >10, `--difficulty hard`). However, if we allow agentic evaluation, more capable models will easily write python script that can analyze the image and they will score 100% on the task.