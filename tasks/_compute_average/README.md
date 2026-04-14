# Compute average - minimal working example task

This task is simple reading numbers from the file and computing average of these numbers. Moderately capable LLM can do this task simply in non-agentic regime, however, smaller models might halucinate average for medium and hard difficulty when the list is long.

You can check the main logic behind the task implementation and also see how metarubrics (rubric templating) works.

This task is, by default, not included when running benchmark, but it is in the repository to demonstrate the framework on a minimal working example. If you want to run this task use flag `--task _compute_average` when running `run.py`. You can do:

```bash
python run.py --models gemini/gemini-3.1-flash-lite-preview \
              --judge  gemini/gemini-2.5-flash \
              --difficulty medium \
              --seeds 0 1
              --task _compute_average
```

Or use agentic evaluation:

```bash
python run.py --models gemini/gemini-3.1-flash-lite-preview \
              --judge  gemini/gemini-2.5-flash \
              --difficulty medium \
              --seeds 0 1
              --task _compute_average
              --agentic 
```