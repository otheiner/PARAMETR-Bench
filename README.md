![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![LiteLLM](https://img.shields.io/badge/LiteLLM-blueviolet?style=for-the-badge)

# Physics analysis benchmark 📊

A framework for building contamination-free scientific benchmarks for LLM evaluation with deterministically generated rubrics.

## What is this?

XXXX is a framework for evaluating LLMs on realistic scientific analysis workflows using procedurally generated tasks with perfectly synchronized rubrics.

Every run produces fresh multimodal instances (plots, CSVs, data tables) from a controlled generative process. The key innovation is source-grounded metarubrics: rubric templates that are automatically populated directly from the generated ground truth. This guarantees that evaluation criteria are always perfectly aligned with the task data — eliminating rubric drift by construction.

Because tasks are generated from a fixed distribution controlled by difficulty parameters and random seeds, the framework supports statistically rigorous evaluation. Multiple independent seeds at the same difficulty level allow treating each run as an independent trial, enabling proper confidence intervals, per-rubric breakdowns, and robust model comparisons.


## The core idea

Traditional benchmarks rely on fixed test sets that leak into training data, becoming contaminated or saturated. Common solutions — hiding test sets or constantly adding new questions — either sacrifice transparency or require unsustainable effort.

Procedural generation solves leakage by creating fresh instances every run. But it introduces a new problem: keeping rubrics aligned with dynamically generated data, especially in multi-step scientific tasks.

Our solution: use the same generating process that creates the task data to also instantiate the rubrics. We call these templates metarubrics. Every rubric criterion is mathematically guaranteed to match the generated instance  by construction, not by validation. 

Because rubric criteria contain specific numerical values drawn from the simulation, they cannot be gamed by memorising fixed evaluation criteria. A model must solve each instance on its own merits.


## Quick start

Clone repo and install dependencies:

```bash
git clone https://github.com/otheiner/physics-analysis-benchmarks
cd physics-analysis-benchmarks
pip install -r requirements.txt
```

The framework uses [litellm](https://github.com/BerriAI/litellm), supporting both local models via [Ollama](https://ollama.com) and API-based models. To use API models, add your keys:

```bash
cp .env.example .env   # fill in your API keys to .env
```

Run the benchmark and produce your results:

```bash
python run.py --models gemini/gemini-3.1-flash-lite-preview \
              --judge  gemini/gemini-2.5-flash \
              --difficulty medium \
              --seeds 0 1 2 3 4
```

Or validate task generation without API calls and inspect generated data:

```bash
python run.py --validate-only
```

## Results


## How it works

Each task is defined by three files:

- `generate.py` — simulation code producing input data and ground truth
- `metarubrics.json` — rubric templates instantiated from generated data  
- `config.json` — difficulty parameters

The pipeline:

1) generate_task() - generates fresh input_data/ + ground_truth/
2) populate_metarubrics() - fills templates from ground truth
3) generate_rubrics() - creates instances of metarubrics and produces rubrics.json
4) evaluator.run() - sends to model, judges output, saves results


## Contributing tasks

This repo started as a small personal passion project, however, if anybody feels motivated to contribute task from their domain, I will be more than happy to assist.

If you feel like contributing, fork this repository and implement `generate_task()` in your domain — the framework handles everything else. Do following:

```bash
python new_task.py --name my_task --author "Your Name"
```

Fill in `tasks/my_task/generate.py`, `tasks/my_task/config.json`, `tasks/my_task/metarubrics.json`.
Validate without API calls:

```bash
python run.py --task my_task --validate-only
```

Open pull request and you are done! Any scientific process with a simulatable generating distribution can become a task — physics, mathematics, chemistry, biology, climate science,... 


## Citation

If you find any idea presented in this repo interesting, you are welcome to use it freely in your work, but I will be happy if you cite this repo.

