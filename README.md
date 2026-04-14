[![CI](https://github.com/otheiner/PARAMETR-Bench/actions/workflows/ci.yml/badge.svg)](https://github.com/otheiner/PARAMETR-Bench/actions/workflows/ci.yml)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![LiteLLM](https://img.shields.io/badge/LiteLLM-blueviolet?style=for-the-badge)


# 📊 PARAMETR-Bench  

**P**rocedural **A**nd **R**igorous **A**ssessment using **MET**a**R**ubrics: A framework for building contamination-free scientific benchmarks for agentic LLM evaluation with deterministically generated rubrics.

## What is this?

PARAMETR-Bench is a framework for evaluating LLMs on realistic scientific analysis workflows using procedurally generated tasks with perfectly synchronized rubrics.

Every run produces fresh multimodal instances (plots, CSVs, data tables) from a controlled generative process. The key innovation is source-grounded metarubrics: rubric templates that are automatically populated directly from the generated ground truth. This guarantees that evaluation criteria are always perfectly aligned with the task data, eliminating rubric drift by construction.

Because tasks are generated from a fixed distribution controlled by difficulty parameters and random seeds, the framework enables statistically rigorous evaluation. Running multiple independent seeds at the same difficulty level turns each evaluation into a set of independent trials, allowing proper confidence intervals, per-metarubric breakdowns, and more reliable model comparisons.

The repository includes several tasks inspired by landmark discoveries in particle physics and cosmology, such as invariant mass reconstruction and Cepheid variable calibration. However, any scientific process with a simulatable generating distribution can become a task — physics, mathematics, chemistry, biology, climate science, and beyond.

Expand the section below and see the concrete example of how metarubrics and rubrics work.

<details>
<summary><strong>✅ Metarubrics vs. Rubrics (click to expand)</strong></summary>

## Metarubrics and Rubrics

The user only needs to define the metarubric (the template); the framework handles the rest. In this context, metarubrics are analogous to classes in object-oriented programming, while rubrics are specific instances of those classes instantiated with unique parameters for a given task.

### 1. User-Defined Metarubric (The Template)
The user provides a high-level template with placeholders.

```json
"metarubrics": [
    {
      "key": "z_estimation",
      "source": "analyzed_galaxies",
      "name": "Redshift estimation",
      "description": "Did the model compute that galaxy {galaxy_ID} has redshift {z}, or a value strictly inside the interval [{z_min}, {z_max}]?",
      "weight": 5.0
    }
]
```

### 2. Framework-Generated Rubrics (The Instances)
The framework populates the template using the ground truth from the procedurally generated dataset.

```json
"metarubrics": [
    {
      "key": "z_estimation",
      "name": "Redshift estimation",
      "weight": 5.0,
      "total": 3,
      "rubrics": [
        {
          "id": 1,
          "criterion": "Did the model compute that galaxy GID075008 has redshift 0.02978, or value strictly inside interval [0.02928 , 0.03028]?"
        },
        {
          "id": 2,
          "criterion": "Did the model compute that galaxy GID104365 has redshift 0.01951, or value strictly inside interval [0.01901 , 0.02001]?"
        },
        {
          "id": 3,
          "criterion": "Did the model compute that galaxy GID173179 has redshift 0.01831, or value strictly inside interval [0.01781 , 0.01881]?"
        }
      ]
    }
]
```

### Why this matters

Because tasks are generated procedurally, the number of generated rubrics (instances) can vary between individual runs. This dynamic approach prevents rubric drift - where an agent might memorize static answers - and enables granular, automated grading with zero human intervention.

</details>

<p align="center">
    <br>
  <img src="https://github.com/user-attachments/assets/2411dc3a-c4b0-4d94-81fb-c01289c5835b" width="900" title="Preview of some input files fenerated procedurally.">
    <br><br>
      <i> Figure: Preview of a few input files from tasks in PARAMETR-Bench generated procedurally.</i>
</p>

## The core idea

Traditional benchmarks rely on fixed test sets that leak into training data, becoming contaminated or saturated. Common solutions are hiding test sets or constantly adding new questions. These approaches either sacrifice benchmark transparency or require unsustainable effort.

Procedural generation solves leakage by creating fresh instances every run. But it introduces a new problem: keeping rubrics aligned with dynamically generated data, especially in multi-step scientific tasks.

Our solution is to use the same generating process that creates the task data to also instantiate the rubrics. We call these templates metarubrics. Every rubric criterion is mathematically guaranteed to match the generated instance  by construction, not by validation. Templating allows us also automatically generate variable number of atomic rubric criteria for repeated data extraction, which is common in scientific data analyses.

Since rubric criteria contain specific numerical values drawn from the simulation, they cannot be gamed by memorising fixed evaluation criteria. A model must solve each instance on its own merits.

<details>
<summary><strong>⚠️ How to detect "cheaters"? (click to expand)</strong></summary>

## Seeded generation

PARAMETR-Bench utilizes user-specified seeds to ensure that task generation is randomized yet fully reproducible. While the repository itself contains no raw data, the specific set of seeds serves as a precise "recipe" for reconstructing the evaluation dataset. This approach is advantageous for several reasons:

1) **Minimizing Data Contamination:** Since the evaluation data is generated on the fly and never stored statically in the repository, the risk of it being scraped and contaminating future LLM training sets is significantly reduced.
2) **Leak Detection:** If evaluation data from specific public seeds were to leak into a model's training set, the model might show inflated performance due to memorization. We can detect this by re-running the benchmark with a fresh set of random seeds. A statistically significant performance gap between public seeds and fresh private seeds would provide a clear indication of a potential data leak — making contamination detectable in principle, unlike static benchmarks where held-out sets differ in content.
3) **Resilience to Leaks:** If a specific dataset is compromised, the seeds can simply be rotated. Because our framework relies on statistical evaluation across multiple independent seeds, the resulting performance metrics remain comparable and valid even after the seeds are changed.
    
</details>

## Motivation 

I am a particle physicist who recently got into LLM evaluation and I built this as a passion project to test if models can actually do science. I tried to design the architecture to specifically address major issues in the field, such as benchmark contamination, rubric drift, and the stochastic nature of LLM responses. If you have any questions, comments, suggestions, or you would be interested in contributing, don't hesitate to reach out to me [here](https://otheiner.github.io/#contact).

# Quick start

Clone repo and install dependencies:

```bash
git clone https://github.com/otheiner/PARAMETR-Bench.git
cd PARAMETR-Bench
pip install -r requirements.txt
```

Validate task generation without API calls and inspect generated data locally:

```bash
python run.py --validate-only
```

Run the benchmark and produce your results (you can plug any models of your choice supported by `litellm`). Framework allows non-agentic (no tools allowed) and agentic (allows running python scripts) evaluation. Details on how to run actual evaluation with or without agent is described in the following expandable section.

<details>
<summary><strong>👍 Agentic vs. non-agentic evaluation (click to expand)</strong></summary>

PARAMETR-Bench uses [litellm](https://github.com/BerriAI/litellm), supporting both local models via [Ollama](https://ollama.com) and API-based models. If you want to use local models make sure that your `ollama` server is installed and running (see Ollama link). To use API models, add your keys:

```bash
cp .env.example .env   # fill in your API keys to .env
```
    
## Non-agentic evaluation

For non-agentic evaluation simply run:

```bash
python run.py --models gemini/gemini-3.1-flash-lite-preview \
              --judge  gemini/gemini-2.5-flash \
              --difficulty medium \
              --seeds 0 1
```

## Agentic evaluation

Agentic evaluation enables running python with a few allowed python libraries specified in `sandbox/requirements.txt`. Python is executed in safe Docker sandbox environment without access to the Internet, memory-capped to 512 MB, not allowing writing `.pyc` files. **First of all, run docker daemon on your machine!** Dockerfile for building the sandbox is located in `sandbox` folder. 

Build the sandbox Docker image and run benchmark using flag `--agentic`:

```bash
docker build -t benchmark-sandbox sandbox/
python run.py --models gemini/gemini-3.1-flash-lite-preview \
              --judge  gemini/gemini-2.5-flash \
              --difficulty medium \
              --agentic \
              --seeds 0 1
```
</details>

# Results


# How it works

Each task is defined by four files:

- `prompt.md` - prompt defining the task written in natural language
- `generate.py` - simulation code producing input data and ground truth (implement `_generate()`)
- `config.json` - difficulty parameters loaded in `generate.py` (each task currently defines easy, medium, hard levels)
- `metarubrics.json` - rubric templates instantiated from generated data  

The framework then automatically follows this pipeline:

1) `task.generate_task()` - calls `_generate()` to produce fresh input_data/ + ground_truth/
2) `task.populate_metarubrics()` - fills metarubrics (rubric templates) from ground truth
3) `task.generate_rubrics()` - creates instances of metarubrics and produces rubrics.json
4) `evaluator.run()` - sends to model, uses LLM-as-judge to judge output, saves results


# Contributing tasks

If you are considering contributing task from your domain, refer to [`CONTRIBUTING.md`](https://github.com/otheiner/PARAMETR-Bench/tree/repo-redesign?tab=contributing-ov-file#contributing-a-task).

# Citation

If you find this work useful or interesting, please consider citing it:

```bibtex
@misc{theiner2026parametr-bench,
  author = {Theiner, Ondrej},
  title  = {PARAMETR-Bench: Procedural And Rigorous Assessment using METaRubrics},
  year   = {2026},
  url    = {https://github.com/otheiner/PARAMETR-Bench}
}
```
