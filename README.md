![CI](https://github.com/otheiner/physics-analysis-benchmarks/actions/workflows/ci.yml/badge.svg)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![LiteLLM](https://img.shields.io/badge/LiteLLM-blueviolet?style=for-the-badge)

# Physics analysis benchmark 📊

A procedural framework for contamination-resistant evaluation of LLMs multimodal agentic capabilities on real scientific analysis workflows — with perfectly synchronized rubrics and statistically robust multi-seed evaluation.

## What is this?

XXXX is a framework for evaluating LLMs on realistic scientific analysis workflows using procedurally generated tasks with perfectly synchronized rubrics.

Every run produces fresh multimodal instances (plots, CSVs, data tables) from a controlled generative process. The key innovation is source-grounded metarubrics: rubric templates that are automatically populated directly from the generated ground truth. This guarantees that evaluation criteria are always perfectly aligned with the task data, eliminating rubric drift by construction.

Because tasks are generated from a fixed distribution controlled by difficulty parameters and random seeds, the framework enables statistically rigorous evaluation. Running multiple independent seeds at the same difficulty level turns each evaluation into a set of independent trials, allowing proper confidence intervals, per-metarubric breakdowns, and more reliable model comparisons.

The repository includes several tasks inspired by landmark discoveries in particle physics and cosmology, such as invariant mass reconstruction and Cepheid variable calibration.

## The core idea

Traditional benchmarks rely on fixed test sets that leak into training data, becoming contaminated or saturated. Common solutions are hiding test sets or constantly adding new questions. These approaches either sacrifice benchmark transparency or require unsustainable effort.

Procedural generation solves leakage by creating fresh instances every run. But it introduces a new problem: keeping rubrics aligned with dynamically generated data, especially in multi-step scientific tasks.

Our solution is to use the same generating process that creates the task data to also instantiate the rubrics. We call these templates metarubrics. Every rubric criterion is mathematically guaranteed to match the generated instance  by construction, not by validation. Templating allows us also automatically generate variable number of atomic rubric criteria for repeated data extraction, which is common in scientific data analyses.

Since rubric criteria contain specific numerical values drawn from the simulation, they cannot be gamed by memorising fixed evaluation criteria. A model must solve each instance on its own merits.

## Motivation 

I am a particle physicist who recently got into LLM evaluation and I built this as a passion project to test if models can actually do science. I designed the architecture to specifically address major issues in the field, such as benchmark contamination, rubric drift, and the stochastic nature of LLM responses. I welcome any suggestions, feedback, and pull requests from other scientists or AI researchers. Contact me [here](https://otheiner.github.io/#contact).


# Quick start

Clone repo and install dependencies:

```bash
git clone https://github.com/otheiner/physics-analysis-benchmarks
cd physics-analysis-benchmarks
pip install -r requirements.txt
```

The framework uses [litellm](https://github.com/BerriAI/litellm), supporting both local models via [Ollama](https://ollama.com) and API-based models. If you want to use local models make sure that your `ollama` server is installed and running (see Ollama link). To use API models, add your keys:

```bash
cp .env.example .env   # fill in your API keys to .env
```

Validate task generation without API calls and inspect generated data locally:

```bash
python run.py --validate-only
```

Run the benchmark and produce your results (you can plug any models of your choice supported by `litellm`). Framework allows non-agentic (no tools allowed) and agentic (allows running python scripts) evaluation. See details in corresponding subsections below.

## Non-agentic evaluation

<details>
<summary><strong>Expand here</strong></summary>

For non-agentic evaluation simply run:

```bash
python run.py --models gemini/gemini-3.1-flash-lite-preview \
              --judge  gemini/gemini-2.5-flash \
              --difficulty medium \
              --seeds 0 1
```
</details>

## Agentic evaluation

<details>
<summary><strong>Expand here</strong></summary>

Agentic evaluation enables running python with a few allowed python libraries specified in `sandbox/requirements.txt`. Python is executed in safe Docker sandbox environment without access to the Internet, memory-capped to 512 MB, not allowing writing `.pyc` files. **First of all, run docker daemon on your machine!** Build the sandbox Docker image and run benchmark using flag `--agentic`:

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
- `generate.py` - simulation code producing input data and ground truth
- `config.json` - difficulty parameters loaded in `generate.py` (each task currently defines easy, medium, hard levels)
- `metarubrics.json` - rubric templates instantiated from generated data  

The framework then automatically follows this pipeline:

1) `task.generate_task()` - generates fresh input_data/ + ground_truth/
2) `task.populate_metarubrics()` - fills metarubrics (rubric templates) from ground truth
3) `task.generate_rubrics()` - creates instances of metarubrics and produces rubrics.json
4) `evaluator.run()` - sends to model, uses LLM-as-judge to judge output, saves results


# Contributing tasks

If you are considering contributing task from your domain, refer to [`CONTRIBUTING.md`](https://github.com/otheiner/physics-analysis-benchmarks/blob/repo-redesign/CONTRIBUTING.md).


# Citation

If you find this work useful or interesting, please consider citing it:

```bibtex
@misc{---------,
  author = {Theiner, Ondrej},
  title  = {------------},
  year   = {------------},
  url    = {------------}
}
```
