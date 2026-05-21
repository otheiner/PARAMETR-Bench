[![DOI](https://github.com/user-attachments/assets/799b8ec5-cc08-4e1b-a182-4e784fa78244)](https://doi.org/10.5281/zenodo.20076421)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow?style=flat)](https://huggingface.co/spaces/otheiner/PARAMETR-Bench_demo)
[![Blog](https://img.shields.io/badge/Blog-PARAMETR--Bench-2A6A9E?style=flat&logoColor=white)](https://otheiner.github.io/PARAMETR-Bench-blog/)
[![CI](https://github.com/otheiner/PARAMETR-Bench/actions/workflows/ci.yml/badge.svg)](https://github.com/otheiner/PARAMETR-Bench/actions/workflows/ci.yml)

<h1 align="center">📊 PARAMETR-Bench</h1>

<p align="center">
  <b>P</b>rocedural <b>A</b>nd <b>R</b>igorous <b>A</b>ssessment using <b>MET</b>a<b>R</b>ubrics: A framework for building contamination-free scientific benchmarks for agentic LLM evaluation with deterministically generated rubrics.
</p>

<p align="center">
    <br>
  <img src="https://github.com/user-attachments/assets/a20e775e-8a33-4f3d-b4d3-8f16318c9ec3" width="900" title="Preview of some input and ground-truth reference files generated procedurally.">
    <br><br>
      <i> Figure: Preview of a few input and ground truth files from tasks in the PARAMETR-Bench generated procedurally.</i>
</p>

# What is this?

PARAMETR-Bench is a framework for evaluating LLMs on realistic scientific analysis workflows using procedurally generated tasks with perfectly synchronized rubrics.

Every run produces fresh multimodal instances (plots, CSVs, data tables) from a controlled generative process. The key innovation is source-grounded [metarubrics](https://otheiner.github.io/PARAMETR-Bench-blog/#metarubrics-design-guidance), which are rubric templates that are automatically populated directly from the generated ground truth. This guarantees that evaluation criteria are always perfectly aligned with the task data, eliminating rubric drift by construction.

Because tasks are generated from a fixed distribution controlled by difficulty parameters and random seeds, the framework enables statistically rigorous evaluation. Running multiple independent seeds at the same difficulty level turns each evaluation into a set of independent trials, allowing proper confidence intervals, per-metarubric breakdowns, and more reliable model comparisons.

The repository includes several tasks inspired by landmark discoveries in particle physics and cosmology, such as invariant mass reconstruction and Cepheid variable calibration. However, any scientific process with a simulatable generating distribution can become a task — physics, mathematics, chemistry, biology, climate science, and beyond.

Try the interactive task generation online or read the detailed technical blog post about the framework: 

<p align="center">
  <a href="https://huggingface.co/spaces/otheiner/PARAMETR-Bench_demo"><img src="https://img.shields.io/badge/🤗_Try_the_demo-FFD21E?style=for-the-badge" alt="Try the demo"/></a>&nbsp;&nbsp;<a href="https://otheiner.github.io/PARAMETR-Bench-blog"><img src="https://img.shields.io/badge/📖_Read_the_post-444444?style=for-the-badge" alt="Read the blog post"/></a>
</p>

# Results

Initial evaluations across frontier models are in progress and will be published here soon to demonstarte the framework.

# Motivation 

I am a particle physicist who recently got into LLM evaluation and I built this as a passion project to test if models can actually do science. I tried to design the architecture to specifically address major issues in the field, such as benchmark contamination, rubric drift, and the stochastic nature of LLM responses. If you have any questions, comments, suggestions, or you would be interested in contributing, don't hesitate to reach out to me [here](https://otheiner.github.io/#contact).

# Quick start

Clone repo and install dependencies:

```bash
git clone https://github.com/otheiner/PARAMETR-Bench.git
cd PARAMETR-Bench
pip install .
```

## API models

Run the benchmark and produce your own results (you can plug any models of your choice supported by `litellm`). How to do this is described bellow. To use API models, add your keys:

```bash
cp .env.example .env   # fill in your API keys to .env
```

Framework allows non-agentic (no tools allowed) and agentic (allows running python scripts) evaluation. Details on how to run actual evaluation with or without agent is described in the following sections.

## Model evaluation

For **non-agentic evaluation** simply run:

```bash
python run.py --model gemini/gemini-3.1-flash-lite-preview \
              --judge  gemini/gemini-2.5-flash \
              --difficulty medium \
              --seeds 0 1
```
This sends LLM prompt and all the data in one message and LLM has one shot to return the result.

**Agentic evaluation** enables performing more realistic scientific tasks. Agent gets only the prompt and list of files to work with and it has then the ability to use tools to inspect these files and analyze them by python, which is executed in safe Docker sandbox environment without access to the Internet. First of all, run docker daemon on your machine. Description of the sandbox environment and its Dockerfile is located in `sandbox` folder. 

Build the sandbox Docker image and run benchmark using flag `--agentic`:

```bash
docker build -t benchmark-sandbox sandbox/
python run.py --model gemini/gemini-3.1-flash-lite-preview \
              --judge  gemini/gemini-2.5-flash \
              --difficulty medium \
              --agentic \
              --seeds 0 1
```

## Resuming failed evals

API failures outside your control shouldn't cost you a full re-run. Each run creates a folder in `results/` named `<run_ID>_<date>_<time>`, where `<run_ID>` is a 6-character alphanumeric string. To resume an interrupted run:

```bash
python run.py --continue-run run_ID
```

The framework checks that the git commit and working tree match the state at the start of the run. If anything has changed, resuming may produce results that are not comparable to the original run. If you want to continue anyway:

```bash
python run.py --continue-run run_ID:force
```

**What gets preserved**

Completed tasks are flushed to disk as they finish, so they survive any kind of failure. For the in-flight sequence (mid-agentic loop, or when judge fails), the framework snapshots state on caught failures (API errors, timeouts, exceptions) and can resume from the exact turn where it failed — both the LLM and the Docker sandbox are stateless, so replaying the message history and remounting the working directory fully reconstructs the run. A hard kill (Ctrl-C, crash, power loss) still preserves all completed sequences, but the in-flight one restarts from turn zero.

**Repository state**

Each run stores the git commit hash at start time. If the repository has changed since, you'll be asked to check out the original commit before resuming — this keeps the initial run and the continuation consistent.

## No API keys? No problem!
   
You can just inspect generated data (no model calls). The [Hugging Face Space](https://huggingface.co/spaces/otheiner/PARAMETR-Bench_demo) does the same in-browser but may hit resource limits on heavier tasks. To run the task generation locally just do:

```bash
python run.py --validate-only
```

You can also run full evals with local models via Ollama. You need to start the Ollama server and pass an Ollama model to `run.py`, e.g. `--model ollama/qwen2.5:3b`.

# Contributing tasks

If you are considering contributing task from your domain, refer to [`CONTRIBUTING.md`](CONTRIBUTING.md#contributing-a-task).

# Citation

If you use PARAMETR-Bench in your work, please cite this software as:

> Theiner, O. (2026). *PARAMETR-Bench: Procedural And Rigorous Assessment using METaRubrics* [Software]. Zenodo. https://doi.org/10.5281/zenodo.20076421

Or use the BibTeX entry:

```bibtex
@software{theiner_parametr_bench_2026,
  author       = {Theiner, Ondrej},
  title        = {{PARAMETR-Bench: Procedural And
                Rigorous Assessment using METaRubrics}},
  month        = may,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20076421},
  url          = {https://doi.org/10.5281/zenodo.20076421}
}
```
