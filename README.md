# Physics analysis benchmarks

This repository is a collection of original simulation-based physics analysis tasks with known ground truth that I developed. They are designed for human learners and multimodal LLM benchmarking. Can you perform these analyses correctly and beat the LLMs?

Even if the tasks are grounded in real analyses, the structure of the input data might make them look like a toy example. However, this structure is excellent for multimodal LLM benchmarking for several reasons:

- Multi-step domain specific reasoning
- Parsing large CSV or PDF files (the latter is especially tricky with big tables, since LLMs usually need an intermediate step of converting them to a text file, which is prone to errors)
- Extracting data from images with features that are not easy for LLMs to retrieve
- Data are simulated with known parameters, so the ground truth is known and well-defined

Tasks are designed such that the user can define the size of the generated input data and they allow tweaking parameters, such as noise or simulating reading error, which can easily make the task more difficult. Randomness is introduced into the data generation pipeline which makes it easy to generate multiple different datasets.

I tested these tasks and each of them had at least one aspect that proved to be challenging for current LLMs. However, each analysis task could potentially be solved "by hand" without the need for complicated image extraction tools, which makes them suitable for human learners.

# Tasks

This repository contains a few tasks but more may be added in the future. Details of each task are written in their respective folders together with a small sample of the data that the data generation pipeline produces and the ground truth answers. To make these tasks challenging for LLMs, sample sizes have to be larger than the snippets shown in this repository. This can be easily tweaked in the notebooks and data can be produced locally. Important parameters of each input file are stored in the pandas dataframe, which gives the user immediate access to the ground truth values.

1. **Estimating Hubble's Constant**
This task requires analyzing spectroscopic data and identifying redshifts of fictitious galaxies. This information is then combined with photometric information about Cepheid variables in these galaxies, which allows distance calibration. The goal of the task is to use these data to estimate the local rate of expansion of the Universe — Hubble's constant.
