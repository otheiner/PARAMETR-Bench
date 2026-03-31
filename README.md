# Physics analysis benchmarks 📊

This repository is a collection of original simulation-based physics analysis tasks with known ground truth that I developed. They are designed for human learners and multimodal LLM benchmarking. Can you perform these analyses correctly and beat the LLMs?

Even if the tasks are grounded in real analyses, the structure of the input data might make them look like a toy example. However, this structure is excellent for multimodal LLM benchmarking for several reasons:

- Multi-step domain-specific reasoning
- Parsing large CSV or PDF files (the latter is especially tricky with big tables, since LLMs usually need an intermediate step of converting them to a text file, which is prone to errors)
- Extracting data from images with features that are not easy for LLMs to retrieve
- Data are simulated with known parameters, so the ground truth is known and well-defined
- Data simulate real physics using random seeds, so these exact data have never been seen by LLMs 

Tasks are designed such that the user can define the size of the generated input data and they allow tweaking parameters, such as noise or simulating reading error, which can easily make the task more difficult. Randomness is introduced into the data generation pipeline which makes it easy to generate multiple different datasets. Ground truth answer is generated together with the dataset.

I tested these tasks and each of them had at least one aspect that proved to be challenging for current LLMs. However, each analysis task could potentially be solved "by hand" without the need for complicated image extraction tools, which makes them suitable for human learners.

# Tasks

This repository contains a few tasks but more may be added in the future. Details of each task are written in their respective folders together with a small sample of the data that the data generation pipeline produces and the ground truth answers. To make these tasks challenging for LLMs, sample sizes have to be larger than the snippets shown in this repository. This can be easily tweaked in the notebooks and data can be produced locally. Important parameters of each input file are stored in the pandas dataframe, which gives the user immediate access to the intermediate ground truth values.

1. **Cepheid period-luminosity calibration** 🪐

This analysis focuses on the well-known relation between absolute magnitude of Cepheid variables and their period, recreating (though not exactly) the discovery by Henrietta Swan Leavitt, whose foundational contribution to observational cosmology was never recognized with a Nobel Prize. The task requires combining Hubble's law, spectroscopic data of galaxies, and photometric data about Cepheid variables. Beyond basic concepts from astrophysics, it tests methods of physical data analysis such as template cross-correlation in log-λ space.

2. **Estimating Hubble's Constant** 🪐

This task mimics the analysis done by Edwin Hubble which led to one of the most significant discoveries in observational cosmology. It is essentially an inverse approach to the Cepheid calibration from the previous task, with a different visual representation of the spectra. It requires analyzing spectroscopic data and identifying redshifts of fictitious galaxies. This information is then combined with photometric information about Cepheid variables in these galaxies, which allows distance calibration. The goal of the task is to use the data to estimate the local rate of expansion of the Universe — Hubble's constant.

3. **Reconstruction of invariant mass of an unknown particle** ⚛️

A toy example simulating a real particle physics analysis pipeline — from raw detector data reconstruction all the way to the statistical "discovery" of an unknown hypothetical particle. Data reconstruction requires extracting track curvature of secondary particles inside a tracking detector in order to determine their momenta. In this simplified setup, track identification can be done visually by a human learner using physical intuition, but proves surprisingly difficult for LLMs. Momentum information is then combined with calorimetric energy measurements to reconstruct the four-momenta of the secondary particles. This allows building a histogram of the reconstructed invariant mass, to which a composed signal-plus-background model is fitted. The mass and width of the unknown particle are extracted from the fit parameters, and the statistical significance of the signal over background is estimated.

4. **Lissajous figures and quality assurance** ⚡

This task puts the solver in the position of a quality assurance expert assessing AC power supplies produced by manufacturing machines in a factory. It requires combining tabulated production data with oscilloscope images showing Lissajous figures. The figures are used to determine the frequency of each tested power supply, which is then combined with knowledge of available inductor coils to assess whether the unit meets the impedance specification. The multi-step pipeline — frequency extraction from images, impedance calculation, QA form completion, daily failure histogram, and per-machine failure rate analysis — makes this task deceptively challenging for LLMs despite the straightforward underlying physics.
