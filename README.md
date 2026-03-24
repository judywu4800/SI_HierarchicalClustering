This repository contains code for **[Hierarchical Clustering with Confidence](https://arxiv.org/abs/2512.06522)**, with a focus on
simulation studies, reproducible figure generation and example usage.


## Repository Structure

```text
.
├── src/                # Core implementations and helper functions
├── docs/               # Jupyter notebook containing example usage
├── simulations/        # Simulation scripts
├── scripts/            # Slurm scripts for batch execution
├── plotting/           # Plotting and figure-generation scripts
├── results/
│   ├── raw/             # Raw simulation outputs
│   └── figures/         # Generated figures
├── Makefile             # Reproducible workflow
├── environment.yml      # Conda environment specification
└── README.md
```

## Installation

This project requires both **Python** and **R**.  
We recommend using **conda** to reproduce the computational environment.

---

### 1. Create and activate the conda environment

From the repository root, run:

```bash
conda env create -f environment.yml
conda activate sihc
```
### 2. Install the GitHub-only R dependency
In addition to CRAN / conda-forge packages, some comparison in simulations relies on an `clusterpval` package
that must be installed from GitHub:
```bash
R -q -e 'remotes::install_github("lucylgao/clusterpval", upgrade="never")'
```

## Running Simulations

While most simulations were run on a computing cluster, we provide small-scale local executions that can be performed by running the following command:

```bash 
make fig1_sim
make validity_sim
make power_sim
make fig5_sim
make fig6_sim
make fig10_sim
make fig11_sim
```
Simulation outputs are saved under:
```text
results/raw/
```

## Generating Figures
```bash 
make figure1
make figure2
make figure3
make figure4
make figure5
make figure6
make figure7
make figure8
make figure9
make figure10
make figure11
```
Outputs are saved under:
```text
results/figures/
```
Note that full-scale simulations required to reproduce all figures were conducted on a computing cluster.
The corresponding Slurm scripts are provided in the `scripts/` directory and can be used directly
after setting up the appropriate software environment.


