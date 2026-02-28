# Installation

`mfbo` is a Python library for multi-fidelity surrogate modeling and Bayesian
optimization, compatible with BoTorch.

---

## Requirements

- Python **≥ 3.12** (required by `mfbo`)
- PyTorch **≥ 2.5**
- GPyTorch **≥ 1.15**
- BoTorch **≥ 0.16**
- tqdm **≥ 4.67**

> mfbo depends on the PyTorch + BoTorch ecosystem.  
> Install **PyTorch first** (CPU or CUDA), then install `mfbo`.

---

## Recommended: conda environment

Create and activate a clean environment:

```bash
conda create -n mfbo python=3.12
conda activate mfbo
```

## Step 1 - Install PyTorch

Install PyTorch following the official instructions (CPU or CUDA build):
- PyTorch "Get Started" page: https://pytorch.org/get-started/locally/

## Step 2 - Install mfbo

### Option A: Install from PyPI
```bash
pip install mfbo
```
### Option B: Install from conda-forge
```bash
conda install -c conda-forge mfbo
```
### Option C: (recommended for development): Install from source
```bash
git clone https://github.com/MaxPassakorn/mfbo.git
cd mfbo
pip install -e .
```