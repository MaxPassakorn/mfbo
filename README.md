# mfbo

"mfbo" is a Python library for "multi-fidelity surrogate modeling and Bayesian optimization". It provides neural-network ensemble surrogates and Gaussian-process co-kriging models that are fully compatible with "BoTorch".  The library is designed for scientific and engineering optimization problems.


**mfbo** is a Python library for **multi-fidelity surrogate modeling and Bayesian optimization**.
It provides neural-network ensemble surrogates and Gaussian-process co-kriging models
that are fully compatible with **BoTorch**.

The library is designed for scientific and engineering optimization problems
where high-fidelity evaluations are expensive and multiple fidelity levels are available.

---

## Reference

This library implements methods described in:

> **Passakorn Paladaechanan et al.**  
> *Adaptive Gated Multi-Fidelity Neural Networks for Bayesian Optimization*  
> **Ocean Engineering**, 2025  
> https://www.sciencedirect.com/science/article/pii/S002980182502997X

If you use this library in academic work, please cite the above paper.

---

## Features

- Multi-fidelity neural surrogate ensembles
  - `MLPEnsemble`
  - `MFNNEnsemble`
  - `AGMFNetEnsemble`
  - `Ada2MFEnsemble`
- Gaussian-process surrogate models
  - Kriging
  - AR(1) Co-Kriging (Kennedy–O’Hagan model)
- Native **BoTorch** compatibility
  - All models expose a `posterior()` method
- Supports multi-output objectives
- Designed for Bayesian optimization workflows

---

## Dependencies

`mfbo` is built on the PyTorch Bayesian optimization ecosystem:

- **PyTorch** ≥ 2.5
- **GPyTorch** ≥ 1.15
- **BoTorch** ≥ 0.16
- **tqdm** ≥ 4.67

Please install PyTorch separately according to your system
(CPU or CUDA) before installing `mfbo`.

## Installation

### 1. Install PyTorch
Install PyTorch first (CPU or CUDA version depending on your system):

https://pytorch.org/get-started/locally/

### 2. Install mfbo

#### pip (PyPI)

```bash
pip install mfbo
```

#### conda (conda-forge)

```bash
conda install -c conda-forge mfbo
```

## Contact

For questions related to the implementation or the associated research,  
please contact:

**Passakorn Paladaechanan**  
Email: p.paladaechanan@gmail.com