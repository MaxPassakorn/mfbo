"""
mfbo: Multi-Fidelity Bayesian Optimization Surrogate Models
"""

from ._version import __version__

# ---- Posteriors ----
from .posteriors import MFDiagNormalPosterior, samples_to_mf_posterior

# ---- Neural ensemble surrogates ----
from .nn.ensembles import (
    MLPEnsemble,
    MFNNEnsemble,
    AGMFNetEnsemble,
    Ada2MFEnsemble,
)

# ---- GP-based surrogate ----
from .gp import CoKrigingAR1

__all__ = [
    "__version__",
    "MFDiagNormalPosterior",
    "samples_to_mf_posterior",
    "MLPEnsemble",
    "MFNNEnsemble",
    "AGMFNetEnsemble",
    "Ada2MFEnsemble",
    "CoKrigingAR1",
]