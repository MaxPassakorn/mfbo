"""
Diagonal normal posterior for multi-fidelity ensemble surrogates.

This module defines :class:`MFDiagNormalPosterior`, a lightweight
BoTorch-compatible posterior representing a factorized (diagonal)
Gaussian distribution over outputs.

It is primarily used to convert ensemble predictions of shape
``[B, M, q, E]`` into a Gaussian posterior with:

- mean  : ``[B, q, E]``
- std   : ``[B, q, E]``

where:

- ``B`` = broadcast batch dimension
- ``M`` = ensemble size
- ``q`` = number of query points
- ``E`` = number of outputs

The posterior assumes independence across outputs and query points
(i.e., diagonal covariance).
"""

import torch
from botorch.posteriors import Posterior
from botorch.sampling.get_sampler import GetSampler

class MFDiagNormalPosterior(Posterior):
    """
    Diagonal Gaussian posterior for multi-output ensemble models.

    This posterior represents a factorized normal distribution:

    .. math::

        y ~ \\mathcal{N}(\\mu, \\mathrm{diag}(\\sigma^2))

    where both the mean and standard deviation have shape
    ``[..., q, m]``.

    Parameters
    ----------
    mean : torch.Tensor
        Mean tensor of shape ``[B, q, E]`` (or with arbitrary leading
        batch dimensions).
    std : torch.Tensor
        Standard deviation tensor of the same shape as ``mean``.
        Values are clamped to be at least ``1e-12`` for numerical stability.

    Raises
    ------
    RuntimeError
        If ``mean`` and ``std`` do not have identical shapes, or if
        the tensor rank is less than 2.

    Notes
    -----
    - Assumes independence across query points and outputs
      (diagonal covariance structure).
    - Compatible with BoTorch sampling via :class:`IIDNormalSampler`.
    - Intended for ensemble-based uncertainty quantification where
      mean and variance are estimated empirically.
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        if mean.shape != std.shape:
            raise RuntimeError("mean and std must have the same shape")
        if mean.ndim < 2:
            raise RuntimeError("mean must have shape [..., q, m]")
        self._mean = mean
        self._std = std.clamp_min(1e-12)

    @property
    def device(self) -> torch.device:
        return self._mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self._mean.dtype

    @property
    def mean(self): return self._mean

    @property
    def variance(self): return self._std.pow(2)

    @property
    def stddev(self): return self._std

    @property
    def batch_shape(self): return torch.Size(self._mean.shape[:-2])

    @property
    def event_shape(self): return torch.Size(self._mean.shape[-2:])

    @property
    def base_sample_shape(self): return self.batch_shape + self.event_shape

    def _extended_shape(self, sample_shape=torch.Size()):
        return sample_shape + self.batch_shape + self.event_shape

    def rsample_from_base_samples(self, sample_shape, base_samples):
        return self._mean.expand_as(base_samples) + self._std.expand_as(base_samples) * base_samples

    def rsample(self, sample_shape=torch.Size()):
        """
        Draw reparameterized samples from the posterior.

        Parameters
        ----------
        sample_shape : torch.Size, optional
            Shape of the sample batch.

        Returns
        -------
        torch.Tensor
            Samples of shape:
            ``sample_shape + batch_shape + event_shape``.
        """
        eps = torch.randn(self._extended_shape(sample_shape), device=self._mean.device, dtype=self._mean.dtype)
        return self.rsample_from_base_samples(sample_shape, eps)

    @property
    def batch_range(self):
        return (0, -2)

@GetSampler.register(MFDiagNormalPosterior)
def _get_sampler_mf_diag_normal_posterior(posterior, sample_shape, **kwargs):
    from botorch.sampling.normal import IIDNormalSampler
    return IIDNormalSampler(sample_shape=sample_shape)


def samples_to_mf_posterior(samples: torch.Tensor) -> MFDiagNormalPosterior:
    """
    Convert ensemble samples to a diagonal Gaussian posterior.

    Parameters
    ----------
    samples : torch.Tensor
        Ensemble predictions of shape:

        - ``[B, M, q, E]`` (standard case), or
        - ``[B, M, E]`` (interpreted as ``q = 1``)

        where:

        - ``B`` = broadcast batch dimension
        - ``M`` = ensemble size
        - ``q`` = number of query points
        - ``E`` = number of outputs

    Returns
    -------
    MFDiagNormalPosterior
        Posterior with:

        - mean = ensemble mean across dimension ``M``
        - std  = ensemble standard deviation across dimension ``M``

    Raises
    ------
    RuntimeError
        If the input tensor does not have 3 or 4 dimensions.

    Notes
    -----
    The posterior assumes independence across outputs and query points.
    The standard deviation is computed with ``unbiased=False`` and
    clamped to a minimum of ``1e-12`` for numerical stability.
    """
    if samples.ndim == 3:
        samples = samples.unsqueeze(-2)  # [B,M,1,E]
    if samples.ndim != 4:
        raise RuntimeError(f"Expected [B,M,q,E] (or [B,M,E]), got {tuple(samples.shape)}")
    mean = samples.mean(dim=1)
    std = samples.std(dim=1, unbiased=False).clamp_min(1e-12)
    return MFDiagNormalPosterior(mean=mean, std=std)