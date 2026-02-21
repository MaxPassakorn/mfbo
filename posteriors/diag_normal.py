import torch
from botorch.posteriors import Posterior
from botorch.sampling.get_sampler import GetSampler

class MFDiagNormalPosterior(Posterior):
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
    if samples.ndim == 3:
        samples = samples.unsqueeze(-2)  # [B,M,1,E]
    if samples.ndim != 4:
        raise RuntimeError(f"Expected [B,M,q,E] (or [B,M,E]), got {tuple(samples.shape)}")
    mean = samples.mean(dim=1)
    std = samples.std(dim=1, unbiased=False).clamp_min(1e-12)
    return MFDiagNormalPosterior(mean=mean, std=std)