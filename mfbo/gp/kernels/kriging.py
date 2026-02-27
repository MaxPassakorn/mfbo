from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn
import gpytorch


class Kriging(gpytorch.kernels.Kernel):
    r"""
    Power-exponential (generalized exponential) kernel.

    .. math::

       k(x, x') = \exp\left(
           -\sum_{j=1}^{d} \theta_j \, \lvert x_j - x'_j \rvert^{p}
       \right)

    Notes
    -----
    - Uses ARD parameters :math:`\theta_j > 0` (learned directly).
    - The exponent :math:`p` (``power``) is treated as a fixed hyperparameter.
    """
    has_lengthscale = False
    is_stationary = True

    def __init__(self, ard_num_dims: int, power: float = 1.5, **kwargs):
        super().__init__(ard_num_dims=ard_num_dims, **kwargs)
        if ard_num_dims is None or int(ard_num_dims) <= 0:
            raise ValueError("ard_num_dims must be a positive integer.")

        self.power = float(power)

        init_theta = torch.ones(int(ard_num_dims))
        self.register_parameter("raw_theta", nn.Parameter(init_theta))
        self.register_constraint("raw_theta", gpytorch.constraints.GreaterThan(0.0))

    @property
    def theta(self) -> Tensor:
        return self.raw_theta_constraint.transform(self.raw_theta)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        diff = torch.abs(x1.unsqueeze(-2) - x2.unsqueeze(-3)).pow(self.power)  # [..., n, m, d]
        theta = self.theta.view(*([1] * (diff.ndim - 1)), -1)                  # broadcast to [...,1,1,d]
        K = torch.exp(-(diff * theta).sum(dim=-1))                             # [..., n, m]
        if diag:
            return torch.diagonal(K, dim1=-2, dim2=-1)
        return K