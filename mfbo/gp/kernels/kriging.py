from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn
import gpytorch


class Kriging(gpytorch.kernels.Kernel):
    r"""
    Power-exponential (generalized exponential) covariance kernel.

    This kernel implements the ARD (Automatic Relevance Determination)
    power-exponential form:

    .. math::

       k(x, x') = \exp\left(
           -\sum_{j=1}^{d} \theta_j \, \lvert x_j - x'_j \rvert^{p}
       \right),

    where:

    - :math:`\theta_j > 0` are dimension-wise scaling parameters
      (learned during training),
    - :math:`p` is a fixed exponent controlling smoothness.

    Notes
    -----
    - When ``p = 2``, this reduces to an RBF-like (squared exponential) kernel.
    - When ``p = 1``, it corresponds to an exponential kernel.
    - Intermediate values ``1 < p < 2`` provide adjustable smoothness.
    - The parameters :math:`\theta_j` behave similarly to inverse
      lengthscales raised to the power ``p``.

    Parameters
    ----------
    ard_num_dims : int
        Number of input dimensions (``d``). Must be positive.
        One ARD parameter :math:`\theta_j` is created per dimension.
    power : float, default=1.5
        Exponent :math:`p` controlling smoothness.
    **kwargs
        Additional keyword arguments passed to
        :class:`gpytorch.kernels.Kernel`.

    Attributes
    ----------
    theta : torch.Tensor
        Positive ARD parameters of shape ``[d]``.
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
        """
        Positive ARD parameters.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``[d]`` containing the positive scaling
            parameters :math:`\theta_j`.
        """
        return self.raw_theta_constraint.transform(self.raw_theta)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        r"""
        Compute the covariance matrix between ``x1`` and ``x2``.

        Parameters
        ----------
        x1 : torch.Tensor
            Input tensor of shape ``[..., n, d]``.
        x2 : torch.Tensor
            Input tensor of shape ``[..., m, d]``.
        diag : bool, default=False
            If ``True``, return only the diagonal of the covariance matrix.
        **params
            Unused. Included for GPyTorch API compatibility.

        Returns
        -------
        torch.Tensor
            If ``diag=False``:
                Covariance matrix of shape ``[..., n, m]``.

            If ``diag=True``:
                Diagonal elements of shape ``[..., min(n, m)]``.

        Notes
        -----
        The covariance is computed as:

        .. math::

            k(x, x') = \exp\left(
                -\sum_{j=1}^{d}
                \theta_j |x_j - x'_j|^p
            \right).

        The kernel is stationary and depends only on pairwise differences.
        """
        diff = torch.abs(x1.unsqueeze(-2) - x2.unsqueeze(-3)).pow(self.power)  # [..., n, m, d]
        theta = self.theta.view(*([1] * (diff.ndim - 1)), -1)                  # broadcast to [...,1,1,d]
        K = torch.exp(-(diff * theta).sum(dim=-1))                             # [..., n, m]
        if diag:
            return torch.diagonal(K, dim1=-2, dim2=-1)
        return K