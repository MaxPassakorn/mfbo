from __future__ import annotations

"""
Co-kriging (multi-fidelity Gaussian processes).

This module provides an autoregressive AR(1) co-kriging model (Kennedy–O’Hagan)
implemented with GPyTorch and exposed through the BoTorch ``Model`` interface.

Classes
-------
CoKrigingAR1
    Autoregressive AR(1) co-kriging model for multi-fidelity data.
"""

from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import gpytorch
from botorch.models.model import Model
from botorch.posteriors import Posterior

from ..posteriors.diag_normal import MFDiagNormalPosterior
from .kernels import Kriging


class ExactKrigingGP(gpytorch.models.ExactGP):
    """
    Exact Gaussian process with a Kriging covariance.

    Notes
    -----
    This model uses:

    - a zero mean function (:class:`gpytorch.means.ZeroMean`)
    - a Kriging kernel (:class:`~mfbo.gp.kernels.Kriging`) wrapped by
      :class:`gpytorch.kernels.ScaleKernel`.
    """
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood, power: float = 1.5):
        super().__init__(train_x, train_y, likelihood)
        d = train_x.shape[-1]
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(Kriging(ard_num_dims=d, power=power))

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class CoKrigingAR1(Model, gpytorch.Module):
    r"""
    Autoregressive AR(1) co-kriging model for multi-fidelity Gaussian processes.

    The high-fidelity response is modeled as

    .. math::

       y_H(x) \approx \rho \, y_L(x) + \delta(x),

    where :math:`\rho` is an autoregressive scaling parameter and
    :math:`\delta(x)` is a discrepancy Gaussian process.

    Training procedure
    ------------------
    The implementation fits the model in three stages:

    1. Fit a low-fidelity GP on ``(X_low, y_low)``.
    2. Estimate :math:`\rho` by least squares using the low-fidelity GP
       predictions at the high-fidelity locations ``X_high``.
    3. Fit a discrepancy GP on residuals
       ``y_high - rho * y_low_pred(X_high)``.

    Multi-output support
    --------------------
    For vector-valued outputs (``D > 1``), the model constructs one pair of
    (low-fidelity GP, discrepancy GP) per output dimension. In that case,
    ``rho`` is a vector of length ``D``.

    Parameters
    ----------
    X_low
        Low-fidelity inputs with shape ``[nL, d]`` (or any array reshapeable
        to ``[nL, d]``).
    y_low
        Low-fidelity outputs with shape ``[nL]`` or ``[nL, D]``.
    X_high
        High-fidelity inputs with shape ``[nH, d]`` (or any array reshapeable
        to ``[nH, d]``).
    y_high
        High-fidelity outputs with shape ``[nH]`` or ``[nH, D]``.
    power
        Power parameter used by the Kriging kernel.

    Attributes
    ----------
    rho
        Autoregressive scaling parameters with shape ``[D]``.
    """

    def __init__(self, X_low: Tensor, y_low: Tensor, X_high: Tensor, y_high: Tensor, power: float = 1.5):
        super().__init__()

        Xl = X_low.view(X_low.shape[0], -1).to(torch.float64)
        Xh = X_high.view(X_high.shape[0], -1).to(torch.float64)

        yl = y_low if y_low.ndim == 2 else y_low.unsqueeze(-1)
        yh = y_high if y_high.ndim == 2 else y_high.unsqueeze(-1)

        if yl.shape[1] != yh.shape[1]:
            raise ValueError("Low- and high-fidelity must have same output dimension.")

        self._num_outputs = yl.shape[1]
        self.power = float(power)

        self.register_buffer("Xl", Xl)
        self.register_buffer("Xh", Xh)
        self.register_buffer("y_low", yl.to(torch.float64))
        self.register_buffer("y_high", yh.to(torch.float64))

        self.lik_l_list = nn.ModuleList()
        self.lik_h_list = nn.ModuleList()
        self.low_models = nn.ModuleList()
        self.delta_models = nn.ModuleList()

        for d in range(self._num_outputs):
            lik_l = gpytorch.likelihoods.GaussianLikelihood()
            lik_h = gpytorch.likelihoods.GaussianLikelihood()

            low_gp = ExactKrigingGP(self.Xl, self.y_low[:, d], lik_l, power=self.power)

            zeros = torch.zeros_like(self.y_high[:, d])
            delta_gp = ExactKrigingGP(self.Xh, zeros, lik_h, power=self.power)

            self.lik_l_list.append(lik_l)
            self.lik_h_list.append(lik_h)
            self.low_models.append(low_gp)
            self.delta_models.append(delta_gp)

        # rho per output (float64 for consistency with GP)
        self.register_parameter("rho", nn.Parameter(torch.ones(self._num_outputs, dtype=torch.float64)))

    @property
    def num_outputs(self) -> int:
        """
        Number of output dimensions.

        Returns
        -------
        int
            The number of outputs ``D`` modeled by the co-kriging surrogate.
            For scalar-valued problems, this is ``1``. For vector-valued
            problems, a separate low-fidelity GP and discrepancy GP are
            constructed for each output dimension.
        """
        return self._num_outputs

    def _fit_rho(self) -> None:
        r"""
        Estimate the autoregressive scaling parameters ``rho``.

        This routine computes a least-squares estimate of :math:`\rho` for each
        output dimension using the current low-fidelity GP predictions evaluated
        at the high-fidelity locations ``Xh``.

        Notes
        -----
        For output dimension :math:`j`, this solves

        .. math::

            \rho_j = \arg\min_{\rho} \| y_{H,j} - \rho \, \hat{y}_{L,j}(X_H) \|_2^2,

        which has the closed-form solution

        .. math::

            \rho_j = \frac{\hat{y}_{L,j}(X_H)^\top y_{H,j}}
                        {\hat{y}_{L,j}(X_H)^\top \hat{y}_{L,j}(X_H)}.

        The result is stored in-place in ``self.rho``.
        """
        eps = 1e-12
        for d in range(self._num_outputs):
            low_gp = self.low_models[d]
            low_gp.eval()
            with torch.no_grad():
                yLh = low_gp(self.Xh).mean
            yH = self.y_high[:, d]
            denom = (yLh @ yLh) + eps
            self.rho.data[d] = (yLh @ yH) / denom

    def fit(
        self,
        iters_per_stage: int = 100,
        stages: int = 3,
        lr_low: float = 0.05,
        lr_delta: float = 0.05,
        verbose: bool = True,
    ) -> None:
        """
        Fit the AR(1) co-kriging model.

        The training alternates between (i) fitting the low-fidelity GPs,
        (ii) updating ``rho``, and (iii) fitting the discrepancy (delta) GPs.

        Parameters
        ----------
        iters_per_stage
            Number of optimizer iterations used to fit each GP per stage.
        stages
            Number of outer stages. Each stage refits the low-fidelity GPs, updates
            ``rho``, and refits the discrepancy GPs.
        lr_low
            Learning rate for the low-fidelity GP optimizers.
        lr_delta
            Learning rate for the discrepancy GP optimizers.
        verbose
            If True, prints per-iteration losses (every 10 iterations) and the
            current ``rho`` values.

        Notes
        -----
        This method performs in-place training and does not return anything.
        After fitting, you can call :meth:`predict` or :meth:`posterior`.
        """
        for s in range(stages):
            if verbose:
                print(f"\n=== Stage {s+1}/{stages} ===")

            # (A) train low-fidelity GPs
            for d in range(self._num_outputs):
                low_gp, lik = self.low_models[d], self.lik_l_list[d]
                low_gp.train(); lik.train()
                opt = torch.optim.AdamW(low_gp.parameters(), lr=lr_low)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, low_gp)

                for it in range(iters_per_stage):
                    opt.zero_grad()
                    out = low_gp(self.Xl)
                    loss = -mll(out, low_gp.train_targets)
                    loss.backward()
                    opt.step()
                    if verbose and (it + 1) % 10 == 0:
                        print(f"[LF {d}] iter {it+1:03d}/{iters_per_stage}  loss={loss.item():.3e}")

            # (B) fit rho
            self._fit_rho()
            if verbose:
                rhos = ", ".join([f"{float(self.rho[d]):.4f}" for d in range(self._num_outputs)])
                print(f"[rho] [{rhos}]")

            # (C) train delta GPs on residuals
            for d in range(self._num_outputs):
                low_gp = self.low_models[d]
                delta_gp, lik_h = self.delta_models[d], self.lik_h_list[d]

                low_gp.eval()
                with torch.no_grad():
                    yLh = low_gp(self.Xh).mean
                    delta = self.y_high[:, d] - self.rho[d] * yLh
                    delta_gp.set_train_data(self.Xh, delta, strict=False)

                delta_gp.train(); lik_h.train()
                opt = torch.optim.AdamW(delta_gp.parameters(), lr=lr_delta)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik_h, delta_gp)

                for it in range(iters_per_stage):
                    opt.zero_grad()
                    out = delta_gp(self.Xh)
                    loss = -mll(out, delta_gp.train_targets)
                    loss.backward()
                    opt.step()
                    if verbose and (it + 1) % 10 == 0:
                        print(f"[Δ {d}] iter {it+1:03d}/{iters_per_stage}  loss={loss.item():.3e}")

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predict the mean and standard deviation at query points ``X``.

        Parameters
        ----------
        X
            Query points with shape ``[..., q, d]`` or ``[N, d]``.
            The method flattens the leading dimensions internally and restores the
            original batch shape in the outputs.

        Returns
        -------
        mean
            Predictive mean with shape ``[..., q, D]`` (or ``[N, D]`` if ``X`` is
            two-dimensional).
        std
            Predictive standard deviation with shape ``[..., q, D]`` (or ``[N, D]``).

        Notes
        -----
        The predictive moments are combined as

        - ``mean = rho * mean_low + mean_delta``
        - ``var  = (rho**2) * var_low + var_delta``

        assuming independence between the low-fidelity and discrepancy GPs.
        """
        Xf = X.reshape(-1, X.shape[-1]).to(torch.float64)

        means, vars_ = [], []
        for d in range(self._num_outputs):
            low_gp = self.low_models[d]
            delta_gp = self.delta_models[d]
            rho = self.rho[d]

            low_gp.eval(); delta_gp.eval()
            with gpytorch.settings.fast_pred_var():
                low_out = low_gp(Xf)
                delta_out = delta_gp(Xf)

            m = rho * low_out.mean + delta_out.mean
            v = (rho ** 2) * low_out.variance + delta_out.variance
            means.append(m)
            vars_.append(v)

        mean = torch.stack(means, dim=-1)
        var = torch.stack(vars_, dim=-1).clamp_min(0.0)
        std = var.sqrt()

        out_shape = X.shape[:-1] + (self._num_outputs,)
        return mean.view(*out_shape), std.view(*out_shape)

    def posterior(self, X: Tensor, **kwargs) -> Posterior:
        """
        Return a BoTorch-compatible posterior at ``X``.

        Parameters
        ----------
        X
            Query points with shape ``[..., q, d]`` or ``[N, d]``.
        **kwargs
            Unused. Included for BoTorch API compatibility.

        Returns
        -------
        botorch.posteriors.Posterior
            A diagonal Normal posterior with batch shape matching ``X`` and event
            shape ``D`` (number of outputs). The returned object is an instance of
            :class:`~mfbo.posteriors.diag_normal.MFDiagNormalPosterior`.
        """
        mean, std = self.predict(X)
        return MFDiagNormalPosterior(mean=mean, std=std)