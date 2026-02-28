from __future__ import annotations

"""
Deterministic MLP ensemble surrogate.

This module provides :class:`MLPEnsemble`, a lightweight ensemble of
independent multilayer perceptrons (MLPs) trained on a shared dataset.

The ensemble produces Monte-Carlo-like predictive samples by aggregating
outputs from independently initialized networks.

All outputs follow the tensor convention:

    samples: [B, M, q, E]

where

- B : batch dimension induced by input broadcasting
- M : ensemble size
- q : number of query points
- E : number of output dimensions

Posterior construction is delegated to :class:`_BaseEnsemble`, which converts
samples into a diagonal Normal posterior compatible with BoTorch.
"""

import torch
from torch import Tensor
import torch.nn as nn

from ..mlp import MLP
from ...utils.shapes import (
    ensure_2d_X,
    ensure_2d_y,
    normalize_to_bqd,
)
from .base import (
    FitConfig,
    _fit_ensemble_nets,
    _BaseEnsemble,
    cfg_from_legacy_kwargs,
)


class MLPEnsemble(_BaseEnsemble):
    """
    Deterministic ensemble of independent MLP regressors.

    This model trains multiple independently initialized
    :class:`mfbo.nn.mlp.MLP` networks on the same training dataset.
    Ensemble diversity arises from random weight initialization and
    independent optimization.

    The ensemble approximates predictive uncertainty through
    variability across ensemble members.

    Parameters
    ----------
    X_train : torch.Tensor
        Training inputs of shape ``[N, d]`` or any tensor flattenable
        to that shape.
    y_train : torch.Tensor
        Training targets of shape ``[N]`` or ``[N, E]``.
    ensemble_size : int, default=50
        Number of independent MLP members ``M``.
    hid_features : int, default=5
        Width of hidden layers in each MLP.
    n_hid_layers : int, default=2
        Number of hidden layers in each MLP.
    activation : torch.nn.Module or None, default=None
        Activation function used in hidden layers.
        If ``None``, the default activation of :class:`MLP` is used.
    bias : bool, default=True
        Whether linear layers include bias terms.

    Attributes
    ----------
    nets : torch.nn.ModuleList
        List of MLP ensemble members.
    train_x : torch.Tensor
        Stored training inputs.
    train_y : torch.Tensor
        Stored training targets.
    ensemble_size : int
        Number of ensemble members.
    _num_outputs : int
        Number of output dimensions ``E``.

    Notes
    -----
    - Training is performed independently for each ensemble member.
    - No parameter sharing occurs between members.
    - The model is fully deterministic; stochasticity arises only from
      independent random initialization.
    - Predictive uncertainty is estimated from ensemble variability.
    """

    def __init__(
        self,
        X_train: Tensor,
        y_train: Tensor,
        ensemble_size: int = 50,
        hid_features: int = 5,
        n_hid_layers: int = 2,
        activation: nn.Module | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__(ensemble_size=ensemble_size)

        X_train = ensure_2d_X(X_train)
        y_train = ensure_2d_y(y_train)

        self.train_x = X_train
        self.train_y = y_train
        self._num_outputs = y_train.shape[-1]

        in_dim = X_train.shape[-1]
        out_dim = self._num_outputs

        # ensemble of base nets
        self.nets = nn.ModuleList(
            [
                MLP(
                    in_features=in_dim,
                    out_features=out_dim,
                    hid_features=hid_features,
                    n_hid_layers=n_hid_layers,
                    activation=activation,
                    bias=bias,
                ).to(torch.get_default_dtype())
                for _ in range(ensemble_size)
            ]
        )

    def fit(self, cfg: FitConfig | None = None, **kwargs) -> None:
        cfg = cfg_from_legacy_kwargs(cfg, **kwargs)
        _fit_ensemble_nets(self.nets, self.train_x, self.train_y, cfg)
        """
        Fit all ensemble members independently.

        Parameters
        ----------
        cfg : FitConfig or None, optional
            Training configuration. If None, default values are used.
        **kwargs
            Legacy training keyword arguments such as
            ``optimizer``, ``epochs``, ``lr``, ``loss``, and ``verbose``.

        Returns
        -------
        None

        Notes
        -----
        This method performs in-place optimization of each ensemble member.
        Training is full-batch and does not use mini-batching or schedulers.
        """

    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate ensemble predictions at query points.

        Parameters
        ----------
        X : torch.Tensor
            Query inputs with shape:

            - ``[d]``
            - ``[q, d]``
            - ``[B, q, d]``

        Returns
        -------
        torch.Tensor
            Ensemble predictive samples with shape ``[B, M, q, E]`` where

            - ``B`` : broadcast batch dimension
            - ``M`` : ensemble size
            - ``q`` : number of query points
            - ``E`` : number of outputs

        Notes
        -----
        - Inputs are normalized to shape ``[B, q, d]`` internally.
        - Each ensemble member produces predictions independently.
        - For single-output models (``E=1``), the final dimension is retained.
        """
        Xb = normalize_to_bqd(X)      # [B,q,d]
        B, q, d = Xb.shape
        Xflat = Xb.reshape(B * q, d)     # [B*q,d]

        outs = []
        for net in self.nets:
            y = net(Xflat)              # [B*q] or [B*q,E]
            if y.ndim == 1:
                y = y.unsqueeze(-1)     # [B*q,1]
            y = y.view(B, q, -1)        # [B,q,E]
            outs.append(y.unsqueeze(1)) # [B,1,q,E]

        return torch.cat(outs, dim=1)   # [B,M,q,E]