from __future__ import annotations

"""
Deterministic MFNN ensemble surrogate.

This module provides :class:`MFNNEnsemble`, an ensemble of independent
Multi-Fidelity Neural Networks (MFNNs).

Each ensemble member is trained on augmented inputs:

    X_aug = concat([x, low_fn(x)], dim=-1)

where ``low_fn`` is a user-supplied low-fidelity function.

The ensemble returns predictive samples with shape:

    samples: [B, M, q, E]

where

- B : broadcast batch dimension
- M : ensemble size
- q : number of query points
- E : number of outputs

Posterior construction is inherited from :class:`_BaseEnsemble`.
"""

from typing import Callable

import torch
from torch import Tensor
import torch.nn as nn

from ..mfnn import MFNN

from ...utils.shapes import (
    ensure_2d_X,
    ensure_2d_y,
    normalize_to_bqd,
)
from .base import (
    FitConfig,
    _fit_ensemble_nets,
    _as_feature,
    _BaseEnsemble,
    cfg_from_legacy_kwargs,
)

LowFn = Callable[[Tensor], Tensor]


class MFNNEnsemble(_BaseEnsemble):
    """
    Deterministic ensemble of Multi-Fidelity Neural Networks (MFNN).

    This model trains multiple independently initialized MFNNs on
    augmented training inputs:

        X_aug = concat([x, low_fn(x)], dim=-1)

    where ``low_fn`` provides low-fidelity predictions or features.

    Ensemble diversity arises from independent random initialization
    and optimization of each MFNN member.

    Parameters
    ----------
    X_train : torch.Tensor
        High-fidelity training inputs of shape ``[N, d]``.
    y_train : torch.Tensor
        High-fidelity training targets of shape ``[N]`` or ``[N, E]``.
    low_fn : Callable[[torch.Tensor], torch.Tensor]
        Low-fidelity function mapping ``x -> y_L``.
        Must accept input of shape ``[N, d]`` and return
        ``[N]`` or ``[N, f_L]``.
    ensemble_size : int, default=50
        Number of independent MFNN members ``M``.
    hid_features : int, default=5
        Width of hidden layers in each MFNN.
    n_hid_layers : int, default=2
        Number of hidden layers in each MFNN.
    activation : torch.nn.Module or None, default=None
        Activation function used in hidden layers.
        If ``None``, the default activation of :class:`MFNN` is used.
    bias : bool, default=True
        Whether linear layers include bias terms.

    Attributes
    ----------
    low_fn : Callable
        Stored low-fidelity function.
    nets : torch.nn.ModuleList
        List of MFNN ensemble members.
    train_x_raw : torch.Tensor
        Original high-fidelity training inputs.
    train_x : torch.Tensor
        Augmented training inputs used for fitting.
    train_y : torch.Tensor
        High-fidelity targets.
    ensemble_size : int
        Number of ensemble members.
    _num_outputs : int
        Number of output dimensions ``E``.

    Notes
    -----
    - The ensemble is deterministic; predictive uncertainty arises
      from variability across independently trained members.
    - The low-fidelity function is evaluated at both training and
      inference time.
    - No parameter sharing occurs between ensemble members.
    """

    def __init__(
        self,
        X_train: Tensor,
        y_train: Tensor,
        low_fn: LowFn,
        ensemble_size: int = 50,
        hid_features: int = 5,
        n_hid_layers: int = 2,
        activation: nn.Module | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__(ensemble_size=ensemble_size)

        if low_fn is None:
            raise ValueError("low_fn must be provided for MFNNEnsemble.")

        X_train = ensure_2d_X(X_train)
        y_train = ensure_2d_y(y_train)

        self.low_fn = low_fn
        self.train_x_raw = X_train
        self.train_y = y_train
        self._num_outputs = y_train.shape[-1]

        # Build augmented training input
        yL_train = _as_feature(low_fn(X_train))              # [N,fL]
        X_aug_train = torch.cat([X_train, yL_train], dim=-1) # [N,d+fL]
        self.train_x = X_aug_train

        in_dim = X_aug_train.shape[-1]
        out_dim = self._num_outputs

        self.nets = nn.ModuleList(
            [
                MFNN(
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
        """
        Fit all MFNN ensemble members independently.

        Parameters
        ----------
        cfg : FitConfig or None, optional
            Training configuration controlling optimizer, learning rate,
            loss function, and number of epochs.
        **kwargs
            Legacy keyword arguments for training hyperparameters
            (e.g., ``optimizer``, ``epochs``, ``lr``, ``loss``, ``verbose``).

        Returns
        -------
        None

        Notes
        -----
        Each ensemble member is trained on the same augmented dataset:

            X_aug_train = concat([X_train, low_fn(X_train)], dim=-1)

        Training is full-batch and deterministic.
        """
        _fit_ensemble_nets(self.nets, self.train_x, self.train_y, cfg)

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
        - Inputs are normalized internally to shape ``[B, q, d]``.
        - Low-fidelity features are computed as:

              y_L = low_fn(X)

        - The augmented input

              X_aug = concat([X, y_L], dim=-1)

          is passed to each MFNN member.
        - For single-output models (``E=1``), the final dimension is retained.
        """
        Xb = normalize_to_bqd(X)     # [B,q,d]
        B, q, d = Xb.shape
        Xflat = Xb.reshape(B * q, d)    # [B*q,d]

        # Augment with low-fidelity features
        yL = _as_feature(self.low_fn(Xflat))                 # [B*q,fL]
        Xaug = torch.cat([Xflat, yL], dim=-1)                # [B*q,d+fL]

        outs = []
        for net in self.nets:
            y = net(Xaug)               # [B*q] or [B*q,E]
            if y.ndim == 1:
                y = y.unsqueeze(-1)     # [B*q,1]
            y = y.view(B, q, -1)        # [B,q,E]
            outs.append(y.unsqueeze(1)) # [B,1,q,E]

        return torch.cat(outs, dim=1)   # [B,M,q,E]