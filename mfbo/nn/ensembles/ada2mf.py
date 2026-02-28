from __future__ import annotations

"""
Ada2MF ensemble surrogate.

This module defines :class:`~mfbo.nn.ensembles.ada2mf.Ada2MFEnsemble`, a
deterministic ensemble surrogate built from multiple independent
:class:`~mfbo.nn.ada2mf.Ada2MF` networks.

Each ensemble member models the high-fidelity response using both the original
inputs ``x`` and low-fidelity features ``y_L(x)`` produced by a user-supplied
callable ``low_fn``.

The ensemble returns predictive samples with shape ``[B, M, q, E]``:

- ``B``: broadcast batch dimension
- ``M``: ensemble size
- ``q``: number of query points
- ``E``: number of outputs

Posterior construction is inherited from :class:`~mfbo.nn.ensembles.base._BaseEnsemble`.
"""

from typing import Callable

import torch
from torch import Tensor
import torch.nn as nn
from tqdm.auto import trange

from ..ada2mf import Ada2MF
from ...utils.shapes import (
    ensure_2d_X,
    ensure_2d_y,
    normalize_to_bqd,
)
from .base import (
    FitConfig,
    _as_feature,
    _make_optimizer,
    _BaseEnsemble,
    cfg_from_legacy_kwargs,
)

LowFn = Callable[[Tensor], Tensor]


class Ada2MFEnsemble(_BaseEnsemble):
    """
    Deterministic ensemble of Ada2MF networks for multi-fidelity regression.

    The ensemble approximates a high-fidelity mapping ``x -> y_H`` by combining:

    - input features ``x`` of shape ``[N, d]``
    - low-fidelity features ``y_L(x)`` produced by ``low_fn``

    Each ensemble member is an independently initialized and trained
    :class:`~mfbo.nn.ada2mf.Ada2MF` network.

    Parameters
    ----------
    X_train : torch.Tensor
        High-fidelity training inputs of shape ``[N, d]`` (or any tensor
        flattenable to that).
    y_train : torch.Tensor
        High-fidelity training targets of shape ``[N]`` or ``[N, E]``.
    low_fn : Callable[[torch.Tensor], torch.Tensor]
        Low-fidelity function used to generate features ``y_L(x)``.
        Must accept an input tensor of shape ``[N, d]`` and return either
        ``[N]`` or ``[N, f_L]``.
    ensemble_size : int, default=50
        Number of ensemble members ``M``.
    hid_features : int, default=5
        Hidden layer width used inside each :class:`~mfbo.nn.ada2mf.Ada2MF`.
    n_layers : int, default=2
        Number of hidden layers used inside each :class:`~mfbo.nn.ada2mf.Ada2MF`.

    Attributes
    ----------
    low_fn : Callable
        Stored low-fidelity function.
    nets : torch.nn.ModuleList
        List of :class:`~mfbo.nn.ada2mf.Ada2MF` ensemble members.
    train_x : torch.Tensor
        Training inputs (high-fidelity locations), shape ``[N, d]``.
    train_y : torch.Tensor
        Training targets, shape ``[N, E]``.
    ensemble_size : int
        Number of ensemble members ``M``.
    _num_outputs : int
        Number of output dimensions ``E``.

    Notes
    -----
    - The model is deterministic; predictive uncertainty is estimated from
      disagreement across ensemble members.
    - ``low_fn`` is evaluated at training time (for fitting) and at inference
      time (inside :meth:`forward`).
    """

    def __init__(
        self,
        X_train: Tensor,
        y_train: Tensor,
        low_fn: LowFn,
        ensemble_size: int = 50,
        hid_features: int = 5,
        n_layers: int = 2,
    ) -> None:
        super().__init__(ensemble_size=ensemble_size)

        if low_fn is None:
            raise ValueError("low_fn must be provided for Ada2MFEnsemble.")

        X_train = ensure_2d_X(X_train)
        y_train = ensure_2d_y(y_train)

        self.low_fn = low_fn
        self.train_x = X_train
        self.train_y = y_train
        self._num_outputs = y_train.shape[-1]

        # Infer low-fidelity feature dimension (fL)
        yL_train = _as_feature(low_fn(X_train))   # [N,fL]
        x_dim = X_train.shape[-1]
        yL_dim = yL_train.shape[-1]

        self.nets = nn.ModuleList(
            [
                Ada2MF(
                    x_dim=x_dim,
                    yL_dim=yL_dim,
                    out_features=self._num_outputs,
                    hid_features=hid_features,
                    n_layers=n_layers,
                ).to(torch.get_default_dtype())
                for _ in range(ensemble_size)
            ]
        )

    def fit(self, cfg: FitConfig | None = None, **kwargs) -> None:
        """
        Fit all Ada2MF ensemble members independently.

        Parameters
        ----------
        cfg : FitConfig or None, optional
            Training configuration (optimizer, learning rate, loss type,
            epochs, verbosity). If ``None``, defaults are used.
        **kwargs
            Legacy keyword arguments for training hyperparameters (e.g.,
            ``optimizer``, ``epochs``, ``lr``, ``loss``, ``verbose``).

        Returns
        -------
        None

        Notes
        -----
        Each ensemble member is trained on the same dataset, where the
        low-fidelity features are computed as:

        - ``y_L = low_fn(X_train)``

        and the network learns a mapping:

        - ``(X_train, y_L) -> y_train``

        The ensemble members are trained independently with full-batch updates.
        """
        cfg = cfg_from_legacy_kwargs(cfg, **kwargs)

        loss_fn = nn.HuberLoss()

        X = self.train_x
        yL = _as_feature(self.low_fn(X))
        Y = self.train_y

        for net in self.nets:
            opt = _make_optimizer(net.parameters(), cfg.optimizer, cfg.lr)
            pbar = trange(cfg.epochs, leave=False)
            for _ in pbar:
                opt.zero_grad()

                pred = net(X, yL)  # [N] or [N,E]

                # Align shapes for E=1 if needed
                if pred.ndim == 1 and Y.ndim == 2 and Y.shape[1] == 1:
                    tgt = Y.squeeze(-1)
                else:
                    tgt = Y

                loss = loss_fn(pred, tgt)
                loss.backward()
                opt.step()
                pbar.set_postfix(loss=float(loss.detach().cpu().item()))

    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate ensemble predictions at query points.

        Parameters
        ----------
        X : torch.Tensor
            Query inputs with shape ``[d]``, ``[q, d]``, or ``[B, q, d]``.

        Returns
        -------
        torch.Tensor
            Predictive samples with shape ``[B, M, q, E]`` where:

            - ``B`` is the broadcast batch dimension
            - ``M`` is the ensemble size
            - ``q`` is the number of query points
            - ``E`` is the number of outputs

        Notes
        -----
        The inputs are normalized to ``[B, q, d]`` internally and flattened to
        ``[B*q, d]`` for evaluation. Low-fidelity features are computed by

        - ``y_L = low_fn(X)``

        and each ensemble member predicts using ``Ada2MF(X, y_L)``.
        """
        Xb = normalize_to_bqd(X)     # [B,q,d]
        B, q, d = Xb.shape
        Xflat = Xb.reshape(B * q, d)    # [B*q,d]

        yL = _as_feature(self.low_fn(Xflat))  # [B*q,fL]

        outs = []
        for net in self.nets:
            y = net(Xflat, yL)          # [B*q] or [B*q,E]
            if y.ndim == 1:
                y = y.unsqueeze(-1)     # [B*q,1]
            y = y.view(B, q, -1)        # [B,q,E]
            outs.append(y.unsqueeze(1)) # [B,1,q,E]

        return torch.cat(outs, dim=1)   # [B,M,q,E]