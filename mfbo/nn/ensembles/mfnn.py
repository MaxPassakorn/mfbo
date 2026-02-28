from __future__ import annotations

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
    Deterministic MFNN ensemble.

    MFNN consumes augmented inputs:
        X_aug = cat([x, low_fn(x)], dim=-1)

    - Trains an ensemble of independent MFNNs on (X_aug_train, y_train).
    - forward(X) returns samples with shape [B, M, q, E].
    - posterior(X) is inherited from _BaseEnsemble and uses samples_to_mf_posterior.
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
        Fit each MFNN on augmented training data.
        """
        _fit_ensemble_nets(self.nets, self.train_x, self.train_y, cfg)

    def forward(self, X: Tensor) -> Tensor:
        """
        X: [d] or [q,d] or [B,q,d]
        Returns samples: [B, M, q, E]
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