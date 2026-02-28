from __future__ import annotations

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
    Deterministic MLP ensemble.

    - Trains an ensemble of independent MLPs on (X_train, y_train).
    - forward(X) returns samples with shape [B, M, q, E].
    - posterior(X) is inherited from _BaseEnsemble and uses samples_to_mf_posterior.
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

    def forward(self, X: Tensor) -> Tensor:
        """
        X: [d] or [q,d] or [B,q,d]
        Returns samples: [B, M, q, E]
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