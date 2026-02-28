from __future__ import annotations

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
    Deterministic Ada2MF ensemble.

    Each Ada2MF net consumes (x, yL(x)) and outputs yH prediction.
    - forward(X) returns samples [B, M, q, E]
    - posterior(X) inherited from _BaseEnsemble
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
        cfg = cfg_from_legacy_kwargs(cfg, **kwargs)
        """
        Train each net independently on (train_x, low_fn(train_x)) -> train_y.
        """
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
        X: [d] or [q,d] or [B,q,d]
        Returns samples: [B, M, q, E]
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