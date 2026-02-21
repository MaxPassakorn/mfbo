from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Adam, AdamW
from tqdm.auto import trange
from botorch.models.ensemble import EnsembleModel

from ...posteriors.diag_normal import samples_to_mf_posterior
from ...utils.shapes import ensure_2d_X, ensure_2d_y


OptimizerName = Literal["Adam", "AdamW"]


@dataclass
class FitConfig:
    optimizer: OptimizerName = "AdamW"
    epochs: int = 1000
    lr: float = 1e-3
    loss: Literal["huber", "mse", "l1"] = "huber"
    verbose: bool = True


def _as_feature(y: Tensor) -> Tensor:
    """Ensure feature tensor is [N, f]."""
    if y.dim() == 1:
        return y.unsqueeze(-1)
    if y.dim() == 2:
        return y
    raise ValueError(f"Expected 1D or 2D tensor, got {tuple(y.shape)}")


def _make_optimizer(params: Iterable[torch.nn.Parameter], name: OptimizerName, lr: float):
    if name == "Adam":
        return Adam(params, lr=lr)
    if name == "AdamW":
        return AdamW(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def _make_loss(name: str) -> nn.Module:
    if name == "huber":
        return nn.HuberLoss()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unknown loss: {name}")


def _fit_ensemble_nets(nets: nn.ModuleList, X: Tensor, Y: Tensor, cfg: FitConfig) -> None:
    """
    Fit each net independently like your original file.

    - X: [N,d]
    - Y: [N,E] (or [N,1]) ; will squeeze for E=1 if pred is [N]
    """
    X = ensure_2d_X(X)
    Y = ensure_2d_y(Y)

    criterion = _make_loss(cfg.loss)

    for i, net in enumerate(nets):
        opt = _make_optimizer(net.parameters(), cfg.optimizer, cfg.lr)

        pbar = trange(cfg.epochs, leave=True, disable=not cfg.verbose)
        for _ in pbar:
            opt.zero_grad()
            pred = net(X)  # [N] or [N,E]

            tgt = Y
            if pred.dim() == 1:
                # backward-compatible behavior for single-output
                tgt = tgt.squeeze(-1)

            loss = criterion(pred, tgt)
            loss.backward()
            opt.step()

            if cfg.verbose:
                pbar.set_postfix(loss=float(loss.detach().cpu().item()))

def cfg_from_legacy_kwargs(cfg: "FitConfig | None" = None, **kwargs) -> "FitConfig":
    if cfg is not None and kwargs:
        raise ValueError("Use either cfg=FitConfig(...) or legacy kwargs, not both.")
    if cfg is not None:
        return cfg
    if not kwargs:
        return FitConfig()
    return FitConfig(
        optimizer=kwargs.get("optimizer", "AdamW"),
        epochs=int(kwargs.get("epochs", 1000)),
        lr=float(kwargs.get("lr", 1e-3)),
        loss=kwargs.get("loss", "huber"),
        verbose=bool(kwargs.get("verbose", True)),
    )

class _BaseEnsemble(EnsembleModel):
    """
    Base class to unify:
    - forward(X) -> samples [B, M, q, E]
    - posterior(X) -> MFDiagNormalPosterior using samples_to_mf_posterior
    """

    def __init__(self, ensemble_size: int):
        super().__init__()  # IMPORTANT: EnsembleModel takes no args
        self.ensemble_size = int(ensemble_size)

    def posterior(self, X: Tensor, **kwargs):
        samples = self.forward(X)
        return samples_to_mf_posterior(samples)

    @property
    def num_outputs(self) -> int:
        return int(self._num_outputs)