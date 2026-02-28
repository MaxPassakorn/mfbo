"""
Base utilities for neural ensemble surrogate models.

This module contains small, reusable building blocks shared by ensemble-based
surrogate models in :mod:`mfbo.nn.ensembles`.

It provides:

- :class:`~mfbo.nn.ensembles.base.FitConfig`:
  a lightweight configuration container for training ensemble members.
- helper utilities for creating optimizers and losses, and for normalizing
  tensor shapes.
- :class:`~mfbo.nn.ensembles.base._BaseEnsemble`:
  a minimal base class that standardizes the API expected by BoTorch
  (:class:`botorch.models.ensemble.EnsembleModel`).

Design conventions
------------------
All ensemble surrogates follow these conventions:

- ``forward(X)`` returns Monte-Carlo-like samples with shape
  ``[B, M, q, E]`` where

  - ``B``: batch shape induced by ``X`` (after broadcasting)
  - ``M``: ensemble size (number of members)
  - ``q``: number of query points
  - ``E``: number of outputs

- ``posterior(X)`` wraps these samples into a diagonal Normal posterior via
  :func:`mfbo.posteriors.diag_normal.samples_to_mf_posterior`.

The training helper :func:`~mfbo.nn.ensembles.base._fit_ensemble_nets` fits each
ensemble member independently using standard PyTorch optimization loops.
"""

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
    """
    Training configuration for fitting ensemble members.

    Parameters
    ----------
    optimizer : {"Adam", "AdamW"}, default="AdamW"
        Name of the optimizer used for training each ensemble member.
    epochs : int, default=1000
        Number of optimization steps used to fit each network.
    lr : float, default=1e-3
        Learning rate passed to the optimizer.
    loss : {"huber", "mse", "l1"}, default="huber"
        Loss function used to regress predictions against targets.
    verbose : bool, default=True
        If True, show a progress bar and per-iteration loss values.

    Notes
    -----
    This configuration is intentionally minimal and is designed for lightweight
    fitting of independent ensemble members. More advanced training behaviors
    (mini-batching, early stopping, schedulers) should be implemented in the
    concrete ensemble wrapper if needed.
    """
    optimizer: OptimizerName = "AdamW"
    epochs: int = 1000
    lr: float = 1e-3
    loss: Literal["huber", "mse", "l1"] = "huber"
    verbose: bool = True


def _as_feature(y: Tensor) -> Tensor:
    """
    Convert a target tensor into a 2D feature tensor.

    Parameters
    ----------
    y : torch.Tensor
        Input tensor of shape ``[N]`` or ``[N, F]``.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[N, F]`` (where ``F=1`` if the input was 1D).

    Raises
    ------
    ValueError
        If ``y`` is not 1D or 2D.
    """
    if y.dim() == 1:
        return y.unsqueeze(-1)
    if y.dim() == 2:
        return y
    raise ValueError(f"Expected 1D or 2D tensor, got {tuple(y.shape)}")


def _make_optimizer(params: Iterable[torch.nn.Parameter], name: OptimizerName, lr: float):
    """
    Construct a PyTorch optimizer for ensemble member training.

    Parameters
    ----------
    params : Iterable[torch.nn.Parameter]
        Parameters to optimize.
    name : {"Adam", "AdamW"}
        Optimizer choice.
    lr : float
        Learning rate.

    Returns
    -------
    torch.optim.Optimizer
        Instantiated optimizer.

    Raises
    ------
    ValueError
        If ``name`` is not recognized.
    """
    if name == "Adam":
        return Adam(params, lr=lr)
    if name == "AdamW":
        return AdamW(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def _make_loss(name: str) -> nn.Module:
    """
    Construct a regression loss module.

    Parameters
    ----------
    name : {"huber", "mse", "l1"}
        Loss type.

    Returns
    -------
    torch.nn.Module
        Loss module instance.

    Raises
    ------
    ValueError
        If ``name`` is not recognized.
    """
    if name == "huber":
        return nn.HuberLoss()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unknown loss: {name}")


def _fit_ensemble_nets(nets: nn.ModuleList, X: Tensor, Y: Tensor, cfg: FitConfig) -> None:
    """
    Fit a list of ensemble networks independently.

    Each network is optimized on the full dataset using the optimizer and loss
    specified by ``cfg``. No batching or shuffling is performed.

    Parameters
    ----------
    nets : torch.nn.ModuleList
        List of neural networks. Each network must implement ``forward(X)``
        and return either:

        - ``[N]`` for single-output regression, or
        - ``[N, E]`` for multi-output regression.
    X : torch.Tensor
        Training inputs. Expected shape is ``[N, d]`` (or any tensor flattenable
        to this shape). The function normalizes the shape via
        :func:`mfbo.utils.shapes.ensure_2d_X`.
    Y : torch.Tensor
        Training targets. Expected shape is ``[N]`` or ``[N, E]``. The function
        normalizes the shape via :func:`mfbo.utils.shapes.ensure_2d_y`.
    cfg : FitConfig
        Training configuration controlling optimizer, learning rate, loss, and
        number of epochs.

    Returns
    -------
    None

    Notes
    -----
    - If a network returns 1D predictions (``[N]``), the targets are squeezed to
      match for backward compatibility with scalar-output networks.
    - This routine performs in-place optimization; it does not return training
      history. If you need metrics, logging, or callbacks, implement them in the
      ensemble wrapper and call this function as a primitive.
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
    """
    Build a :class:`~mfbo.nn.ensembles.base.FitConfig` from either a config object
    or legacy keyword arguments.

    This helper exists to support older calling patterns where training
    hyperparameters were passed directly as keyword arguments.

    Parameters
    ----------
    cfg : FitConfig or None, optional
        If provided, this config is returned directly (and ``kwargs`` must be empty).
    **kwargs
        Legacy options such as ``optimizer``, ``epochs``, ``lr``, ``loss``, and
        ``verbose``.

    Returns
    -------
    FitConfig
        Constructed training configuration.

    Raises
    ------
    ValueError
        If both ``cfg`` and keyword arguments are provided simultaneously.
    """
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
    Minimal BoTorch-compatible base class for ensemble surrogates.

    This class standardizes the interface used by all neural ensemble surrogates
    in :mod:`mfbo.nn.ensembles`.

    Subclasses must implement
    -------------------------
    forward(X)
        Must return samples with shape ``[B, M, q, E]`` where ``M`` is the
        ensemble size and ``E`` is the number of outputs.

    Provided functionality
    ----------------------
    posterior(X)
        Converts samples returned by :meth:`forward` into a BoTorch-compatible
        posterior object via :func:`mfbo.posteriors.diag_normal.samples_to_mf_posterior`.

    Parameters
    ----------
    ensemble_size : int
        Number of ensemble members ``M``.

    Attributes
    ----------
    ensemble_size : int
        Number of ensemble members.

    Notes
    -----
    :class:`botorch.models.ensemble.EnsembleModel` does not accept constructor
    arguments, so subclasses must call ``super().__init__()`` and store any
    configuration locally.
    """
    
    def __init__(self, ensemble_size: int, num_outputs: int):
        super().__init__()
        self.ensemble_size = int(ensemble_size)
        self._num_outputs = int(num_outputs)

    def posterior(self, X: Tensor, **kwargs):
        """
        Construct a posterior distribution at ``X`` from ensemble samples.

        Parameters
        ----------
        X : torch.Tensor
            Query points. The exact shape is determined by the subclass, but it
            typically follows BoTorch conventions (e.g., ``[..., q, d]``).
        **kwargs
            Unused. Included for API compatibility with BoTorch.

        Returns
        -------
        botorch.posteriors.Posterior
            A diagonal Normal posterior constructed from ensemble samples.
        """
        samples = self.forward(X)
        return samples_to_mf_posterior(samples)

    @property
    def num_outputs(self) -> int:
        """
        Number of output dimensions.

        Returns
        -------
        int
            The number of outputs ``E`` produced by this ensemble surrogate.
        """
        return int(self._num_outputs)