from __future__ import annotations

"""
AGMFNet ensemble surrogate.

This module defines :class:`~mfbo.nn.ensembles.agmfnet.AGMFNetEnsemble`, a
deterministic ensemble surrogate built from multiple independently trained
:class:`~mfbo.nn.agmfnet.AGMFNet` networks.

Each ensemble member models a high-fidelity response using both:

- inputs ``x``
- low-fidelity features ``y_L(x)`` provided by a user-supplied callable ``low_fn``

The ensemble returns predictive samples with shape ``[B, M, q, E]`` where:

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

from ..agmfnet import AGMFNet
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


class AGMFNetEnsemble(_BaseEnsemble):
    """
    Deterministic ensemble of AGMFNet networks for multi-fidelity regression.

    The ensemble approximates a high-fidelity mapping ``x -> y_H`` by
    augmenting inputs with low-fidelity features ``y_L(x)`` produced by
    ``low_fn``. Each ensemble member is an independently initialized and
    trained :class:`~mfbo.nn.agmfnet.AGMFNet` network.

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
        Hidden layer width used inside each :class:`~mfbo.nn.agmfnet.AGMFNet`.
    n_layers : int, default=2
        Number of hidden layers used inside each :class:`~mfbo.nn.agmfnet.AGMFNet`.

    Attributes
    ----------
    low_fn : Callable
        Stored low-fidelity function.
    nets : torch.nn.ModuleList
        List of :class:`~mfbo.nn.agmfnet.AGMFNet` ensemble members.
    train_x : torch.Tensor
        Training inputs, shape ``[N, d]``.
    train_y : torch.Tensor
        Training targets, shape ``[N, E]``.
    ensemble_size : int
        Number of ensemble members ``M``.
    _num_outputs : int
        Number of output dimensions ``E``.

    Notes
    -----
    **Uncertainty**:
        This class is deterministic; uncertainty is estimated from disagreement
        across ensemble members.

    **Two kinds of weights are used**:

    1. *Model-internal gate weights* (per sample):
       Each :class:`~mfbo.nn.agmfnet.AGMFNet` has a gating network that outputs
       a softmax weight vector over three branches (linear-on-``[x,y_L]``,
       nonlinear-on-``[x,y_L]``, nonlinear-on-``x``). These vary with ``x``.

    2. *Adaptive loss weights (AFW)* (per output dimension):
       During training, this ensemble optionally reweights three loss terms
       using a dual update in log-loss space. AFW is a training-time mechanism
       and does not change the model's inference interface.
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
            raise ValueError("low_fn must be provided for AGMFNetEnsemble.")

        X_train = ensure_2d_X(X_train)
        y_train = ensure_2d_y(y_train)

        self.low_fn = low_fn
        self.train_x = X_train
        self.train_y = y_train
        self._num_outputs = y_train.shape[-1]

        # infer yL feature dimension
        yL_train = _as_feature(low_fn(X_train))  # [N, fL]
        x_dim = X_train.shape[-1]
        yL_dim = yL_train.shape[-1]

        self.nets = nn.ModuleList(
            [
                AGMFNet(
                    x_dim=x_dim,
                    yL_dim=yL_dim,
                    out_features=self._num_outputs,
                    hid_features=hid_features,
                    n_layers=n_layers,
                ).to(torch.get_default_dtype())
                for _ in range(ensemble_size)
            ]
        )

    def fit(
        self,
        cfg: FitConfig | None = None,
        **kwargs,  # supports optimizer=..., epochs=..., lr=...
    ) -> None:
        """
        Fit all AGMFNet ensemble members independently using adaptive loss weighting.

        Parameters
        ----------
        cfg : FitConfig or None, optional
            Training configuration (optimizer, learning rate, number of epochs,
            loss name, verbosity). If ``None``, defaults are used.
        **kwargs
            Legacy keyword arguments forwarded to :func:`cfg_from_legacy_kwargs`.
            Typical keys include ``optimizer``, ``epochs``, ``lr``, ``loss``,
            and ``verbose``.

        Returns
        -------
        None

        Notes
        -----
        Each ensemble member is trained on the same dataset using the input
        augmentation:

        - ``y_L = low_fn(X_train)``
        - model input is ``(X_train, y_L)``

        Loss decomposition (per output dimension)
        -----------------------------------------
        For each output dimension, training uses three Huber losses:

        - ``L_H``  : mismatch between predicted high-fidelity mean ``mu`` and ``y_H``
        - ``L_LH`` : mismatch between the combined low-fidelity branches ``(g1 + g2)``
                    and ``y_H``
        - ``L_R``  : mismatch between the residual branch ``g3`` and the residual target
                    ``(y_H - y_L)``

        These are stacked into a vector ``L = [L_H, L_LH, L_R]`` and combined using
        adaptive weights ``w`` per output dimension:

        - ``L_total(e) = sum_k w_e[k] * L_e[k]``

        Adaptive loss weighting (AFW)
        -----------------------------
        AFW maintains a per-output parameter vector ``u`` and converts it to weights
        via a softmax:

        - ``w = softmax(u)``

        After each optimizer step, AFW updates ``u`` using a dual update based on
        differences of log-loss values (before vs. after the step). The update is
        performed per output dimension and clipped for numerical stability.

        Logging
        -------
        The progress bar reports the total loss and the mean AFW weights over
        outputs. If the underlying network returns gate weights, their dataset-mean
        values are also logged.
        """
        cfg = cfg_from_legacy_kwargs(cfg, **kwargs)

        y_train = self.train_y
        X = self.train_x
        y_L = _as_feature(self.low_fn(X))  # [N,fL]

        # AFW per output
        criterion = nn.HuberLoss(reduction="none")

        for net in self.nets:
            optim = _make_optimizer(net.parameters(), cfg.optimizer, cfg.lr)

            K = 3
            E = self._num_outputs
            device = next(net.parameters()).device

            u = torch.zeros(E, K, device=device)  # [E,3]
            eta_u = 0.1
            clip_u = 8.0
            eps = 1e-12

            pbar = trange(cfg.epochs, leave=True)

            for _ in pbar:
                # ---- forward (before step) ----
                out = net(X, y_L, return_parts=True)
                mu, g1, g2, g3 = out[0], out[1], out[2], out[3]
                w_gate = out[4] if len(out) > 4 else None

                if mu.ndim == 1:
                    mu = mu.unsqueeze(-1)
                    g1 = g1.unsqueeze(-1)
                    g2 = g2.unsqueeze(-1)
                    g3 = g3.unsqueeze(-1)

                # y shapes
                Y = y_train
                if Y.ndim == 1:
                    Y = Y.unsqueeze(-1)
                yL = y_L
                if yL.ndim == 1:
                    yL = yL.unsqueeze(-1)

                diff_H  = criterion(mu,       Y)       # [N,E]
                diff_LH = criterion(g1 + g2,  Y)       # [N,E]
                diff_R  = criterion(g3,       Y - yL)  # [N,E]

                L_H_vec  = diff_H.mean(dim=0)          # [E]
                L_LH_vec = diff_LH.mean(dim=0)         # [E]
                L_R_vec  = diff_R.mean(dim=0)          # [E]
                L_stack  = torch.stack([L_H_vec, L_LH_vec, L_R_vec], dim=-1)  # [E,3]

                with torch.no_grad():
                    w_tasks = torch.softmax(u, dim=-1)  # [E,3]

                L_per_output = (w_tasks * L_stack).sum(dim=-1)  # [E]
                L = L_per_output.mean()

                # ---- step ----
                optim.zero_grad()
                L.backward()
                optim.step()

                # ---- recompute after step for AFW update ----
                with torch.no_grad():
                    out2 = net(X, y_L, return_parts=True)
                    mu2, g12, g22, g32 = out2[0], out2[1], out2[2], out2[3]
                    if mu2.ndim == 1:
                        mu2 = mu2.unsqueeze(-1)
                        g12 = g12.unsqueeze(-1)
                        g22 = g22.unsqueeze(-1)
                        g32 = g32.unsqueeze(-1)

                    diff2_H  = criterion(mu2,       Y)
                    diff2_LH = criterion(g12 + g22, Y)
                    diff2_R  = criterion(g32,       Y - yL)

                    L2_H_vec  = diff2_H.mean(dim=0)
                    L2_LH_vec = diff2_LH.mean(dim=0)
                    L2_R_vec  = diff2_R.mean(dim=0)
                    L2_stack  = torch.stack([L2_H_vec, L2_LH_vec, L2_R_vec], dim=-1)  # [E,3]

                    logL  = (L_stack  + eps).log()
                    logL2 = (L2_stack + eps).log()
                    dlog  = logL - logL2  # [E,3]

                    # dual update
                    for e_idx in range(E):
                        w_e = w_tasks[e_idx]          # [3]
                        dlog_e = dlog[e_idx]          # [3]
                        Jsoft = torch.diag(w_e) - torch.outer(w_e, w_e)  # [3,3]
                        grad_u = Jsoft @ dlog_e
                        u[e_idx] = (u[e_idx] - eta_u * grad_u).clamp(-clip_u, clip_u)

                    w_tasks_new = torch.softmax(u, dim=-1)
                    w_avg = w_tasks_new.mean(dim=0)

                # ---- logging ----
                log_dict = {
                    "loss": float(L.detach().cpu().item()),
                    "afw_w0": float(w_avg[0].cpu().item()),
                    "afw_w1": float(w_avg[1].cpu().item()),
                    "afw_w2": float(w_avg[2].cpu().item()),
                }

                if w_gate is not None:
                    # w_gate: [N,E,3] or [N,1,3]
                    if w_gate.dim() == 3:
                        wg = w_gate.mean(dim=1)  # [N,3]
                    else:
                        wg = w_gate
                    log_dict.update({
                        "gate_w0": float(wg[:, 0].mean().cpu().item()),
                        "gate_w1": float(wg[:, 1].mean().cpu().item()),
                        "gate_w2": float(wg[:, 2].mean().cpu().item()),
                    })

                pbar.set_postfix(**log_dict)

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
            Predictive samples with shape ``[B, M, q, E]``, where:

            - ``B`` is the broadcast batch dimension
            - ``M`` is the ensemble size
            - ``q`` is the number of query points
            - ``E`` is the number of outputs

        Notes
        -----
        Inputs are normalized to ``[B, q, d]`` internally and flattened to
        ``[B*q, d]`` for evaluation. Low-fidelity features are computed by

        - ``y_L = low_fn(X)``

        and each ensemble member predicts using :class:`~mfbo.nn.agmfnet.AGMFNet`.
        The returned tensor stacks predictions from all ensemble members along the
        ensemble dimension ``M``.
        """
        Xb = normalize_to_bqd(X)     # [B,q,d]
        B, q, d = Xb.shape
        Xflat = Xb.reshape(B * q, d)    # [B*q,d]

        yL = _as_feature(self.low_fn(Xflat))  # [B*q,fL]

        outs = []
        for net in self.nets:
            y = net(Xflat, yL, return_parts=False)  # [B*q] or [B*q,E]
            if y.ndim == 1:
                y = y.unsqueeze(-1)                 # [B*q,1]
            y = y.view(B, q, -1)                    # [B,q,E]
            outs.append(y.unsqueeze(1))             # [B,1,q,E]

        return torch.cat(outs, dim=1)               # [B,M,q,E]