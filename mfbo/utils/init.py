from __future__ import annotations

"""
Neural network initialization utilities.

This module provides standardized helper functions for constructing and
initializing feedforward neural network layers used throughout the library.

The primary initialization strategy is **Kaiming (He) initialization**
(:func:`torch.nn.init.kaiming_normal_`), which preserves variance in deep
networks when used with ReLU-like nonlinearities.

These utilities ensure:

- Consistent initialization across surrogate architectures
- Stable training behavior for ensemble members
- Reduced duplication of initialization logic
"""

import torch.nn as nn


def init_linear_kaiming(lin: nn.Linear, nonlinearity: str = "relu") -> None:
    """
    Construct a fully-connected multilayer perceptron (MLP).

    The architecture is:

    - ``n_layers`` hidden layers of width ``hid_features``
    - Activation applied after each hidden linear layer
    - Final linear output layer of size ``out_features``

    All linear layers are initialized using
    :func:`init_linear_kaiming`.

    Parameters
    ----------
    in_features : int
        Input dimensionality ``d``.
    hid_features : int
        Number of hidden units per hidden layer.
    n_layers : int
        Number of hidden layers. Must be >= 0.
    activation : torch.nn.Module
        Activation function applied after each hidden linear layer.
    out_features : int, default=1
        Output dimensionality.
    bias : bool, default=True
        Whether to include bias terms in linear layers.

    Returns
    -------
    torch.nn.Sequential
        A fully initialized feedforward network.

    Notes
    -----
    - If ``n_layers = 0``, the network reduces to a single linear layer.
    - This utility does not include output activation; that should be
      applied externally if needed.
    - Intended for deterministic surrogate networks and ensemble members.
    """
    nn.init.kaiming_normal_(lin.weight, nonlinearity=nonlinearity)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)


def make_mlp(
    in_features: int,
    hid_features: int,
    n_layers: int,
    activation: nn.Module,
    out_features: int = 1,
    bias: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_features
    for _ in range(n_layers):
        lin = nn.Linear(last, hid_features, bias=bias)
        init_linear_kaiming(lin)
        layers += [lin, activation]
        last = hid_features

    out = nn.Linear(last, out_features, bias=bias)
    init_linear_kaiming(out)
    layers.append(out)
    return nn.Sequential(*layers)