from __future__ import annotations

import torch.nn as nn


def kaiming_init_linear(layer: nn.Linear, nonlinearity: str = "relu", bias_zero: bool = True) -> None:
    """
    Kaiming init for nn.Linear weights and optional zero bias.
    """
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(layer)}")

    nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
    if layer.bias is not None and bias_zero:
        nn.init.zeros_(layer.bias)


def init_mlp_sequential(seq: nn.Sequential, nonlinearity: str = "relu") -> None:
    """
    Initialize all Linear layers inside an nn.Sequential.
    """
    for m in seq.modules():
        if isinstance(m, nn.Linear):
            kaiming_init_linear(m, nonlinearity=nonlinearity)