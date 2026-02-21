from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn


def init_linear_kaiming(lin: nn.Linear, nonlinearity: str = "relu") -> None:
    """Kaiming init for Linear layers."""
    nn.init.kaiming_normal_(lin.weight, nonlinearity=nonlinearity)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)


class MLP(nn.Module):
    """
    Simple multi-head MLP.

    - Input:  x shape [N, d] (or anything flattenable to that)
    - Output: y shape [N] if out_features=1 else [N, out_features]

    This is a base network only; training and BoTorch wrappers live elsewhere.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hid_features: int = 64,
        n_hid_layers: int = 2,
        activation: nn.Module | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be > 0")
        if out_features <= 0:
            raise ValueError("out_features must be > 0")
        if hid_features <= 0:
            raise ValueError("hid_features must be > 0")
        if n_hid_layers < 0:
            raise ValueError("n_hid_layers must be >= 0")

        self.in_features = in_features
        self.out_features = out_features
        self.hid_features = hid_features
        self.n_hid_layers = n_hid_layers

        self.activation = activation if activation is not None else nn.Mish()

        # One head per output dim (keeps your current behavior)
        self.heads = nn.ModuleList()
        for _ in range(out_features):
            layers: list[nn.Module] = []
            last = in_features

            for _ in range(n_hid_layers):
                lin = nn.Linear(last, hid_features, bias=bias)
                init_linear_kaiming(lin)
                layers.append(lin)
                layers.append(self.activation)
                last = hid_features

            out_lin = nn.Linear(last, 1, bias=bias)
            init_linear_kaiming(out_lin)
            layers.append(out_lin)

            self.heads.append(nn.Sequential(*layers))

    def forward(self, x: Tensor) -> Tensor:
        # flatten to [N, d]
        x = x.view(x.shape[0], -1)

        outs = [head(x).squeeze(-1) for head in self.heads]  # list of [N]
        if self.out_features == 1:
            return outs[0]  # [N]
        return torch.stack(outs, dim=-1)  # [N, out_features]