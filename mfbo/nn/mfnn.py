from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn


def init_linear_kaiming(lin: nn.Linear, nonlinearity: str = "relu") -> None:
    """Kaiming init for Linear layers."""
    nn.init.kaiming_normal_(lin.weight, nonlinearity=nonlinearity)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)


class MFNN(nn.Module):
    r"""
    Multi-Fidelity Neural Network (MFNN) base architecture.

    For each output dimension e:
      z_lin = Linear(x)
      z_nl  = MLP(x)
      a     = sigmoid(alpha_e)
      y_e   = a * z_lin + (1 - a) * z_nl

    Input:
      x: [N, d] (or flattenable to [N, d])
    Output:
      y: [N] if out_features=1 else [N, out_features]

    Notes:
    - This file defines ONLY the network.
    - Any concatenation with low-fidelity features (x, yL) happens outside,
      typically inside an ensemble wrapper.
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

        # One linear head + one nonlinear head per output dimension
        self.linear_heads = nn.ModuleList()
        self.nonlinear_heads = nn.ModuleList()

        for _ in range(out_features):
            # linear head
            lin = nn.Linear(in_features, 1, bias=bias)
            init_linear_kaiming(lin)
            self.linear_heads.append(lin)

            # nonlinear head (MLP)
            layers: list[nn.Module] = []
            last = in_features
            for _ in range(n_hid_layers):
                hl = nn.Linear(last, hid_features, bias=bias)
                init_linear_kaiming(hl)
                layers.append(hl)
                layers.append(self.activation)
                last = hid_features

            out_lin = nn.Linear(last, 1, bias=bias)
            init_linear_kaiming(out_lin)
            layers.append(out_lin)

            self.nonlinear_heads.append(nn.Sequential(*layers))

        # raw_alpha[e] is a learnable logit (scalar) per output dimension
        self.raw_alpha = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.shape[0], -1)

        ys = []
        for e in range(self.out_features):
            z_lin = self.linear_heads[e](x).squeeze(-1)      # [N]
            z_nl = self.nonlinear_heads[e](x).squeeze(-1)    # [N]
            a = torch.sigmoid(self.raw_alpha[e])             # scalar
            ys.append(a * z_lin + (1.0 - a) * z_nl)          # [N]

        if self.out_features == 1:
            return ys[0]                                     # [N]
        return torch.stack(ys, dim=-1)                        # [N, E]