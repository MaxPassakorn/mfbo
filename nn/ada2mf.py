from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn


def init_linear_kaiming(lin: nn.Linear, nonlinearity: str = "relu") -> None:
    nn.init.kaiming_normal_(lin.weight, nonlinearity=nonlinearity)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)


def make_mlp(
    in_features: int,
    hid_features: int,
    n_layers: int,
    activation: nn.Module,
    bias: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_features
    for _ in range(n_layers):
        lin = nn.Linear(last, hid_features, bias=bias)
        init_linear_kaiming(lin)
        layers += [lin, activation]
        last = hid_features
    out = nn.Linear(last, 1, bias=bias)
    init_linear_kaiming(out)
    layers.append(out)
    return nn.Sequential(*layers)


class Ada2MF(nn.Module):
    r"""
    Ada2MF base architecture (deterministic).

    For each output dim e:
      Inputs:
        x   : [N, x_dim]
        y_L : [N, yL_dim]
        cat = [x, y_L] : [N, x_dim + yL_dim]

      Branches:
        g1(cat) -> [N]     (linear)
        g2(cat) -> [N]     (MLP)
        g3(x)   -> [N]     (MLP)

      Weights:
        w_e = tanh(alpha[e])  -> 3 scalars in [-1, 1]

      Output:
        y_e = w0*g1 + w1*g2 + w2*g3

    Output shape:
      [N] if out_features=1 else [N, out_features]
    """

    def __init__(
        self,
        x_dim: int,
        yL_dim: int = 1,
        out_features: int = 1,
        hid_features: int = 64,
        n_layers: int = 2,
        activation: nn.Module | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if x_dim <= 0:
            raise ValueError("x_dim must be > 0")
        if yL_dim <= 0:
            raise ValueError("yL_dim must be > 0")
        if out_features <= 0:
            raise ValueError("out_features must be > 0")
        if hid_features <= 0:
            raise ValueError("hid_features must be > 0")
        if n_layers < 0:
            raise ValueError("n_layers must be >= 0")

        self.x_dim = x_dim
        self.yL_dim = yL_dim
        self.out_features = out_features

        self.activation = activation if activation is not None else nn.Mish()

        cat_dim = x_dim + yL_dim

        # One set of (g1, g2, g3) per output dimension (matches your original)
        self.g1_heads = nn.ModuleList()
        self.g2_heads = nn.ModuleList()
        self.g3_heads = nn.ModuleList()

        for _ in range(out_features):
            # g1: linear(cat) -> 1
            g1 = nn.Linear(cat_dim, 1, bias=bias)
            init_linear_kaiming(g1)
            self.g1_heads.append(g1)

            # g2: MLP(cat) -> 1
            self.g2_heads.append(
                make_mlp(cat_dim, hid_features, n_layers, self.activation, bias=bias)
            )

            # g3: MLP(x) -> 1
            self.g3_heads.append(
                make_mlp(x_dim, hid_features, n_layers, self.activation, bias=bias)
            )

        # alpha[e, 3] -> tanh -> weights in [-1,1]
        self.alpha = nn.Parameter(torch.zeros(out_features, 3))

    def forward(self, x: Tensor, y_L: Tensor) -> Tensor:
        # Flatten to [N, d]
        x = x.view(x.shape[0], -1)
        y_L = y_L.view(y_L.shape[0], -1)

        if x.shape[1] != self.x_dim:
            raise RuntimeError(f"Expected x dim {self.x_dim}, got {x.shape}")
        if y_L.shape[1] != self.yL_dim:
            raise RuntimeError(f"Expected y_L dim {self.yL_dim}, got {y_L.shape}")

        cat = torch.cat([x, y_L], dim=-1)  # [N, x_dim+yL_dim]

        ys = []
        for e in range(self.out_features):
            g1 = self.g1_heads[e](cat).squeeze(-1)  # [N]
            g2 = self.g2_heads[e](cat).squeeze(-1)  # [N]
            g3 = self.g3_heads[e](x).squeeze(-1)    # [N]
            w = torch.tanh(self.alpha[e])           # [3]
            ys.append(w[0] * g1 + w[1] * g2 + w[2] * g3)

        if self.out_features == 1:
            return ys[0]                            # [N]
        return torch.stack(ys, dim=-1)              # [N, E]