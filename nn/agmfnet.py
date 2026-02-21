from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def init_linear_kaiming(lin: nn.Linear, nonlinearity: str = "relu") -> None:
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


class AGMFNet(nn.Module):
    r"""
    Adaptive Gated Multi-Fidelity Network (AGMFNet) base architecture.

    For each output dim e:
      cat = [x, y_L]
      g1(cat) -> linear(cat)
      g2(cat) -> MLP(cat)
      g3(x)   -> MLP(x)
      gate(cat) -> logits(3) -> softmax weights w (3)

      y = w0*g1 + w1*g2 + w2*g3

    Input:
      x   : [N, x_dim]
      y_L : [N, yL_dim]  (low-fidelity outputs/features)
    Output:
      y: [N] if out_features=1 else [N, out_features]

    If return_parts=True, returns:
      (mu, g1, g2, g3, w_gate, (s_H, s_LH, s_R))
    where:
      mu    : [N, E]
      g1/g2/g3 : [N, E]
      w_gate: [N, E, 3]
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

        # One set of heads per output dim (matches your original code)
        self.g1_heads = nn.ModuleList()
        self.g2_heads = nn.ModuleList()
        self.g3_heads = nn.ModuleList()
        self.gate_heads = nn.ModuleList()

        for _ in range(out_features):
            # g1: linear(cat) -> 1
            g1 = nn.Linear(cat_dim, 1, bias=bias)
            init_linear_kaiming(g1)
            self.g1_heads.append(g1)

            # g2: MLP(cat) -> 1
            self.g2_heads.append(make_mlp(cat_dim, hid_features, n_layers, self.activation, out_features=1, bias=bias))

            # g3: MLP(x) -> 1
            self.g3_heads.append(make_mlp(x_dim, hid_features, n_layers, self.activation, out_features=1, bias=bias))

            # gate: cat -> hidden -> 3 logits
            self.gate_heads.append(
                make_mlp(cat_dim, hid_features, max(1, n_layers), self.activation, out_features=3, bias=bias)
            )

        # Optional scalars for AFW logging (do not affect forward math)
        self.s_H = nn.Parameter(torch.zeros(out_features))
        self.s_LH = nn.Parameter(torch.zeros(out_features))
        self.s_R = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: Tensor, y_L: Tensor, return_parts: bool = False):
        x = x.view(x.shape[0], -1)
        y_L = y_L.view(y_L.shape[0], -1)

        if x.shape[1] != self.x_dim:
            raise RuntimeError(f"Expected x dim {self.x_dim}, got {tuple(x.shape)}")
        if y_L.shape[1] != self.yL_dim:
            raise RuntimeError(f"Expected y_L dim {self.yL_dim}, got {tuple(y_L.shape)}")

        cat = torch.cat([x, y_L], dim=-1)  # [N, cat_dim]
        N = x.shape[0]
        E = self.out_features

        mu_list: list[Tensor] = []
        g1_list: list[Tensor] = []
        g2_list: list[Tensor] = []
        g3_list: list[Tensor] = []
        w_list: list[Tensor] = []

        for e in range(E):
            g1 = self.g1_heads[e](cat).squeeze(-1)         # [N]
            g2 = self.g2_heads[e](cat).squeeze(-1)         # [N]
            g3 = self.g3_heads[e](x).squeeze(-1)           # [N]
            logits = self.gate_heads[e](cat)               # [N, 3]
            w = F.softmax(logits, dim=-1)                  # [N, 3]
            mu = w[:, 0] * g1 + w[:, 1] * g2 + w[:, 2] * g3

            mu_list.append(mu)
            g1_list.append(g1)
            g2_list.append(g2)
            g3_list.append(g3)
            w_list.append(w)

        if E == 1:
            # For single-output, keep output consistent
            mu = mu_list[0].unsqueeze(-1)                  # [N,1]
            if not return_parts:
                return mu.squeeze(-1)                      # [N]
            g1t = g1_list[0].unsqueeze(-1)                 # [N,1]
            g2t = g2_list[0].unsqueeze(-1)
            g3t = g3_list[0].unsqueeze(-1)
            wt = w_list[0].unsqueeze(1)                    # [N,1,3]
            return mu, g1t, g2t, g3t, wt, (self.s_H, self.s_LH, self.s_R)

        # Multi-output: stack along last dim for values, along dim=1 for gates
        mu = torch.stack(mu_list, dim=-1)                  # [N,E]
        if not return_parts:
            return mu
        g1t = torch.stack(g1_list, dim=-1)                 # [N,E]
        g2t = torch.stack(g2_list, dim=-1)
        g3t = torch.stack(g3_list, dim=-1)
        wt = torch.stack(w_list, dim=1)                    # [N,E,3]
        return mu, g1t, g2t, g3t, wt, (self.s_H, self.s_LH, self.s_R)