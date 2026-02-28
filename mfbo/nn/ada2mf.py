from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn


def init_linear_kaiming(lin: nn.Linear, nonlinearity: str = "relu") -> None:
    """
    Initialize a ``nn.Linear`` layer using Kaiming (He) initialization.

    Parameters
    ----------
    lin : torch.nn.Linear
        Linear layer to initialize.
    nonlinearity : str, default="relu"
        Nonlinearity used after the layer. Passed to
        :func:`torch.nn.init.kaiming_normal_`.
    """
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
    """
    Construct a fully connected MLP ending in a scalar output.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    hid_features : int
        Width of hidden layers.
    n_layers : int
        Number of hidden layers.
    activation : torch.nn.Module
        Activation function applied after each hidden linear layer.
    bias : bool, default=True
        Whether linear layers include bias terms.

    Returns
    -------
    torch.nn.Sequential
        MLP mapping ``in_features -> 1``.
    """
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
    Adaptive two-stage multi-fidelity neural network (Ada2MF).

    Ada2MF combines low-fidelity information and high-fidelity input features
    using three learnable branches per output dimension:

    - :math:`g_1(x, y_L)` — linear trend over concatenated inputs
    - :math:`g_2(x, y_L)` — nonlinear correction over concatenated inputs
    - :math:`g_3(x)` — nonlinear residual model over high-fidelity inputs only

    For each output dimension :math:`e`, the prediction is

    .. math::

        y_e(x, y_L) =
        w_{e,0} g_1(x, y_L)
        + w_{e,1} g_2(x, y_L)
        + w_{e,2} g_3(x),

    where

    .. math::

        w_e = \tanh(\alpha_e) \in [-1, 1]^3.

    This formulation allows adaptive weighting between linear,
    nonlinear, and residual components.

    Notes
    -----
    - This implementation is deterministic.
    - Each output dimension has independent branches and weights.
    - Multi-fidelity training logic and Bayesian wrappers are defined elsewhere.
    - The mixing weights are unconstrained parameters mapped through ``tanh``
      to the interval ``[-1, 1]``.

    Parameters
    ----------
    x_dim : int
        Dimension of high-fidelity input features ``x``.
    yL_dim : int, default=1
        Dimension of low-fidelity input features ``y_L``.
    out_features : int, default=1
        Number of output dimensions.
    hid_features : int, default=5
        Width of hidden layers in nonlinear branches.
    n_layers : int, default=2
        Number of hidden layers in nonlinear branches.
    activation : torch.nn.Module or None, default=None
        Activation function used in nonlinear branches.
        If ``None``, :class:`torch.nn.Mish` is used.
    bias : bool, default=True
        Whether linear layers include bias terms.

    Attributes
    ----------
    g1_heads : torch.nn.ModuleList
        Linear branches over concatenated inputs.
    g2_heads : torch.nn.ModuleList
        Nonlinear MLP branches over concatenated inputs.
    g3_heads : torch.nn.ModuleList
        Nonlinear MLP branches over high-fidelity inputs only.
    alpha : torch.nn.Parameter
        Learnable mixing logits of shape ``[out_features, 3]``.
        The effective weights are computed as ``tanh(alpha)``.
    """

    def __init__(
        self,
        x_dim: int,
        yL_dim: int = 1,
        out_features: int = 1,
        hid_features: int = 5,
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
        """
        Compute the Ada2MF prediction.

        Parameters
        ----------
        x : torch.Tensor
            High-fidelity input tensor of shape ``[N, x_dim]``.
        y_L : torch.Tensor
            Low-fidelity input tensor of shape ``[N, yL_dim]``.

        Returns
        -------
        torch.Tensor
            If ``out_features == 1``:
                Tensor of shape ``[N]``.

            If ``out_features > 1``:
                Tensor of shape ``[N, out_features]``.

        Raises
        ------
        RuntimeError
            If input dimensions do not match the configured ``x_dim`` or ``yL_dim``.

        Notes
        -----
        Inputs are flattened internally to shape ``[N, d]``.
        The prediction is a weighted combination of three branches,
        with weights constrained to ``[-1, 1]`` via ``tanh``.
        """
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