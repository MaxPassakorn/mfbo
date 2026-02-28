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


class MFNN(nn.Module):
    r"""
    Multi-Fidelity Neural Network (MFNN) architecture.

    MFNN blends a linear trend model and a nonlinear MLP using a learnable
    mixing weight per output dimension. This design helps the model represent
    both global linear behavior and localized nonlinear corrections.

    For each output dimension :math:`e`, MFNN computes

    .. math::

        z_{\mathrm{lin}}(x) &= \mathrm{Linear}_e(x), \\
        z_{\mathrm{nl}}(x)  &= \mathrm{MLP}_e(x), \\
        a_e                 &= \sigma(\alpha_e), \\
        y_e(x)              &= a_e \, z_{\mathrm{lin}}(x)
                               + (1-a_e)\, z_{\mathrm{nl}}(x),

    where :math:`\sigma(\cdot)` is the logistic sigmoid and :math:`\alpha_e`
    is a learnable logit parameter.

    Notes
    -----
    - This module defines only the neural architecture. Multi-fidelity coupling
      (e.g., concatenating low-fidelity predictions with inputs) should be done
      outside this class, typically in an ensemble or wrapper model.
    - Each output dimension uses an independent linear head, nonlinear head,
      and mixing parameter :math:`\alpha_e`.

    Parameters
    ----------
    in_features : int
        Number of input features (dimension of input ``x``).
    out_features : int, default=1
        Number of output dimensions.
    hid_features : int, default=5
        Width of hidden layers in the nonlinear (MLP) head.
    n_hid_layers : int, default=2
        Number of hidden layers in the nonlinear (MLP) head.
    activation : torch.nn.Module or None, default=None
        Activation function used in the nonlinear head. If ``None``,
        :class:`torch.nn.Mish` is used.
    bias : bool, default=True
        Whether linear layers include bias terms.

    Attributes
    ----------
    linear_heads : torch.nn.ModuleList
        Linear heads, one per output dimension. Each head maps
        ``in_features -> 1``.
    nonlinear_heads : torch.nn.ModuleList
        Nonlinear heads (MLPs), one per output dimension. Each head maps
        ``in_features -> 1``.
    raw_alpha : torch.nn.Parameter
        Learnable logits controlling the linear/nonlinear mixing weights,
        with shape ``[out_features]``. The mixing weight is computed as
        ``sigmoid(raw_alpha[e])``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hid_features: int = 5,
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
        """
        Compute the MFNN output.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[N, d]`` or any tensor that can be reshaped
            to that form. The first dimension is treated as the batch dimension.

        Returns
        -------
        torch.Tensor
            If ``out_features == 1``:
                Tensor of shape ``[N]``.

            If ``out_features > 1``:
                Tensor of shape ``[N, out_features]``.

        Notes
        -----
        The input is flattened internally to shape ``[N, d]`` before evaluation.
        The predictive output is formed as a convex combination of a linear head
        and a nonlinear (MLP) head per output dimension.
        """
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