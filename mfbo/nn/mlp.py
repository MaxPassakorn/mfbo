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


class MLP(nn.Module):
    r"""
    Multi-head fully connected neural network.

    This module implements a standard multilayer perceptron (MLP)
    with configurable depth and width. A separate output head is
    constructed for each output dimension.

    The network has the structure:

    .. math::

        x \rightarrow
        \text{Linear} \rightarrow
        \text{Activation} \rightarrow
        \cdots \rightarrow
        \text{Linear}_{\text{out}}

    where each output dimension is modeled by an independent head.

    Notes
    -----
    - Each output dimension uses its own MLP head.
    - There is no shared final linear layer across outputs.
    - This class defines only the neural architecture.
      Training logic and Bayesian optimization wrappers are implemented
      elsewhere in the library.

    Parameters
    ----------
    in_features : int
        Number of input features (dimension of input ``x``).
    out_features : int, default=1
        Number of output dimensions.
    hid_features : int, default=5
        Width of hidden layers.
    n_hid_layers : int, default=2
        Number of hidden layers.
    activation : torch.nn.Module or None, default=None
        Activation function applied after each hidden linear layer.
        If ``None``, :class:`torch.nn.Mish` is used.
    bias : bool, default=True
        Whether linear layers include bias terms.

    Attributes
    ----------
    heads : torch.nn.ModuleList
        List of output heads (one per output dimension).
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
        """
        Compute the network output.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[N, d]`` or any tensor that can
            be reshaped to that form. The first dimension is treated
            as the batch dimension.

        Returns
        -------
        torch.Tensor
            If ``out_features == 1``:
                Tensor of shape ``[N]``.

            If ``out_features > 1``:
                Tensor of shape ``[N, out_features]``.

        Notes
        -----
        The input is flattened internally to shape ``[N, d]`` before
        being passed through each output head.
        """
        x = x.view(x.shape[0], -1)

        outs = [head(x).squeeze(-1) for head in self.heads]  # list of [N]
        if self.out_features == 1:
            return outs[0]  # [N]
        return torch.stack(outs, dim=-1)  # [N, out_features]