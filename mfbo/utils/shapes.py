from __future__ import annotations

"""
Tensor shape utilities.

This module provides helper functions for normalizing and reshaping
tensors used in surrogate models and ensemble inference.

All models in this library follow the BoTorch shape convention:

- ``B`` : broadcast batch dimension
- ``q`` : number of query points
- ``d`` : input dimension
- ``E`` : number of outputs

These utilities ensure consistent handling of input/output shapes
across models and ensemble wrappers.
"""

from typing import Tuple
from torch import Tensor


def ensure_2d_y(y: Tensor) -> Tensor:
    """
    Ensure target tensor has shape ``[N, D]``.

    Parameters
    ----------
    y : torch.Tensor
        Target tensor of shape ``[N]`` or ``[N, D]``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``[N, D]``. If input is ``[N]``,
        it is converted to ``[N, 1]``.

    Raises
    ------
    ValueError
        If ``y`` does not have 1 or 2 dimensions.

    Notes
    -----
    This function standardizes target shapes for training
    multi-output models.
    """
    if y.dim() == 1:
        return y.unsqueeze(-1)
    if y.dim() == 2:
        return y
    raise ValueError(f"Expected y with shape [N] or [N,D], got {tuple(y.shape)}")


def ensure_2d_X(X: Tensor) -> Tensor:
    """
    Ensure input tensor has shape ``[N, d]``.

    Parameters
    ----------
    X : torch.Tensor
        Input tensor of shape ``[N]`` or ``[N, d]``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``[N, d]``. If input is ``[N]``,
        it is converted to ``[N, 1]``.

    Raises
    ------
    ValueError
        If ``X`` does not have 1 or 2 dimensions.

    Notes
    -----
    This function standardizes input shapes for model training.
    """
    if X.dim() == 1:
        return X.unsqueeze(-1)
    if X.dim() == 2:
        return X
    raise ValueError(f"Expected X with shape [N] or [N,d], got {tuple(X.shape)}")


def normalize_to_bqd(X: Tensor) -> Tuple[Tensor, int, int, int]:
    """
    Normalize input tensor to shape ``[B, q, d]``.

    Accepted input shapes
    ---------------------
    - ``[d]``        → interpreted as a single point → ``[1, 1, d]``
    - ``[q, d]``     → single batch → ``[1, q, d]``
    - ``[B, q, d]``  → unchanged

    Parameters
    ----------
    X : torch.Tensor
        Input tensor of dimension 1, 2, or 3.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[B, q, d]``.

    Raises
    ------
    ValueError
        If ``X`` does not have 1, 2, or 3 dimensions.

    Notes
    -----
    This normalization follows the BoTorch input convention for
    batch-aware models and acquisition functions.
    """
    if X.dim() == 1:
        Xb = X.unsqueeze(0).unsqueeze(0)
    elif X.dim() == 2:
        Xb = X.unsqueeze(0)
    elif X.dim() == 3:
        Xb = X
    else:
        raise ValueError(f"Expected X with 1/2/3 dims, got {tuple(X.shape)}")

    return Xb


def flatten_bqd(Xb: Tensor) -> Tensor:
    """
    Flatten a batched query tensor.

    Converts:

        ``[B, q, d]`` → ``[B*q, d]``

    Parameters
    ----------
    Xb : torch.Tensor
        Tensor of shape ``[B, q, d]``.

    Returns
    -------
    torch.Tensor
        Flattened tensor of shape ``[B*q, d]``.

    Raises
    ------
    ValueError
        If ``Xb`` does not have exactly 3 dimensions.

    Notes
    -----
    Useful for evaluating neural networks that expect
    2D input tensors.
    """
    if Xb.dim() != 3:
        raise ValueError(f"Expected [B,q,d], got {tuple(Xb.shape)}")
    B, q, d = Xb.shape
    return Xb.reshape(B * q, d)


def unflatten_N_to_bqd(Y: Tensor, B: int, q: int) -> Tensor:
    """
    Reshape flattened predictions back to batched form.

    Converts:

        ``[B*q, D]`` → ``[B, q, D]``

    Parameters
    ----------
    Y : torch.Tensor
        Tensor of shape ``[N, D]`` where ``N = B*q``.
    B : int
        Batch size.
    q : int
        Number of query points per batch.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``[B, q, D]``.

    Raises
    ------
    ValueError
        If ``Y`` does not have exactly 2 dimensions.

    Notes
    -----
    This is the inverse operation of :func:`flatten_bqd`.
    """
    if Y.dim() != 2:
        raise ValueError(f"Expected [N,D], got {tuple(Y.shape)}")
    return Y.view(B, q, Y.shape[-1])
