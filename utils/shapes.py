from __future__ import annotations

from typing import Tuple
from torch import Tensor


def ensure_2d_y(y: Tensor) -> Tensor:
    """
    Ensure y is shape [N, D]. If y is [N], convert to [N, 1].
    """
    if y.dim() == 1:
        return y.unsqueeze(-1)
    if y.dim() == 2:
        return y
    raise ValueError(f"Expected y with shape [N] or [N,D], got {tuple(y.shape)}")


def ensure_2d_X(X: Tensor) -> Tensor:
    """
    Ensure X is shape [N, d]. If X is [N], convert to [N, 1].
    """
    if X.dim() == 1:
        return X.unsqueeze(-1)
    if X.dim() == 2:
        return X
    raise ValueError(f"Expected X with shape [N] or [N,d], got {tuple(X.shape)}")


def normalize_to_bqd(X: Tensor) -> Tuple[Tensor, int, int, int]:
    """
    Normalize X to [B, q, d].

    Accepts:
      - [d]      -> [1, 1, d]
      - [q, d]   -> [1, q, d]
      - [B, q, d] stays

    Returns:
      Xb, B, q, d
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
    Flatten [B, q, d] -> [B*q, d]
    """
    if Xb.dim() != 3:
        raise ValueError(f"Expected [B,q,d], got {tuple(Xb.shape)}")
    B, q, d = Xb.shape
    return Xb.reshape(B * q, d)


def unflatten_N_to_bqd(Y: Tensor, B: int, q: int) -> Tensor:
    """
    Unflatten [B*q, D] -> [B, q, D]
    """
    if Y.dim() != 2:
        raise ValueError(f"Expected [N,D], got {tuple(Y.shape)}")
    return Y.view(B, q, Y.shape[-1])
