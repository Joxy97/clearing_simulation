from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def _quantile_edges_1d(x: torch.Tensor, qs: List[float], eps: float = 1e-12) -> torch.Tensor:
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if len(qs) < 2:
        raise ValueError("Need at least 2 quantiles")
    if x.numel() == 0:
        raise ValueError("Cannot fit edges on empty data")

    x = x.to(dtype=torch.float32)
    edges = torch.quantile(x, torch.tensor(qs, device=x.device, dtype=x.dtype))

    edges_unique = torch.unique(edges)
    if edges_unique.numel() < len(qs):
        mn = torch.min(x).item()
        mx = torch.max(x).item()
        edges = torch.linspace(mn - eps, mx + eps, steps=len(qs), device=x.device, dtype=x.dtype)
    else:
        edges = edges
        edges[0] -= eps
        edges[-1] += eps
    return edges


def _digitize(x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    if edges.ndim != 1:
        raise ValueError("edges must be 1D")
    K = edges.numel() - 1
    if K <= 0:
        raise ValueError("edges must have at least 2 values")
    thr = edges[1:-1].contiguous()
    return torch.bucketize(x.contiguous(), thr, right=False).to(torch.long)


def _onehot(bins: torch.Tensor, K: int) -> torch.Tensor:
    if bins.dtype != torch.long:
        bins = bins.long()
    return F.one_hot(bins, num_classes=K).to(torch.int8)


@dataclass
class StateEncoderParams:
    V_edges: torch.Tensor
    C_edges: torch.Tensor
    K_v: int
    K_c: int


def encode_states(
    raw_states: torch.Tensor,
    K_v: int = 4,
    K_c: int = 4,
    *,
    params: Optional[StateEncoderParams] = None,
    fit: bool = False,
    train_slice: Optional[slice] = None,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, StateEncoderParams]:
    if raw_states.ndim != 2 or raw_states.shape[1] != 2:
        raise ValueError(f"raw_states must be (T,2), got {tuple(raw_states.shape)}")

    device = raw_states.device
    V = raw_states[:, 0].float()
    C = raw_states[:, 1].float()

    if fit:
        if train_slice is None:
            raise ValueError("fit=True requires train_slice")
        V_edges = _quantile_edges_1d(V[train_slice], qs=torch.linspace(0, 1, K_v + 1).tolist(), eps=eps).to(device)
        C_edges = _quantile_edges_1d(C[train_slice], qs=torch.linspace(0, 1, K_c + 1).tolist(), eps=eps).to(device)
        params = StateEncoderParams(V_edges=V_edges, C_edges=C_edges, K_v=K_v, K_c=K_c)
    else:
        if params is None:
            raise ValueError("Provide params or set fit=True")
        if params.K_v != K_v or params.K_c != K_c:
            raise ValueError("K_v/K_c mismatch with provided params")

    V_bins = _digitize(V, params.V_edges)
    C_bins = _digitize(C, params.C_edges)

    V_oh = _onehot(V_bins, K_v)
    C_oh = _onehot(C_bins, K_c)

    return torch.cat([V_oh, C_oh], dim=1), params


@dataclass
class ReturnEncoderParams:
    loss_quantiles: List[float]
    loss_edges_by_inst: torch.Tensor
    N: int


def encode_returns(
    raw_returns_next: torch.Tensor,
    percentiles: List[float],
    *,
    params: Optional[ReturnEncoderParams] = None,
    fit: bool = False,
    train_slice: Optional[slice] = None,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, ReturnEncoderParams]:
    if raw_returns_next.ndim != 2:
        raise ValueError(f"raw_returns_next must be (T,N), got {tuple(raw_returns_next.shape)}")
    if len(percentiles) != 7:
        raise ValueError("For 6 loss bins you need 7 percentiles (edges).")

    T, N = raw_returns_next.shape
    device = raw_returns_next.device
    X = raw_returns_next.float()

    if fit:
        if train_slice is None:
            raise ValueError("fit=True requires train_slice")
        loss_edges = torch.empty((N, len(percentiles)), device=device, dtype=torch.float32)
        for i in range(N):
            r_tr = X[train_slice, i]
            neg = r_tr[r_tr < 0]
            if neg.numel() == 0:
                neg = torch.tensor([-1.0, -eps], device=device, dtype=torch.float32)
            edges_i = _quantile_edges_1d(neg, qs=percentiles, eps=eps)
            loss_edges[i] = edges_i
        params = ReturnEncoderParams(
            loss_quantiles=percentiles,
            loss_edges_by_inst=loss_edges,
            N=N,
        )
    else:
        if params is None:
            raise ValueError("Provide params or set fit=True")
        if params.N != N:
            raise ValueError(f"N mismatch: params.N={params.N} vs data N={N}")

    out = torch.zeros((T, N, 7), device=device, dtype=torch.int8)

    is_gain = (X >= 0)
    out[is_gain, 0] = 1

    is_loss = ~is_gain
    if is_loss.any():
        for i in range(N):
            mask_i = is_loss[:, i]
            if not mask_i.any():
                continue
            r_i = X[mask_i, i]
            edges_i = params.loss_edges_by_inst[i]
            k = _digitize(r_i, edges_i)
            L_index = 6 - k
            out[mask_i, i, L_index] = 1

    return out.reshape(T, N * 7), params


def decode_states(
    states_onehot: torch.Tensor,
    params: StateEncoderParams,
    *,
    return_midpoints: bool = True,
) -> torch.Tensor:
    K_v, K_c = params.K_v, params.K_c
    if states_onehot.ndim != 2 or states_onehot.shape[1] != K_v + K_c:
        raise ValueError("Bad shape for states_onehot")

    V_oh = states_onehot[:, :K_v]
    C_oh = states_onehot[:, K_v:K_v + K_c]
    V_bin = torch.argmax(V_oh, dim=1)
    C_bin = torch.argmax(C_oh, dim=1)

    if not return_midpoints:
        return torch.stack([V_bin.float(), C_bin.float()], dim=1)

    V_mid = 0.5 * (params.V_edges[V_bin] + params.V_edges[V_bin + 1])
    C_mid = 0.5 * (params.C_edges[C_bin] + params.C_edges[C_bin + 1])
    return torch.stack([V_mid, C_mid], dim=1)


def build_return_midpoints(params: ReturnEncoderParams, device: Optional[torch.device] = None) -> torch.Tensor:
    edges = params.loss_edges_by_inst
    mid_neg = 0.5 * (edges[:, :-1] + edges[:, 1:])
    mid_loss = torch.flip(mid_neg, dims=[1])
    rep = torch.zeros((params.N, 7), device=edges.device, dtype=torch.float32)
    rep[:, 1:] = mid_loss
    if device is not None:
        rep = rep.to(device)
    return rep


def decode_returns(
    returns_onehot: torch.Tensor,
    params: ReturnEncoderParams,
    *,
    return_representative: Optional[torch.Tensor] = None,
    return_midpoints: bool = True,
) -> torch.Tensor:
    T = returns_onehot.shape[0]
    N = params.N
    if returns_onehot.ndim != 2 or returns_onehot.shape[1] != N * 7:
        raise ValueError(f"returns_onehot must be (T, {N * 7})")

    X = returns_onehot.view(T, N, 7)
    idx = torch.argmax(X, dim=2)

    if return_representative is not None:
        if return_representative.shape != (N, 7):
            raise ValueError("return_representative must be (N,7)")
        rep = return_representative.to(device=returns_onehot.device, dtype=torch.float32)
        out = torch.gather(rep.unsqueeze(0).expand(T, -1, -1), 2, idx.unsqueeze(2))
        return out.squeeze(2)

    if not return_midpoints:
        return idx.float()

    rep = build_return_midpoints(params, device=returns_onehot.device)
    out = torch.gather(rep.unsqueeze(0).expand(T, -1, -1), 2, idx.unsqueeze(2))
    return out.squeeze(2)
