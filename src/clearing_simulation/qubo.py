from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import dimod
from dwave.samplers import SimulatedAnnealingSampler

from .risk import margin_and_tail_stats


def _build_trade_mask(
    trades: torch.Tensor,
    min_trade_abs: float = 0.0,
    max_trades_per_client: Optional[int] = None,
) -> torch.Tensor:
    mask = trades.abs() > float(min_trade_abs)
    if max_trades_per_client is None:
        return mask

    C, _ = trades.shape
    final_mask = torch.zeros_like(mask)
    for m in range(C):
        valid = mask[m].nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        if valid.numel() <= max_trades_per_client:
            final_mask[m, valid] = True
            continue
        vals = trades[m, valid].abs()
        topk = torch.topk(vals, k=max_trades_per_client)
        selected = valid[topk.indices]
        final_mask[m, selected] = True
    return final_mask


@torch.no_grad()
def compute_margins_and_sensitivities(
    portfolios: torch.Tensor,
    scenarios: torch.Tensor,
    alpha: float = 0.99,
) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    M0_client, tail_mean_client = margin_and_tail_stats(portfolios, scenarios, alpha=alpha)
    cm_portfolio = portfolios.sum(dim=0)
    M0_cm_tensor, tail_mean_cm = margin_and_tail_stats(cm_portfolio.unsqueeze(0), scenarios, alpha=alpha)
    M0_cm = float(M0_cm_tensor[0].item())
    return M0_client, M0_cm, cm_portfolio, tail_mean_client, tail_mean_cm[0]


@torch.no_grad()
def build_qubo_matrix(
    portfolios: torch.Tensor,
    trades: torch.Tensor,
    collaterals: torch.Tensor,
    cm_funds: float,
    scenarios: torch.Tensor,
    alpha: float = 0.99,
    lambda_client: float = 10.0,
    lambda_cm: float = 10.0,
    trade_value_scale: float = 1.0,
    min_trade_abs: float = 0.0,
    max_trades_per_client: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    device = portfolios.device
    portfolios = portfolios.to(device)
    trades = trades.to(device)
    collaterals = collaterals.to(device)
    scenarios = scenarios.to(device)

    C, _ = portfolios.shape
    M0_client, M0_cm, cm_portfolio, tail_mean_client, tail_mean_cm = compute_margins_and_sensitivities(
        portfolios, scenarios, alpha=alpha
    )

    trade_mask = _build_trade_mask(trades, min_trade_abs=min_trade_abs, max_trades_per_client=max_trades_per_client)
    active_ix = trade_mask.nonzero(as_tuple=False)
    K = active_ix.shape[0]

    if K == 0:
        Q = torch.zeros((0, 0), dtype=torch.float32, device=device)
        return Q, active_ix, M0_client, M0_cm

    deltas = trades[active_ix[:, 0], active_ix[:, 1]]
    a_client = -deltas * tail_mean_client[active_ix[:, 0], active_ix[:, 1]]
    a_cm = -deltas * tail_mean_cm[active_ix[:, 1]]
    v_k = trade_value_scale * deltas

    Q = torch.zeros((K, K), dtype=torch.float32, device=device)
    Q.diagonal().add_(-v_k)

    if lambda_cm != 0.0:
        A_cm = M0_cm - float(cm_funds)
        Q += lambda_cm * torch.outer(a_cm, a_cm)
        Q.diagonal().add_(lambda_cm * (2.0 * A_cm * a_cm))

    if lambda_client != 0.0:
        for m in range(C):
            idx = (active_ix[:, 0] == m).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            a_m = a_client[idx]
            A_m = M0_client[m] - float(collaterals[m])
            Q[idx[:, None], idx] += lambda_client * torch.outer(a_m, a_m)
            Q[idx, idx] += lambda_client * (2.0 * A_m * a_m)

    return Q, active_ix, M0_client, M0_cm


def qubo_to_bqm(Q: torch.Tensor) -> dimod.BinaryQuadraticModel:
    Q_cpu = Q.detach().cpu().float()
    K = Q_cpu.shape[0]
    Q_sym = 0.5 * (Q_cpu + Q_cpu.T)

    linear: Dict[int, float] = {i: float(Q_sym[i, i]) for i in range(K)}
    quadratic: Dict[tuple[int, int], float] = {}

    for i in range(K):
        for j in range(i + 1, K):
            q_ij = float(Q_sym[i, j])
            if q_ij != 0.0:
                quadratic[(i, j)] = q_ij

    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.BINARY)


def solve_qubo(
    Q: torch.Tensor,
    sampler: Optional[dimod.Sampler] = None,
    num_reads: int = 100,
    sampler_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    device = Q.device
    K = Q.shape[0]
    if K == 0:
        return torch.zeros(0, dtype=torch.int8, device=device)

    bqm = qubo_to_bqm(Q)
    if sampler is None:
        sampler = SimulatedAnnealingSampler()
    if sampler_kwargs is None:
        sampler_kwargs = {}

    if hasattr(sampler, "sample") and "num_reads" in sampler.sample.__code__.co_varnames:
        sampler_kwargs.setdefault("num_reads", num_reads)

    sampleset = sampler.sample(bqm, **sampler_kwargs)
    best = sampleset.first.sample
    x_list = [best[i] for i in range(K)]
    return torch.tensor(x_list, dtype=torch.int8, device=device)
