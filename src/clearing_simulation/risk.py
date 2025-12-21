from __future__ import annotations

import torch


def pnl(portfolios: torch.Tensor, scenarios: torch.Tensor) -> torch.Tensor:
    return portfolios @ scenarios.T


def loss(portfolios: torch.Tensor, scenarios: torch.Tensor) -> torch.Tensor:
    return -(portfolios @ scenarios.T)


def margin(
    portfolios: torch.Tensor,
    scenarios: torch.Tensor,
    alpha: float = 0.99,
) -> torch.Tensor:
    l = loss(portfolios, scenarios)
    VaR = torch.quantile(l, q=alpha, dim=1, keepdim=True)
    mask = l >= VaR
    tail_count = torch.clamp(mask.sum(dim=1), min=1)
    ES = (l * mask).sum(dim=1) / tail_count
    return ES


def margin_and_tail_stats(
    portfolios: torch.Tensor,
    scenarios: torch.Tensor,
    alpha: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor]:
    l = loss(portfolios, scenarios)
    VaR = torch.quantile(l, q=alpha, dim=1, keepdim=True)
    mask = l >= VaR
    tail_count = torch.clamp(mask.sum(dim=1), min=1)
    ES = (l * mask).sum(dim=1) / tail_count
    tail_sum = mask.to(scenarios.dtype) @ scenarios
    tail_mean = tail_sum / tail_count.unsqueeze(1)
    return ES, tail_mean
