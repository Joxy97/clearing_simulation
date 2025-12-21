from __future__ import annotations

import torch


def prices_to_returns(prices: torch.Tensor) -> torch.Tensor:
    """
    prices: (T, N) float
    returns: (T, N) float, diff with first row = 0
    """
    if prices.ndim != 2:
        raise ValueError(f"prices must be (T,N), got {tuple(prices.shape)}")
    rets = torch.zeros_like(prices)
    rets[1:] = prices[1:] - prices[:-1]
    return rets


def prices_to_states(prices: torch.Tensor) -> torch.Tensor:
    """
    Computes raw market states from prices via returns:
      V_med_abs[t]  = median_i(|r_ti|)
      C_abs_mean[t] = |mean_i(r_ti)|
    Output has length T-1 (aligned with y_next = returns[t+1]).
    """
    rets = prices_to_returns(prices)
    r_today = rets[:-1]
    V = torch.median(torch.abs(r_today), dim=1).values
    C = torch.abs(torch.mean(r_today, dim=1))
    return torch.stack([V, C], dim=1)
