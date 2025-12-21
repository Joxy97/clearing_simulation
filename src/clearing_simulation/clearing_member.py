from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .client import Client, LiquidityStatus
from .qubo import build_qubo_matrix, solve_qubo


@dataclass
class ClearingMember:
    clients: List[Client]
    cm_funds: float
    name: Optional[str] = None
    device: Optional[torch.device] = None

    cm_portfolio_history: List[torch.Tensor] = field(default_factory=list)
    cm_margin_history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = self.clients[0].device if self.clients else torch.device("cpu")
        if self.clients:
            self.cm_portfolio_history.append(self.aggregate_portfolio())

    def portfolios_tensor(self) -> torch.Tensor:
        if not self.clients:
            raise ValueError("ClearingMember has no clients.")
        return torch.stack([c.portfolio.to(self.device) for c in self.clients], dim=0)

    def collaterals_tensor(self) -> torch.Tensor:
        if not self.clients:
            raise ValueError("ClearingMember has no clients.")
        return torch.tensor(
            [float(c.collateral) for c in self.clients],
            dtype=torch.float32,
            device=self.device,
        )

    def wealth_tensor(self) -> torch.Tensor:
        if not self.clients:
            raise ValueError("ClearingMember has no clients.")
        return torch.tensor(
            [float(c.wealth) for c in self.clients],
            dtype=torch.float32,
            device=self.device,
        )

    def aggregate_portfolio(self) -> torch.Tensor:
        return self.portfolios_tensor().sum(dim=0)

    def propose_trades(self) -> torch.Tensor:
        if not self.clients:
            raise ValueError("ClearingMember has no clients.")
        trades_list: List[torch.Tensor] = []
        for c in self.clients:
            if c.liquidity_status in (LiquidityStatus.DEFAULT, LiquidityStatus.BANKRUPT):
                trades_list.append(torch.zeros_like(c.portfolio))
            else:
                trades_list.append(c.generate_trade())
        return torch.stack(trades_list, dim=0).to(self.device)

    def execute_trades(
        self,
        day: int,
        trades: torch.Tensor,
        accepted_mask: torch.Tensor,
    ) -> None:
        trades = trades.to(self.device)
        accepted_mask = accepted_mask.to(torch.bool).to(self.device)

        if trades.shape[0] != len(self.clients):
            raise ValueError("trades.shape[0] must equal number of clients.")
        if trades.shape != accepted_mask.shape:
            raise ValueError("trades and accepted_mask must have same shape.")

        for m, client in enumerate(self.clients):
            client.record_trades(day, trades[m], accepted_mask[m])

        self.cm_portfolio_history.append(self.aggregate_portfolio())

    def build_qubo_for_trades(
        self,
        trades: torch.Tensor,
        scenarios: torch.Tensor,
        alpha: float = 0.99,
        lambda_client: float = 10.0,
        lambda_cm: float = 10.0,
        trade_value_scale: float = 1.0,
        min_trade_abs: float = 0.0,
        max_trades_per_client: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        portfolios = self.portfolios_tensor()
        collaterals = self.collaterals_tensor()
        Q, active_ix, M0_client, M0_cm = build_qubo_matrix(
            portfolios=portfolios,
            trades=trades,
            collaterals=collaterals,
            cm_funds=self.cm_funds,
            scenarios=scenarios,
            alpha=alpha,
            lambda_client=lambda_client,
            lambda_cm=lambda_cm,
            trade_value_scale=trade_value_scale,
            min_trade_abs=min_trade_abs,
            max_trades_per_client=max_trades_per_client,
        )
        return Q, active_ix, M0_client, M0_cm

    def decide_trades_qubo(
        self,
        trades: torch.Tensor,
        scenarios: torch.Tensor,
        alpha: float = 0.99,
        lambda_client: float = 10.0,
        lambda_cm: float = 10.0,
        trade_value_scale: float = 1.0,
        min_trade_abs: float = 0.0,
        max_trades_per_client: Optional[int] = None,
        sampler=None,
        num_reads: int = 100,
        sampler_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        trades = trades.to(self.device)
        scenarios = scenarios.to(self.device)

        Q, active_ix, M0_client, M0_cm = self.build_qubo_for_trades(
            trades=trades,
            scenarios=scenarios,
            alpha=alpha,
            lambda_client=lambda_client,
            lambda_cm=lambda_cm,
            trade_value_scale=trade_value_scale,
            min_trade_abs=min_trade_abs,
            max_trades_per_client=max_trades_per_client,
        )

        x = solve_qubo(
            Q,
            sampler=sampler,
            num_reads=num_reads,
            sampler_kwargs=sampler_kwargs,
        )

        C, I = trades.shape
        accepted_mask = torch.zeros((C, I), dtype=torch.bool, device=self.device)
        for k in range(active_ix.shape[0]):
            m, i = active_ix[k]
            if x[k].item() == 1:
                accepted_mask[m, i] = True

        info = {
            "Q": Q,
            "active_ix": active_ix,
            "M0_client": M0_client,
            "M0_cm": M0_cm,
            "x": x,
        }
        return accepted_mask, info
