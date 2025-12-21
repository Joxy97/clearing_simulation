from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional

import torch


class VipStatus(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


class LiquidityStatus(Enum):
    LIQUID = auto()
    DEFAULT = auto()
    BANKRUPT = auto()


@dataclass
class Trade:
    day: int
    instrument: int
    amount: float
    accepted: bool


class Client:
    def __init__(
        self,
        n_instruments: int,
        margin_func: Callable[[torch.Tensor], float],
        device: Optional[torch.device] = None,
        income: Optional[int] = None,
        income_frequency: Optional[int] = None,
        client_id: Optional[int] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.client_id = client_id

        self.portfolio = self._init_portfolio(n_instruments).to(self.device)
        self.portfolio_history: List[torch.Tensor] = [self.portfolio.clone()]

        if income is None:
            self.income = int(torch.randint(100, 10001, (1,)).item())
        else:
            self.income = int(income)

        if income_frequency is None:
            self.income_frequency = int(torch.randint(7, 31, (1,)).item())
        else:
            self.income_frequency = int(income_frequency)

        self.wealth: float = float(3 * self.income)
        self.wealth_history: List[float] = [self.wealth]

        self.recklessness: float = 0.5

        initial_margin = float(margin_func(self.portfolio))
        self.collateral: float = initial_margin
        self.collateral_history: List[float] = [self.collateral]

        self.pnl_history: List[float] = []
        self.top_up_history: List[float] = []
        self.top_up_trust: float = 0.0

        self.current_trades: List[Trade] = []
        self.trades_history: List[Trade] = []

        self.vip_status: VipStatus = self._compute_vip_status()
        self.liquidity_status: LiquidityStatus = LiquidityStatus.LIQUID

    @staticmethod
    def _init_portfolio(n_instruments: int) -> torch.Tensor:
        positions = torch.randint(-1000, 1001, (n_instruments,), dtype=torch.int32)
        mask = (torch.rand(n_instruments) < 0.3).to(torch.int32)
        positions = positions * mask
        return positions.to(torch.float32)

    def _compute_vip_status(self) -> VipStatus:
        ratio = self.income / max(self.income_frequency, 1)
        if ratio < 200:
            return VipStatus.LOW
        if ratio < 800:
            return VipStatus.MEDIUM
        return VipStatus.HIGH

    def _update_top_up_trust(self) -> None:
        if not self.top_up_history:
            self.top_up_trust = 0.0
            return
        vals = torch.tensor(self.top_up_history, dtype=torch.float32, device=self.device)
        self.top_up_trust = float(torch.quantile(vals, 0.01))

    def _update_liquidity_status(self) -> None:
        if self.wealth <= 0 and self.liquidity_status == LiquidityStatus.LIQUID:
            self.liquidity_status = LiquidityStatus.BANKRUPT

    def update_recklessness(self, sigma: float = 0.05) -> None:
        noise = float(torch.randn(()).item()) * sigma
        self.recklessness = float(self.recklessness + noise)
        self.recklessness = max(0.0, min(1.0, self.recklessness))

    def generate_trade(self) -> torch.Tensor:
        p = self.portfolio
        delta = torch.zeros_like(p)

        nonzero_mask = p != 0
        zero_mask = ~nonzero_mask

        if nonzero_mask.any():
            p_nz = p[nonzero_mask]
            mean = p_nz * (self.recklessness / 2.0)
            std = torch.sqrt(p_nz.abs()) + 1e-6
            delta_nz = mean + std * torch.randn_like(p_nz)
            delta[nonzero_mask] = delta_nz

        if zero_mask.any():
            rand = torch.rand_like(p[zero_mask])
            trade_mask = rand < 0.05
            zero_part = torch.zeros_like(rand)
            if trade_mask.any():
                zero_part[trade_mask] = 100.0 * torch.rand_like(rand[trade_mask])
            delta[zero_mask] = zero_part

        return delta

    def margin_called(self, amount: float) -> bool:
        if self.liquidity_status != LiquidityStatus.LIQUID:
            return False
        if amount <= self.recklessness * self.wealth:
            self.wealth -= amount
            self.collateral += amount
            self.wealth_history.append(self.wealth)
            self.collateral_history.append(self.collateral)
            self.top_up_history.append(amount)
            self._update_top_up_trust()
            self._update_liquidity_status()
            return True
        self.liquidity_status = LiquidityStatus.DEFAULT
        return False

    def apply_pnl(self, pnl_value: float) -> None:
        self.wealth += pnl_value
        self.wealth_history.append(self.wealth)
        self.pnl_history.append(pnl_value)
        self._update_liquidity_status()

    def apply_income_if_due(self, day: int) -> None:
        if self.liquidity_status != LiquidityStatus.LIQUID:
            return
        if day % self.income_frequency == 0:
            self.wealth += self.income
            self.wealth_history.append(self.wealth)
            self._update_liquidity_status()

    def set_collateral(self, new_collateral: float) -> None:
        self.collateral = float(new_collateral)
        self.collateral_history.append(self.collateral)

    def liquidate_portfolio(self) -> None:
        self.portfolio = torch.zeros_like(self.portfolio)
        self.portfolio_history.append(self.portfolio.clone())

    def record_trades(
        self,
        day: int,
        delta_portfolio: torch.Tensor,
        accepted_mask: torch.Tensor,
    ) -> None:
        accepted_mask = accepted_mask.to(torch.bool)
        if delta_portfolio.shape != self.portfolio.shape:
            raise ValueError("delta_portfolio must match portfolio shape")

        accepted_delta = delta_portfolio * accepted_mask.to(delta_portfolio.dtype)
        self.portfolio = (self.portfolio + accepted_delta).to(self.device)
        self.portfolio_history.append(self.portfolio.clone())

        self.current_trades = []
        for i in range(self.portfolio.numel()):
            amt = float(delta_portfolio[i].item())
            if amt == 0.0:
                continue
            accepted = bool(accepted_mask[i].item())
            t = Trade(day=day, instrument=i, amount=amt, accepted=accepted)
            self.current_trades.append(t)
            self.trades_history.append(t)
