from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .risk import margin


@dataclass
class CCP:
    clearing_members: List["ClearingMember"]
    instrument_indices: Any
    default_fund: float = 0.0
    name: Optional[str] = None
    device: Optional[torch.device] = None

    ccp_margin_history: List[float] = field(default_factory=list)
    cm_margin_history: List[torch.Tensor] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.clearing_members:
            raise ValueError("CCP must be initialized with at least one ClearingMember.")
        if self.device is None:
            self.device = self.clearing_members[0].device

        if isinstance(self.instrument_indices, torch.Tensor):
            idx = self.instrument_indices.to(dtype=torch.long, device=self.device)
        else:
            idx = torch.tensor(self.instrument_indices, dtype=torch.long, device=self.device)
        self.instrument_indices = torch.unique(idx)

    def cm_portfolios_tensor(self) -> torch.Tensor:
        idx = self.instrument_indices
        cm_ports = []
        for cm in self.clearing_members:
            p_full = cm.aggregate_portfolio().to(self.device)
            p_sel = p_full[idx]
            cm_ports.append(p_sel)
        return torch.stack(cm_ports, dim=0)

    def ccp_net_portfolio(self) -> torch.Tensor:
        return self.cm_portfolios_tensor().sum(dim=0)

    def compute_cm_margins(self, scenarios: torch.Tensor, alpha: float = 0.99) -> torch.Tensor:
        scenarios = scenarios.to(self.device)
        idx = self.instrument_indices
        scenarios_sel = scenarios[:, idx]
        cm_ports = self.cm_portfolios_tensor()
        margins_cm = margin(cm_ports, scenarios_sel, alpha=alpha)
        self.cm_margin_history.append(margins_cm.detach().cpu())
        return margins_cm

    def compute_ccp_margin(self, scenarios: torch.Tensor, alpha: float = 0.99) -> float:
        scenarios = scenarios.to(self.device)
        idx = self.instrument_indices
        scenarios_sel = scenarios[:, idx]
        net_port = self.ccp_net_portfolio()
        margin_ccp_tensor = margin(
            net_port.unsqueeze(0),
            scenarios_sel,
            alpha=alpha,
        )[0]
        margin_ccp = float(margin_ccp_tensor.item())
        self.ccp_margin_history.append(margin_ccp)
        return margin_ccp

    def cm_funds_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [float(cm.cm_funds) for cm in self.clearing_members],
            dtype=torch.float32,
            device=self.device,
        )

    def cm_margin_shortfalls(self, scenarios: torch.Tensor, alpha: float = 0.99) -> torch.Tensor:
        margins_cm = self.compute_cm_margins(scenarios, alpha=alpha)
        funds_cm = self.cm_funds_tensor()
        shortfalls = torch.clamp(margins_cm - funds_cm, min=0.0)
        return shortfalls

    def snapshot_state(self, scenarios: torch.Tensor, alpha: float = 0.99) -> Dict[str, Any]:
        margins_cm = self.compute_cm_margins(scenarios, alpha=alpha)
        funds_cm = self.cm_funds_tensor()
        shortfalls = torch.clamp(margins_cm - funds_cm, min=0.0)
        margin_ccp = self.compute_ccp_margin(scenarios, alpha=alpha)
        net_port = self.ccp_net_portfolio()

        return {
            "cm_margins": margins_cm,
            "cm_funds": funds_cm,
            "cm_shortfalls": shortfalls,
            "ccp_margin": margin_ccp,
            "net_portfolio": net_port,
        }
