from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .client import Client
from .clearing_member import ClearingMember
from .ccp import CCP
from .risk import margin


def make_margin_func(scenarios: torch.Tensor, alpha: float) -> Any:
    def _margin_func(portfolio: torch.Tensor) -> float:
        m = margin(portfolio.unsqueeze(0), scenarios, alpha=alpha)[0]
        return float(m.item())
    return _margin_func


def build_clearing_members(
    cm_spec: Dict[str, Dict[str, Any]],
    *,
    n_instruments: int,
    margin_func,
    device: torch.device,
) -> Dict[str, ClearingMember]:
    clearing_members: Dict[str, ClearingMember] = {}
    client_id = 0
    for cm_name, spec in cm_spec.items():
        n_clients = int(spec.get("n_clients", 0))
        cm_funds = float(spec.get("cm_funds", 0.0))
        cm_clients = []
        for _ in range(n_clients):
            c = Client(
                n_instruments=n_instruments,
                margin_func=margin_func,
                device=device,
                client_id=client_id,
            )
            cm_clients.append(c)
            client_id += 1
        clearing_members[cm_name] = ClearingMember(
            clients=cm_clients,
            cm_funds=cm_funds,
            name=cm_name,
            device=device,
        )
    return clearing_members


def build_ccps(
    ccp_spec: Dict[str, Dict[str, Any]],
    *,
    clearing_members: Dict[str, ClearingMember],
    n_instruments: int,
    device: torch.device,
) -> Dict[str, CCP]:
    ccps: Dict[str, CCP] = {}
    for ccp_name, spec in ccp_spec.items():
        indices = spec.get("instrument_indices", "all")
        if indices == "all":
            indices = list(range(n_instruments))
        ccp = CCP(
            clearing_members=list(clearing_members.values()),
            instrument_indices=indices,
            default_fund=float(spec.get("default_fund", 0.0)),
            name=ccp_name,
            device=device,
        )
        ccps[ccp_name] = ccp
    return ccps


def _choose_market_index(strategy: str, day: int, T_total: int, device: torch.device) -> int:
    if strategy == "sequential":
        return day % T_total
    return int(torch.randint(0, T_total, (1,), device=device).item())


def simulate_one_day(
    day: int,
    *,
    clearing_members: Dict[str, ClearingMember],
    ccps: Dict[str, CCP],
    scenario_gen,
    state_oh: torch.Tensor,
    returns_next: torch.Tensor,
    alpha: float = 0.99,
    scenarios_per_day: int = 1000,
    burn_in: int = 500,
    thin: int = 10,
    lambda_client: float = 10.0,
    lambda_cm: float = 10.0,
    trade_value_scale: float = 1.0,
    min_trade_abs: float = 0.0,
    max_trades_per_client: Optional[int] = None,
    qubo_sampler=None,
    qubo_sampler_kwargs: Optional[Dict[str, Any]] = None,
    state_index_strategy: str = "random",
    liquidate_on_default: bool = True,
    cm_absorbs_shortfall: bool = True,
    include_details: bool = False,
    include_portfolios: bool = True,
    include_scenarios: bool = False,
    include_returns: bool = False,
) -> Dict[str, Any]:
    if not clearing_members:
        raise ValueError("simulate_one_day: no clearing members provided.")

    first_cm = next(iter(clearing_members.values()))
    device = first_cm.device

    T_total = state_oh.size(0)
    if T_total == 0:
        raise ValueError("simulate_one_day: state_oh has zero length.")

    idx = _choose_market_index(state_index_strategy, day, T_total, device)
    market_state = state_oh[idx].to(device)
    real_ret = returns_next[idx].to(device)

    scenarios = scenario_gen.sample(
        state_onehot=market_state,
        n_samples=scenarios_per_day,
        burn_in=burn_in,
        thin=thin,
    )

    system_total_client_collateral = 0.0
    system_total_client_margin = 0.0
    system_total_cm_funds = 0.0
    system_total_active_trades = 0
    system_total_accepted_trades = 0
    system_total_zero_collateral_clients = 0

    cms_metrics: Dict[str, Any] = {}
    details: Dict[str, Any] = {}
    if include_details:
        details["market_state"] = market_state
        if include_returns:
            details["real_returns"] = real_ret
        if include_scenarios:
            details["scenarios"] = scenarios

    with torch.no_grad():
        for cm_name, cm in clearing_members.items():
            client_start = []
            recklessness_before = []
            if include_details:
                for client in cm.clients:
                    client_start.append(
                        {
                            "client_id": client.client_id,
                            "liquidity_status": client.liquidity_status.name,
                            "wealth": float(client.wealth),
                            "collateral": float(client.collateral),
                            "recklessness": float(client.recklessness),
                            "vip_status": client.vip_status.name,
                            "portfolio": client.portfolio.clone() if include_portfolios else None,
                        }
                    )
                    recklessness_before.append(float(client.recklessness))

            for client in cm.clients:
                client.update_recklessness()

            trades = cm.propose_trades()
            trade_mask = (trades != 0.0)
            num_active_trades = int(trade_mask.sum().item())

            accepted_mask, qubo_info = cm.decide_trades_qubo(
                trades=trades,
                scenarios=scenarios,
                alpha=alpha,
                lambda_client=lambda_client,
                lambda_cm=lambda_cm,
                trade_value_scale=trade_value_scale,
                min_trade_abs=min_trade_abs,
                max_trades_per_client=max_trades_per_client,
                sampler=qubo_sampler,
                num_reads=100,
                sampler_kwargs=qubo_sampler_kwargs,
            )
            x_sol = qubo_info["x"]
            num_accepted_trades = int(x_sol.sum().item()) if x_sol.numel() > 0 else 0

            cm.execute_trades(day, trades, accepted_mask)

            portfolios_cm = cm.portfolios_tensor()
            real_pnl_clients = portfolios_cm @ real_ret

            income_applied = []
            for i_client, client in enumerate(cm.clients):
                pnl_value = float(real_pnl_clients[i_client].item())
                client.apply_pnl(pnl_value)
                income_due = (
                    client.liquidity_status == client.liquidity_status.LIQUID
                    and day % client.income_frequency == 0
                )
                income_applied.append(bool(income_due))
                client.apply_income_if_due(day)

            client_margins = margin(portfolios_cm, scenarios, alpha=alpha)
            collaterals_cm = cm.collaterals_tensor()
            shortfall = client_margins - collaterals_cm
            needs_call = shortfall > 0

            default_shortfall = 0.0
            margin_call_records = [
                {"called": False, "amount": 0.0, "accepted": None, "liquidated": False}
                for _ in range(len(cm.clients))
            ]
            for i_client, client in enumerate(cm.clients):
                if not needs_call[i_client]:
                    continue
                M = float(shortfall[i_client].item())
                accepted = client.margin_called(M)
                record = {
                    "called": True,
                    "amount": M,
                    "accepted": bool(accepted),
                    "liquidated": False,
                }
                if not accepted:
                    if liquidate_on_default:
                        client.liquidate_portfolio()
                        client.set_collateral(0.0)
                        record["liquidated"] = True
                    if cm_absorbs_shortfall:
                        default_shortfall += M
                margin_call_records[i_client] = record

            cm_pnl = float(real_pnl_clients.sum().item())
            cm.cm_funds = max(0.0, cm.cm_funds + cm_pnl - default_shortfall)

            collaterals_cm = cm.collaterals_tensor()
            total_coll = float(collaterals_cm.sum().item())
            avg_coll = float(collaterals_cm.mean().item())
            min_coll = float(collaterals_cm.min().item())
            num_zero_coll = int((collaterals_cm <= 1e-6).sum().item())

            avg_margin = float(client_margins.mean().item())
            total_margin = float(client_margins.sum().item())

            cms_metrics[cm_name] = {
                "cm_funds": cm.cm_funds,
                "total_client_collateral": total_coll,
                "avg_client_collateral": avg_coll,
                "min_client_collateral": min_coll,
                "avg_client_margin": avg_margin,
                "total_client_margin": total_margin,
                "num_active_trades": num_active_trades,
                "num_accepted_trades": num_accepted_trades,
                "num_zero_collateral_clients": num_zero_coll,
            }
            if include_details:
                client_details = []
                for i_client, client in enumerate(cm.clients):
                    trade_list = []
                    trade_idx = (trades[i_client] != 0.0).nonzero(as_tuple=False).flatten()
                    for idx_trade in trade_idx.tolist():
                        trade_list.append(
                            {
                                "instrument": int(idx_trade),
                                "amount": float(trades[i_client, idx_trade].item()),
                                "accepted": bool(accepted_mask[i_client, idx_trade].item()),
                            }
                        )

                    entry = {
                        "client_id": client.client_id,
                        "vip_status": client.vip_status.name,
                        "liquidity_status_end": client.liquidity_status.name,
                        "wealth_end": float(client.wealth),
                        "collateral_end": float(client.collateral),
                        "recklessness_end": float(client.recklessness),
                        "pnl": float(real_pnl_clients[i_client].item()),
                        "income_applied": income_applied[i_client],
                        "margin": float(client_margins[i_client].item()),
                        "shortfall": float(shortfall[i_client].item()),
                        "margin_call": margin_call_records[i_client],
                        "trades": trade_list,
                    }
                    if client_start:
                        entry["start"] = client_start[i_client]
                        entry["recklessness_start"] = recklessness_before[i_client]
                        entry["liquidity_status_start"] = client_start[i_client]["liquidity_status"]
                        if include_portfolios:
                            entry["portfolio_start"] = client_start[i_client]["portfolio"]
                    if include_portfolios:
                        entry["portfolio_end"] = client.portfolio.clone()
                    client_details.append(entry)

                cms_metrics[cm_name]["default_shortfall"] = default_shortfall
                cms_metrics[cm_name]["cm_pnl"] = cm_pnl
                cms_metrics[cm_name]["clients"] = client_details
                if include_portfolios:
                    cms_metrics[cm_name]["cm_portfolio"] = cm.aggregate_portfolio().clone()

            system_total_client_collateral += total_coll
            system_total_client_margin += total_margin
            system_total_cm_funds += cm.cm_funds
            system_total_active_trades += num_active_trades
            system_total_accepted_trades += num_accepted_trades
            system_total_zero_collateral_clients += num_zero_coll

        ccps_metrics: Dict[str, Any] = {}
        for ccp_name, ccp in ccps.items():
            snapshot = ccp.snapshot_state(scenarios, alpha=alpha)
            ccps_metrics[ccp_name] = {
                "ccp_margin": snapshot["ccp_margin"],
                "cm_margins": snapshot["cm_margins"],
                "cm_funds": snapshot["cm_funds"],
                "cm_shortfalls": snapshot["cm_shortfalls"],
            }
            if include_details and include_portfolios:
                ccps_metrics[ccp_name]["net_portfolio"] = snapshot["net_portfolio"]

    num_clients_total = sum(len(cm.clients) for cm in clearing_members.values())
    avg_client_collateral = (
        system_total_client_collateral / num_clients_total if num_clients_total > 0 else 0.0
    )
    avg_client_margin = (
        system_total_client_margin / num_clients_total if num_clients_total > 0 else 0.0
    )

    system_metrics = {
        "total_client_collateral": system_total_client_collateral,
        "avg_client_collateral": avg_client_collateral,
        "min_client_collateral": min(
            m["min_client_collateral"] for m in cms_metrics.values()
        ) if cms_metrics else 0.0,
        "total_cm_funds": system_total_cm_funds,
        "avg_client_margin": avg_client_margin,
        "total_client_margin": system_total_client_margin,
        "num_active_trades": system_total_active_trades,
        "num_accepted_trades": system_total_accepted_trades,
        "num_zero_collateral_clients": system_total_zero_collateral_clients,
    }

    payload = {
        "day": day,
        "market_index": idx,
        "system": system_metrics,
        "cms": cms_metrics,
        "ccps": ccps_metrics,
    }
    if include_details:
        payload["details"] = details
    return payload


def simulate_days(
    *,
    n_days: int,
    clearing_members: Dict[str, ClearingMember],
    ccps: Dict[str, CCP],
    scenario_gen,
    state_oh: torch.Tensor,
    returns_next: torch.Tensor,
    alpha: float = 0.99,
    scenarios_per_day: int = 1000,
    burn_in: int = 500,
    thin: int = 10,
    lambda_client: float = 10.0,
    lambda_cm: float = 10.0,
    trade_value_scale: float = 1.0,
    min_trade_abs: float = 0.0,
    max_trades_per_client: Optional[int] = None,
    state_index_strategy: str = "random",
    liquidate_on_default: bool = True,
    cm_absorbs_shortfall: bool = True,
    include_details: bool = False,
    include_portfolios: bool = True,
    include_scenarios: bool = False,
    include_returns: bool = False,
) -> list[Dict[str, Any]]:
    metrics = []
    for day in range(n_days):
        metrics.append(
            simulate_one_day(
                day=day,
                clearing_members=clearing_members,
                ccps=ccps,
                scenario_gen=scenario_gen,
                state_oh=state_oh,
                returns_next=returns_next,
                alpha=alpha,
                scenarios_per_day=scenarios_per_day,
                burn_in=burn_in,
                thin=thin,
                lambda_client=lambda_client,
                lambda_cm=lambda_cm,
                trade_value_scale=trade_value_scale,
                min_trade_abs=min_trade_abs,
                max_trades_per_client=max_trades_per_client,
                state_index_strategy=state_index_strategy,
                liquidate_on_default=liquidate_on_default,
                cm_absorbs_shortfall=cm_absorbs_shortfall,
                include_details=include_details,
                include_portfolios=include_portfolios,
                include_scenarios=include_scenarios,
                include_returns=include_returns,
            )
        )
    return metrics
