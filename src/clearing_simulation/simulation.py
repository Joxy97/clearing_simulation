from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .client import Client
from .clearing_member import ClearingMember
from .ccp import CCP
from .risk import margin
from .qubo import compute_trade_margin_requirements


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
    client_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, ClearingMember]:
    def _parse_range(value: Any, label: str) -> Optional[tuple[int, int]]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            low = int(value[0])
            high = int(value[1])
            if low > high:
                raise ValueError(f"{label} must satisfy min <= max, got {value}.")
            return low, high
        raise ValueError(f"{label} must be a 2-item list or tuple, got {value}.")

    clearing_members: Dict[str, ClearingMember] = {}
    client_spec = client_spec or {}
    income_range_default = _parse_range(client_spec.get("income_range"), "clients.income_range")
    freq_range_default = _parse_range(
        client_spec.get("income_frequency_range"),
        "clients.income_frequency_range",
    )
    client_id = 0
    for cm_name, spec in cm_spec.items():
        n_clients = int(spec.get("n_clients", 0))
        cm_funds = float(spec.get("cm_funds", 0.0))
        cm_wealth = spec.get("cm_wealth")
        if cm_wealth is not None:
            cm_wealth = float(cm_wealth)
        income_range = _parse_range(
            spec.get("client_income_range", income_range_default),
            f"clearing_members.{cm_name}.client_income_range",
        )
        freq_range = _parse_range(
            spec.get("client_income_frequency_range", freq_range_default),
            f"clearing_members.{cm_name}.client_income_frequency_range",
        )
        cm_clients = []
        for _ in range(n_clients):
            income = None
            if income_range is not None:
                low, high = income_range
                income = int(torch.randint(low, high + 1, (1,)).item())
            income_frequency = None
            if freq_range is not None:
                low, high = freq_range
                income_frequency = int(torch.randint(low, high + 1, (1,)).item())
            c = Client(
                n_instruments=n_instruments,
                margin_func=margin_func,
                device=device,
                income=income,
                income_frequency=income_frequency,
                client_id=client_id,
            )
            cm_clients.append(c)
            client_id += 1
        clearing_members[cm_name] = ClearingMember(
            clients=cm_clients,
            cm_funds=cm_funds,
            cm_wealth=cm_wealth,
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


def _apply_default_waterfall(
    *,
    shortfall: float,
    cm_funds_available: float,
    cm_absorbs_shortfall: bool,
    ccps: Dict[str, CCP],
    ccp_default_fund_used: Dict[str, float],
) -> tuple[float, float, float]:
    if shortfall <= 0.0:
        return 0.0, 0.0, 0.0

    cm_available = max(0.0, float(cm_funds_available))
    cm_used = min(cm_available, shortfall) if cm_absorbs_shortfall else 0.0
    remaining = shortfall - cm_used

    ccp_used = 0.0
    if remaining > 0.0:
        for ccp_name in sorted(ccps):
            ccp = ccps[ccp_name]
            available = max(0.0, float(ccp.default_fund))
            if available <= 0.0:
                continue
            use = min(available, remaining)
            if use <= 0.0:
                continue
            ccp.default_fund = available - use
            remaining -= use
            ccp_used += use
            ccp_default_fund_used[ccp_name] = ccp_default_fund_used.get(ccp_name, 0.0) + use
            if remaining <= 0.0:
                break

    return cm_used, ccp_used, remaining


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
    allow_trade_topups: bool = False,
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
    system_total_default_shortfall = 0.0
    system_total_cm_funds_used = 0.0
    system_total_ccp_default_used = 0.0
    system_total_residual_shortfall = 0.0
    system_total_collateral_seized = 0.0
    system_total_ccp_cm_shortfall = 0.0
    system_total_ccp_cm_residual_shortfall = 0.0
    system_total_ccp_cm_collateral_seized = 0.0

    cms_metrics: Dict[str, Any] = {}
    details: Dict[str, Any] = {}
    if include_details:
        details["market_state"] = market_state
        if include_returns:
            details["real_returns"] = real_ret
        if include_scenarios:
            details["scenarios"] = scenarios

    ccp_default_fund_used: Dict[str, float] = {name: 0.0 for name in ccps}
    ccp_pre_metrics: Dict[str, Any] = {}

    with torch.no_grad():
        for ccp_name, ccp in ccps.items():
            margins_cm = ccp.compute_cm_margins(scenarios, alpha=alpha)
            funds_cm_pre = ccp.cm_funds_tensor()
            shortfalls_cm_pre = torch.clamp(margins_cm - funds_cm_pre, min=0.0)
            margin_ccp = ccp.compute_ccp_margin(scenarios, alpha=alpha)

            ccp_pre_metrics[ccp_name] = {
                "ccp_margin": margin_ccp,
                "cm_margins": margins_cm,
                "cm_funds": funds_cm_pre,
                "cm_shortfalls": shortfalls_cm_pre,
                "cm_margin_calls": [],
            }
            if include_details and include_portfolios:
                ccp_pre_metrics[ccp_name]["net_portfolio"] = ccp.ccp_net_portfolio()

        for ccp_name, ccp in ccps.items():
            pre = ccp_pre_metrics.get(ccp_name, {})
            shortfalls_cm_pre = pre.get("cm_shortfalls", torch.zeros(0))

            cm_margin_calls = []
            for idx_cm, cm in enumerate(ccp.clearing_members):
                amount = float(shortfalls_cm_pre[idx_cm].item()) if shortfalls_cm_pre.numel() > 0 else 0.0
                record = {
                    "cm_name": cm.name,
                    "called": False,
                    "amount": 0.0,
                    "accepted": None,
                    "liquidated": False,
                    "collateral_used": 0.0,
                    "default_fund_used": 0.0,
                    "residual_shortfall": 0.0,
                }
                if amount > 0:
                    record["called"] = True
                    record["amount"] = amount
                    system_total_ccp_cm_shortfall += amount
                    accepted = cm.margin_called(amount)
                    record["accepted"] = bool(accepted)
                    if not accepted:
                        if liquidate_on_default:
                            cm.liquidate_clients()
                            record["liquidated"] = True

                        collateral_used = max(0.0, float(cm.cm_funds))
                        record["collateral_used"] = collateral_used
                        system_total_ccp_cm_collateral_seized += collateral_used

                        cm.cm_funds = 0.0
                        cm.cm_wealth = 0.0

                        available = max(0.0, float(ccp.default_fund))
                        use = min(available, amount)
                        ccp.default_fund = available - use
                        record["default_fund_used"] = use
                        ccp_default_fund_used[ccp_name] = ccp_default_fund_used.get(ccp_name, 0.0) + use
                        system_total_ccp_default_used += use

                        residual = amount - use
                        record["residual_shortfall"] = residual
                        system_total_ccp_cm_residual_shortfall += residual
                cm_margin_calls.append(record)
            pre["cm_margin_calls"] = cm_margin_calls

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

            income_applied = []
            for client in cm.clients:
                income_due = (
                    client.liquidity_status == client.liquidity_status.LIQUID
                    and day % client.income_frequency == 0
                )
                income_applied.append(bool(income_due))
                client.apply_income_if_due(day)

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

            # Trade margin call logic: offer rejected trades a chance to be accepted
            # if client tops up collateral
            trade_margin_call_records = [
                [] for _ in range(len(cm.clients))
            ]
            if allow_trade_topups:
                # Find rejected trades (proposed but not accepted)
                proposed_mask = (trades != 0.0)
                rejected_mask = proposed_mask & ~accepted_mask

                if rejected_mask.any():
                    portfolios_cm = cm.portfolios_tensor()
                    collaterals_cm = cm.collaterals_tensor()

                    # Compute required top-up for each rejected trade
                    required_topups = compute_trade_margin_requirements(
                        portfolios=portfolios_cm,
                        trades=trades,
                        collaterals=collaterals_cm,
                        scenarios=scenarios,
                        rejected_mask=rejected_mask,
                        alpha=alpha,
                    )

                    # Offer trade margin calls to clients for each rejected trade
                    for i_client, client in enumerate(cm.clients):
                        rejected_instruments = rejected_mask[i_client].nonzero(as_tuple=False).flatten()
                        for i in rejected_instruments.tolist():
                            required = float(required_topups[i_client, i].item())
                            if required <= 0:
                                # No additional margin needed, accept the trade
                                accepted_mask[i_client, i] = True
                                num_accepted_trades += 1
                                trade_margin_call_records[i_client].append({
                                    "instrument": i,
                                    "amount": 0.0,
                                    "accepted": True,
                                    "trade_accepted": True,
                                })
                            else:
                                # Issue trade margin call
                                call_accepted = client.trade_margin_called(required)
                                if call_accepted:
                                    accepted_mask[i_client, i] = True
                                    num_accepted_trades += 1
                                trade_margin_call_records[i_client].append({
                                    "instrument": i,
                                    "amount": required,
                                    "accepted": call_accepted,
                                    "trade_accepted": call_accepted,
                                })

            cm.execute_trades(day, trades, accepted_mask)
            num_rejected_trades = max(0, num_active_trades - num_accepted_trades)
            trade_acceptance_rate = (
                num_accepted_trades / num_active_trades
                if num_active_trades > 0
                else 0.0
            )
            trade_acceptance_warning = (
                num_active_trades > 0 and num_accepted_trades == num_active_trades
            )

            portfolios_cm = cm.portfolios_tensor()
            real_pnl_clients = portfolios_cm @ real_ret

            for i_client, client in enumerate(cm.clients):
                pnl_value = float(real_pnl_clients[i_client].item())
                client.apply_variation_margin(pnl_value)

            client_margins = margin(portfolios_cm, scenarios, alpha=alpha)
            collaterals_cm = cm.collaterals_tensor()
            shortfall = client_margins - collaterals_cm
            needs_call = shortfall > 0

            total_default_shortfall = 0.0
            collateral_seized_total = 0.0
            # Market margin calls (due to portfolio risk from market moves)
            market_margin_call_records = [
                {
                    "called": False,
                    "amount": 0.0,
                    "accepted": None,
                    "liquidated": False,
                    "collateral_used": 0.0,
                }
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
                    "collateral_used": 0.0,
                }
                if not accepted:
                    if liquidate_on_default:
                        client.liquidate_portfolio()
                    record["liquidated"] = True
                    collateral_used = max(0.0, float(client.collateral))
                    collateral_seized_total += collateral_used
                    record["collateral_used"] = collateral_used
                    client.set_collateral(0.0)
                    total_default_shortfall += M
                market_margin_call_records[i_client] = record

            cm_pnl = float(real_pnl_clients.sum().item())
            cm_funds_available = cm.cm_funds + cm_pnl
            cm_used, ccp_used, residual_shortfall = _apply_default_waterfall(
                shortfall=total_default_shortfall,
                cm_funds_available=cm_funds_available,
                cm_absorbs_shortfall=cm_absorbs_shortfall,
                ccps=ccps,
                ccp_default_fund_used=ccp_default_fund_used,
            )
            cm.cm_funds = max(0.0, cm_funds_available - cm_used)

            collaterals_cm = cm.collaterals_tensor()
            total_coll = float(collaterals_cm.sum().item())
            avg_coll = float(collaterals_cm.mean().item())
            min_coll = float(collaterals_cm.min().item())
            num_zero_coll = int((collaterals_cm <= 1e-6).sum().item())

            avg_margin = float(client_margins.mean().item())
            total_margin = float(client_margins.sum().item())

            cms_metrics[cm_name] = {
                "cm_funds": cm.cm_funds,
                "cm_wealth": cm.cm_wealth,
                "cm_liquidity_status": cm.liquidity_status.name,
                "total_client_collateral": total_coll,
                "avg_client_collateral": avg_coll,
                "min_client_collateral": min_coll,
                "avg_client_margin": avg_margin,
                "total_client_margin": total_margin,
                "num_active_trades": num_active_trades,
                "num_accepted_trades": num_accepted_trades,
                "num_rejected_trades": num_rejected_trades,
                "trade_acceptance_rate": trade_acceptance_rate,
                "trade_acceptance_warning": trade_acceptance_warning,
                "num_zero_collateral_clients": num_zero_coll,
                "default_shortfall": total_default_shortfall,
                "cm_funds_used_for_default": cm_used,
                "ccp_default_fund_used": ccp_used,
                "residual_shortfall": residual_shortfall,
                "collateral_seized": collateral_seized_total,
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
                        "market_margin_call": market_margin_call_records[i_client],
                        "trade_margin_calls": trade_margin_call_records[i_client],
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
            system_total_default_shortfall += total_default_shortfall
            system_total_cm_funds_used += cm_used
            system_total_ccp_default_used += ccp_used
            system_total_residual_shortfall += residual_shortfall
            system_total_collateral_seized += collateral_seized_total
        ccps_metrics: Dict[str, Any] = {}
        for ccp_name, ccp in ccps.items():
            pre = ccp_pre_metrics.get(ccp_name, {})
            ccps_metrics[ccp_name] = {
                "ccp_margin": pre.get("ccp_margin", 0.0),
                "cm_margins": pre.get("cm_margins", torch.zeros(0)),
                "cm_funds": pre.get("cm_funds", torch.zeros(0)),
                "cm_shortfalls": pre.get("cm_shortfalls", torch.zeros(0)),
                "default_fund_remaining": float(ccp.default_fund),
                "default_fund_used": float(ccp_default_fund_used.get(ccp_name, 0.0)),
                "cm_margin_calls": pre.get("cm_margin_calls", []),
            }
            if include_details and include_portfolios and "net_portfolio" in pre:
                ccps_metrics[ccp_name]["net_portfolio"] = pre["net_portfolio"]

    num_clients_total = sum(len(cm.clients) for cm in clearing_members.values())
    avg_client_collateral = (
        system_total_client_collateral / num_clients_total if num_clients_total > 0 else 0.0
    )
    avg_client_margin = (
        system_total_client_margin / num_clients_total if num_clients_total > 0 else 0.0
    )
    system_total_rejected_trades = max(0, system_total_active_trades - system_total_accepted_trades)
    system_trade_acceptance_rate = (
        system_total_accepted_trades / system_total_active_trades
        if system_total_active_trades > 0
        else 0.0
    )
    system_trade_acceptance_warning = (
        system_total_active_trades > 0
        and system_total_accepted_trades == system_total_active_trades
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
        "num_rejected_trades": system_total_rejected_trades,
        "trade_acceptance_rate": system_trade_acceptance_rate,
        "trade_acceptance_warning": system_trade_acceptance_warning,
        "num_zero_collateral_clients": system_total_zero_collateral_clients,
        "total_default_shortfall": system_total_default_shortfall,
        "total_cm_funds_used_for_default": system_total_cm_funds_used,
        "total_ccp_default_fund_used": system_total_ccp_default_used,
        "total_residual_shortfall": system_total_residual_shortfall,
        "total_collateral_seized": system_total_collateral_seized,
        "total_ccp_cm_shortfall": system_total_ccp_cm_shortfall,
        "total_ccp_cm_residual_shortfall": system_total_ccp_cm_residual_shortfall,
        "total_ccp_cm_collateral_seized": system_total_ccp_cm_collateral_seized,
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
    allow_trade_topups: bool = False,
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
                allow_trade_topups=allow_trade_topups,
                include_details=include_details,
                include_portfolios=include_portfolios,
                include_scenarios=include_scenarios,
                include_returns=include_returns,
            )
        )
    return metrics
