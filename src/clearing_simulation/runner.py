from __future__ import annotations

import json
import os
from typing import Any, Dict

import torch

from .config import DEFAULT_CONFIG
from .dataset import download_sp500_dataset, load_sp500_open_prices, prepare_dataset
from .device import get_device
from .model_loader import load_rbm
from .scenario import ScenarioGenerator
from .simulation import build_clearing_members, build_ccps, make_margin_func, simulate_days
from .utils import seed_everything


def _serialize(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def run_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    seed_everything(config.get("seed"))

    model_cfg = config.get("model", DEFAULT_CONFIG["model"])
    data_cfg = config.get("data", DEFAULT_CONFIG["data"])
    sim_cfg = config.get("simulation", DEFAULT_CONFIG["simulation"])
    output_cfg = config.get("output", DEFAULT_CONFIG["output"])

    device = get_device(model_cfg.get("device", "auto"))

    if data_cfg.get("source") != "sp500":
        raise ValueError("Only 'sp500' data source is supported.")

    data_dir = data_cfg.get("data_dir")
    if data_dir is None:
        data_dir = download_sp500_dataset()

    prices = load_sp500_open_prices(
        data_dir,
        limit_instruments=data_cfg.get("limit_instruments"),
    )

    dataset = prepare_dataset(
        prices,
        K_v=int(data_cfg.get("K_v", 4)),
        K_c=int(data_cfg.get("K_c", 4)),
        loss_percentiles=data_cfg.get("loss_percentiles"),
        train_ratio=float(data_cfg.get("train_ratio", 0.8)),
    )

    model = load_rbm(model_cfg.get("run_folder", "models/run_1"), device)
    scenario_gen = ScenarioGenerator(
        model=model,
        ret_params=dataset.ret_params,
        n_instruments=dataset.returns_next.shape[1],
    )

    idx0 = int(torch.randint(0, dataset.state_oh.size(0), (1,)).item())
    init_state = dataset.state_oh[idx0].to(device)
    init_scenarios = scenario_gen.sample(
        state_onehot=init_state,
        n_samples=int(sim_cfg.get("init_scenarios", 500)),
        burn_in=int(sim_cfg.get("burn_in", 500)),
        thin=int(sim_cfg.get("thin", 10)),
    )

    margin_func = make_margin_func(init_scenarios, alpha=float(sim_cfg.get("alpha", 0.99)))
    clearing_members = build_clearing_members(
        config.get("clearing_members", DEFAULT_CONFIG["clearing_members"]),
        n_instruments=dataset.returns_next.shape[1],
        margin_func=margin_func,
        device=device,
    )
    ccps = build_ccps(
        config.get("ccps", DEFAULT_CONFIG["ccps"]),
        clearing_members=clearing_members,
        n_instruments=dataset.returns_next.shape[1],
        device=device,
    )

    max_trades_per_client = sim_cfg.get("max_trades_per_client")
    if max_trades_per_client is not None:
        max_trades_per_client = int(max_trades_per_client)

    include_details = bool(output_cfg.get("include_details", False))
    include_portfolios = bool(output_cfg.get("include_portfolios", True))
    include_scenarios = bool(output_cfg.get("include_scenarios", False))
    include_returns = bool(output_cfg.get("include_returns", False))

    metrics = simulate_days(
        n_days=int(sim_cfg.get("n_days", 10)),
        clearing_members=clearing_members,
        ccps=ccps,
        scenario_gen=scenario_gen,
        state_oh=dataset.state_oh,
        returns_next=dataset.returns_next,
        alpha=float(sim_cfg.get("alpha", 0.99)),
        scenarios_per_day=int(sim_cfg.get("scenarios_per_day", 1000)),
        burn_in=int(sim_cfg.get("burn_in", 500)),
        thin=int(sim_cfg.get("thin", 10)),
        lambda_client=float(sim_cfg.get("lambda_client", 10.0)),
        lambda_cm=float(sim_cfg.get("lambda_cm", 10.0)),
        trade_value_scale=float(sim_cfg.get("trade_value_scale", 1.0)),
        min_trade_abs=float(sim_cfg.get("min_trade_abs", 0.0)),
        max_trades_per_client=max_trades_per_client,
        state_index_strategy=sim_cfg.get("state_index_strategy", "random"),
        liquidate_on_default=bool(sim_cfg.get("liquidate_on_default", True)),
        cm_absorbs_shortfall=bool(sim_cfg.get("cm_absorbs_shortfall", True)),
        allow_trade_topups=bool(sim_cfg.get("allow_trade_topups", False)),
        include_details=include_details,
        include_portfolios=include_portfolios,
        include_scenarios=include_scenarios,
        include_returns=include_returns,
    )

    summary = {
        "days": len(metrics),
        "final_system": metrics[-1]["system"] if metrics else {},
    }

    output_path = output_cfg.get("path") if isinstance(output_cfg, dict) else None
    payload = {"metrics": metrics, "summary": summary}

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(_serialize(payload), f, indent=2)

    return payload
