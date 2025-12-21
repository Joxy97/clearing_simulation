from __future__ import annotations

import copy
import json
from typing import Any, Dict, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": None,
    "model": {
        "run_folder": "models/run_1",
        "device": "auto",
    },
    "data": {
        "source": "sp500",
        "data_dir": None,
        "train_ratio": 0.8,
        "K_v": 4,
        "K_c": 4,
        "loss_percentiles": [0.0, 0.01, 0.05, 0.10, 0.25, 0.40, 0.60],
        "limit_instruments": None,
    },
    "simulation": {
        "n_days": 10,
        "scenarios_per_day": 500,
        "burn_in": 500,
        "thin": 10,
        "alpha": 0.99,
        "lambda_client": 10.0,
        "lambda_cm": 10.0,
        "trade_value_scale": 1.0,
        "min_trade_abs": 0.0,
        "max_trades_per_client": None,
        "state_index_strategy": "random",
        "init_scenarios": 500,
        "liquidate_on_default": True,
        "cm_absorbs_shortfall": True,
    },
    "clearing_members": {
        "CM_A": {"n_clients": 25, "cm_funds": 1_000_000.0},
        "CM_B": {"n_clients": 25, "cm_funds": 1_000_000.0},
    },
    "ccps": {
        "CCP_ALL": {"instrument_indices": "all", "default_fund": 10_000_000.0},
    },
    "output": {
        "path": None,
    },
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, val in updates.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], val)
        else:
            base[key] = val
    return base


def load_config(path: Optional[str]) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config
    with open(path, "r", encoding="utf-8-sig") as f:
        user_cfg = json.load(f)
    return _deep_update(config, user_cfg)
