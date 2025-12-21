from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import kagglehub
import torch

from .data_utils import prices_to_returns, prices_to_states
from .encoding import encode_states, encode_returns, StateEncoderParams, ReturnEncoderParams


@dataclass
class PreparedDataset:
    prices: torch.Tensor
    returns: torch.Tensor
    raw_states: torch.Tensor
    state_oh: torch.Tensor
    state_params: StateEncoderParams
    returns_next: torch.Tensor
    ret_oh: torch.Tensor
    ret_params: ReturnEncoderParams

    def to(self, device: torch.device) -> "PreparedDataset":
        return PreparedDataset(
            prices=self.prices.to(device),
            returns=self.returns.to(device),
            raw_states=self.raw_states.to(device),
            state_oh=self.state_oh.to(device),
            state_params=self.state_params,
            returns_next=self.returns_next.to(device),
            ret_oh=self.ret_oh.to(device),
            ret_params=self.ret_params,
        )


def download_sp500_dataset() -> str:
    return kagglehub.dataset_download("camnugent/sandp500")


def _resolve_sp500_data_dir(base_dir: str) -> str:
    candidate = os.path.join(base_dir, "individual_stocks_5yr", "individual_stocks_5yr")
    if os.path.isdir(candidate):
        return candidate
    if os.path.isdir(base_dir):
        return base_dir
    raise FileNotFoundError(f"Could not find SP500 data under: {base_dir}")


def load_sp500_open_prices(data_dir: str, limit_instruments: Optional[int] = None) -> torch.Tensor:
    data_dir = _resolve_sp500_data_dir(data_dir)
    csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    if limit_instruments is not None:
        csv_files = csv_files[:limit_instruments]

    open_series = []
    for i, fname in enumerate(csv_files, start=1):
        fpath = os.path.join(data_dir, fname)
        df = pd.read_csv(fpath, usecols=["open"], dtype={"open": np.float32})
        df = df.reset_index(drop=True)
        df.columns = [f"inst_{i}"]
        open_series.append(df)

    raw_prices = pd.concat(open_series, axis=1)
    prices_np = raw_prices.to_numpy(dtype=np.float32, copy=False)
    return torch.from_numpy(prices_np)


def prepare_dataset(
    prices: torch.Tensor,
    *,
    K_v: int = 4,
    K_c: int = 4,
    loss_percentiles: Optional[list[float]] = None,
    train_ratio: float = 0.8,
) -> PreparedDataset:
    if loss_percentiles is None:
        loss_percentiles = [0.0, 0.01, 0.05, 0.10, 0.25, 0.40, 0.60]

    returns = prices_to_returns(prices)
    returns = torch.nan_to_num(returns, nan=0.0)
    raw_states = prices_to_states(prices)
    returns_next = returns[1:]

    T = raw_states.shape[0]
    train_end = max(1, int(T * train_ratio))
    train_slice = slice(0, train_end)

    state_oh, state_params = encode_states(
        raw_states,
        K_v=K_v,
        K_c=K_c,
        fit=True,
        train_slice=train_slice,
    )

    ret_oh, ret_params = encode_returns(
        returns_next,
        percentiles=loss_percentiles,
        fit=True,
        train_slice=train_slice,
    )

    return PreparedDataset(
        prices=prices,
        returns=returns,
        raw_states=raw_states,
        state_oh=state_oh,
        state_params=state_params,
        returns_next=returns_next,
        ret_oh=ret_oh,
        ret_params=ret_params,
    )
