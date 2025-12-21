from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .encoding import decode_returns, build_return_midpoints, ReturnEncoderParams


def sample_scenarios(
    model,
    state_onehot: torch.Tensor,
    n_samples: int,
    ret_params: ReturnEncoderParams,
    burn_in: int = 500,
    thin: int = 10,
    return_representative: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = model.W.device
    state_dim = int(state_onehot.numel())
    if state_dim <= 0:
        raise ValueError("state_onehot must be non-empty")
    if model.nv < state_dim:
        raise ValueError("model.nv is smaller than state dimension")

    clamp_idx = list(range(state_dim))
    v_clamp = torch.zeros(model.nv, device=device, dtype=model.W.dtype)
    v_clamp[clamp_idx] = state_onehot.to(device=device, dtype=model.W.dtype)

    samples = model.sample_clamped(
        v_clamp=v_clamp,
        clamp_idx=clamp_idx,
        n_samples=n_samples,
        burn_in=burn_in,
        thin=thin,
    )

    returns_onehot = samples[:, state_dim:]
    expected_cols = ret_params.N * 7
    if returns_onehot.shape[1] < expected_cols:
        raise ValueError("RBM sample has fewer return bits than expected")
    returns_onehot = returns_onehot[:, :expected_cols]

    scenarios = decode_returns(
        returns_onehot,
        ret_params,
        return_representative=return_representative,
        return_midpoints=return_representative is None,
    )
    return scenarios


@dataclass
class ScenarioGenerator:
    model: Any
    ret_params: ReturnEncoderParams
    n_instruments: int
    return_representative: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.return_representative is None:
            self.return_representative = build_return_midpoints(
                self.ret_params,
                device=self.model.W.device,
            )

    def sample(
        self,
        state_onehot: torch.Tensor,
        n_samples: int,
        burn_in: int = 500,
        thin: int = 10,
    ) -> torch.Tensor:
        scenarios = sample_scenarios(
            model=self.model,
            state_onehot=state_onehot,
            n_samples=n_samples,
            ret_params=self.ret_params,
            burn_in=burn_in,
            thin=thin,
            return_representative=self.return_representative,
        )
        return scenarios[:, :self.n_instruments]
