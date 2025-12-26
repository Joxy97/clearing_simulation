import sys
from pathlib import Path

# Add project root to path for direct script execution
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
from typing import List
import pandas as pd

from src.clearing_simulation.encoding import encode_returns, encode_states
from src.clearing_simulation.data_utils import prices_to_returns, prices_to_states
from boltzmann_machine.utils.create_prices import create_prices

K_v = 4
K_c = 4
percentiles=[0.00, 0.01, 0.05, 0.10, 0.25, 0.40, 0.60]

def bm_to_csv(
    tensor: torch.Tensor,
    K_v: int,
    K_c: int,
    percentiles: List[float],
    path: str = "data.csv",
):
    """
    Export BM-visible tensor to CSV with meaningful column names.

    Parameters
    ----------
    tensor : torch.Tensor
        Shape (T, K_v + K_c + N*7)
    K_v : int
        Number of volatility bins
    K_c : int
        Number of co-move bins
    percentiles : List[float]
        Loss quantiles used for encoding (length 7)
        Example: [0.00,0.01,0.05,0.10,0.25,0.40,0.60]
    path : str
        Output CSV path
    """

    if tensor.ndim != 2:
        raise ValueError("tensor must be 2D (T, D)")

    if len(percentiles) != 7:
        raise ValueError("percentiles must have length 7 (1 gain + 6 loss bins)")

    T, D = tensor.shape
    K_state = K_v + K_c

    if (D - K_state) % 7 != 0:
        raise ValueError("Remaining columns after state are not divisible by 7")

    N = (D - K_state) // 7

    # -----------------------------
    # Build column names
    # -----------------------------

    cols = []

    # Market state
    cols += [f"V_bin_{k}" for k in range(K_v)]
    cols += [f"C_bin_{k}" for k in range(K_c)]

    # Returns
    for i in range(N):
        inst = f"inst_{i+1}"
        cols.append(f"{inst}_G")  # gain / non-loss

        # Loss bins L1..L6 (L6 worst)
        for k in range(1, 7):
            q_lo = percentiles[6 - k]
            q_hi = percentiles[7 - k] if (7 - k) < len(percentiles) else 1.0
            cols.append(f"{inst}_L{k}")

    if len(cols) != D:
        raise RuntimeError("Column count mismatch")

    # -----------------------------
    # Export
    # -----------------------------
    df = pd.DataFrame(
        tensor.detach().cpu().numpy().astype(int),
        columns=cols,
    )

    df.to_csv(path, index=False)
    print(f"BM dataset exported to {path} | shape={df.shape}")

if __name__ == "__main__":

    prices_torch = create_prices()

    raw_returns = prices_to_returns(prices_torch)
    raw_states  = prices_to_states(prices_torch)               # (T-1,2)
    y_next_t    = raw_returns[1:]                              # (T-1,N) == returns[t+1]

    # Fit encoders on train slice
    T1 = raw_states.shape[0]
    optional_slice = slice(0, int(1*T1))

    state_oh, state_params = encode_states(raw_states,
                                           K_v=K_v,
                                           K_c=K_c,
                                           fit=True,
                                           train_slice=optional_slice)

    ret_oh, ret_params = encode_returns(y_next_t,
                                        percentiles=percentiles,
                                        fit=True,
                                        train_slice=optional_slice)

    bm_vis = torch.cat([state_oh, ret_oh], dim=1)  # (T-1, K_v+K_c + N*7)

    bm_to_csv(
        tensor=bm_vis,
        K_v=K_v,
        K_c=K_c,
        percentiles=percentiles,
        path="boltzmann_machine/data/data.csv",
    )