from __future__ import annotations

import os
import pandas as pd
import kagglehub
import torch

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

def create_prices():
  # Download dataset
  path = kagglehub.dataset_download("camnugent/sandp500")

  # Path with individual stock CSVs
  data_dir = os.path.join(path, "individual_stocks_5yr/individual_stocks_5yr")

  # Collect all CSV files (sorted for reproducibility)
  csv_files = sorted(
      f for f in os.listdir(data_dir)
      if f.endswith(".csv")
  )

  open_series = []

  for i, fname in enumerate(csv_files, start=1):
      fpath = os.path.join(data_dir, fname)
      df = pd.read_csv(fpath, usecols=["open"])
      df = df.reset_index(drop=True)  # ensure clean alignment
      df.columns = [f"inst_{i}"]
      open_series.append(df)

  # Concatenate column-wise
  raw_prices = pd.concat(open_series, axis=1)
  raw_prices_torch = torch.tensor(raw_prices.to_numpy(), dtype=torch.float32)

  return raw_prices_torch