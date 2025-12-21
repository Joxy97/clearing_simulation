from __future__ import annotations

import os
import torch

from boltzmann import RBM, load_config


def load_rbm(run_folder: str, device: torch.device) -> RBM:
    config = load_config(os.path.join(run_folder, "config.py"))
    checkpoint = torch.load(os.path.join(run_folder, "model.pt"), map_location=device)
    model = RBM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model
