Clearing Simulation
===================

This project is a modularized version of the original notebook export. It loads a
trained RBM, generates return scenarios conditional on market state, and runs a
multi-day clearing simulation with ES-based margins and QUBO trade selection.

Quick start
-----------
1) Ensure dependencies are available (see pyproject.toml).
2) Place a trained RBM run in `models/run_1` (config.py + model.pt).
3) Run the simulation:

   python -m clearing_simulation --config configs/default.json

Notes
-----
- By default the SP500 dataset is downloaded with kagglehub unless `data_dir` is set.
- The default config limits instruments to keep QUBO sizes reasonable.
- Metrics are written to `outputs/metrics.json` unless overridden.
