# Clearing Simulation Review

## Findings
- **Critical** — Positive returns decode to 0, so short positions never lose and hedges are misvalued; ES and QUBO decisions are biased if shorts are allowed. `src/clearing_simulation/encoding.py:144` `src/clearing_simulation/encoding.py:189` `src/clearing_simulation/encoding.py:221` `src/clearing_simulation/scenario.py:62`
- **Medium** — Returns are raw price differences, not normalized returns; higher-price instruments dominate risk and margin, creating scale bias. `src/clearing_simulation/data_utils.py:6` `src/clearing_simulation/data_utils.py:14`

## Suggestions
- Add a positive-return representative (e.g., median positive return) or enforce long-only positions; for short-allowed models, positive returns cannot be collapsed to zero without understating short risk.
- Normalize returns (pct/log or z-scored per instrument) to remove price-level bias.