# Clearing Simulation Review

## Findings
- **Critical** — Positive returns decode to 0, so short positions never lose and hedges are misvalued; ES and QUBO decisions are biased if shorts are allowed. `src/clearing_simulation/encoding.py:144` `src/clearing_simulation/encoding.py:189` `src/clearing_simulation/encoding.py:221` `src/clearing_simulation/scenario.py:62`
- **High** — Margin penalties are equality-style quadratic terms; when collateral/funds exceed margin (A < 0), the solver is pushed to add risk to “use up” slack rather than simply stay within limits. `src/clearing_simulation/qubo.py:91` `src/clearing_simulation/qubo.py:94` `src/clearing_simulation/qubo.py:102` `src/clearing_simulation/qubo.py:104`
- **High** — Trade utility is proportional to signed delta, so negative deltas are always penalized even if they reduce risk or hedge exposure. `src/clearing_simulation/qubo.py:86` `src/clearing_simulation/qubo.py:89`
- **Medium** — `top_up_trust` is computed but not used in QUBO or margin constraints, so the intended “buffer of trust” has no effect on decisions. `src/clearing_simulation/client.py:90` `src/clearing_simulation/client.py:139`
- **Medium** — Returns are raw price differences, not normalized returns; higher-price instruments dominate risk and margin, creating scale bias. `src/clearing_simulation/data_utils.py:6` `src/clearing_simulation/data_utils.py:14`
- **Low** — Default config uses 10 scenarios/day with alpha 0.99, so ES is effectively the single worst scenario per day; sampling noise can swamp solver comparisons. `configs/default.json:18` `configs/default.json:21`

## Suggestions
- Add a positive-return representative (e.g., median positive return) or enforce long-only positions; for short-allowed models, positive returns cannot be collapsed to zero without understating short risk.
- Replace equality penalties with inequality-style constraints, e.g., penalize only when margin exceeds collateral/funds; introduce a slack variable if you want to keep it as a QUBO.
- Redefine trade utility as a sign-neutral “importance” score (volume, fee proxy, or risk-reduction proxy) and use that in the linear term instead of signed delta.
- Incorporate the trust buffer into QUBO inputs (effective collateral) so it actually affects acceptance decisions.
- Normalize returns (pct/log or z-scored per instrument) to remove price-level bias.

## Assumptions / Questions
- Are short positions intended to be risk-realistic? If yes, gains must be represented; if no, clamp portfolios/trades to nonnegative.
- Is “importance” meant to be throughput, fee income, or risk reduction? The utility term should encode that explicitly.

## Current State Summary
- End-to-end pipeline is in place (RBM scenarios → ES margins → QUBO trade selection → clearing/margin calls).
- Core logic is coherent for a toy model, but several modeling choices currently bias results in ways unrelated to solver quality.
