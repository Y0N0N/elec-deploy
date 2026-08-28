# v10 — two-phase (two-step) prediction of the Guangdong DA-RT spread

v10 puts the two-phase / two-step structure we reproduced in
`reproduce_two_step.py` into a **real electricity-price forecasting** model,
on top of the production v9 framework.

## The setup (what v10 adds over v9)

v9 = XGBoost dual-head (direction classifier + magnitude regressor) +
hour prior + C-strategy rule layer + walk-forward backtest, trained on 404
factors. **v10 keeps all of that** and adds an explicit two-phase structure:

```
Phase I   (all days):      coarse, day-ahead features only (673 clean factors
                           + hour one-hot + regime)
Phase II  (~30% of days):  ALSO observe a fine covariate X (settlement-time
                           information, available only on a subset of days)
```

The fine covariate is handled three ways (same train/valid windows, so the
only difference is how X is used):

| mode | how X is handled | meaning |
|---|---|---|
| `all` | X observed on every day | uses the fine info perfectly (reference) |
| `phase2_only` | X only on Phase-II days, NaN elsewhere | v9-style partial information |
| `twostep` | regression calibration `E[X\|Z]` on Phase II, imputed to **all** days | the two-step estimator |

## The fine covariate — and why it matters

Two candidates were tested (honest A/B):

1. **`h_正备用_zscore_v10`** — real positive reserve (MW), 7-day rolling
   zscore. Built from the raw 15-min disclosure data.
   - **v9's own factor library ships a BROKEN positive-reserve factor**
     (`h_正备用_zscore` in qlib158 is all zeros — verified in the data audit),
     so v9 never effectively used it. v10 builds the factor cleanly.
   - Result: **no signal** — all three modes give identical P&L (+3576).
     corr(reserve, spread) = 0.05. The estimator is only as good as the
     information in X.

2. **`h_价差_前1小时_v10`** — same-day previous-hour spread (a nowcast X,
   known only after the hour settles). corr(X, Y) = **0.74**.
   - Strong signal, but a *partial-information* covariate: at prediction time
     only ~30% of days have it.

## Results (walk-forward, valid = last 60 days, unseen)

### Fine covariate = intraday lag-1 spread (the real signal)

| mode | n_trigger | dir_hit | net_win | **P&L** | MDD |
|---|---|---|---|---|---|
| `all` (X everywhere) | 213 | 0.637 | 0.637 | +8780 | 2219 |
| `phase2_only` (X 30% days) | 112 | 0.589 | 0.589 | **+2319** | 963 |
| **`twostep`** (calibrate + impute) | 780 | 0.597 | 0.597 | **+10854** | 2855 |

**Two-step P&L is 4.7× the Phase-II-only strategy (+10854 vs +2319)** — the
imputed fine covariate lets the model act on ~7× more hours (780 vs 112
triggers) at the same or better hit rate. This is the two-phase paper's
efficiency idea in a real trading P&L.

### Fine covariate = positive reserve (honest control)

All three modes identical (P&L +3576, trig 93) — the reserve carries no
incremental information for the spread. Kept as an honest control showing
the method only helps when the fine covariate itself has signal.

## Why this connects to Prof. Wong's two-phase method

- Day is the unit; Phase II = random subset of days; fine covariate only on
  Phase II → exactly the two-phase data structure.
- `twostep` = regression calibration (the same estimator reproduced in
  `reproduce_two_step.py`), which is more efficient than using only the
  Phase-II subset.
- On time-series: we cluster by day (whole days resampled), the same
  day-clustering argument from the 2026-08-27 email.

## Files

- `scripts/v10_two_step_predict.py` — training + 3-mode comparison + backtest
- `output/v10_results_lag_fine.json` — results, fine = intraday lag-1 spread
- `output/v10_results_reserve_fine.json` — results, fine = reserve (control)
- `data/factors/h_价差_前1小时_v10.fea`, `data/factors/h_正备用_zscore_v10.fea`
  — the two fine-covariate factors (generated automatically by the script)

## Run

```bash
python3 scripts/v10_two_step_predict.py --valid-days 60                # lag-spread fine (main result)
python3 scripts/v10_two_step_predict.py --valid-days 60 --fine h_正备用_zscore_v10   # reserve control
```

## Honest caveats

- Phase II mask is on train days only; the valid window (future) has no fine
  covariate at all, so valid is a realistic out-of-sample test.
- The `all` mode is a *different* strategy (uses true X at inference, which
  makes the magnitude head more conservative), not strictly an upper bound
  for `twostep`. The clean comparison is `twostep` vs `phase2_only`.
- Results are a single 60-day walk-forward window; variance across windows is
  not yet quantified.
