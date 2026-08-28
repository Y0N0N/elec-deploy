# Two-step reproduction (two-phase estimator on Guangdong DA-RT spread)

Reproduces, with a fresh, honest setup, the method claims of two-phase
/two-step (regression-calibration) estimation:

> "I use each day as one unit. This makes the observations approximately
> i.i.d., and I use day clustering to handle the dependence within a day."
> "The problem turns out to have a natural two-phase structure. A coarse
> supply-side model covers all days, and a detailed second stage refines it."

Run:

```bash
python3 scripts/reproduce_two_step.py
```

Outputs `output/exp1_results.json` (all numbers) and
`output/summary_table.txt` (human-readable).

## Data

Copied (read-only subset) from a provincial grid disclosure dataset:

| file | what | unit |
|---|---|---|
| `data/spread_label.feather` | DA−RT price spread, 24 hourly cols | 元/MWh |
| `data/正备用(MW).feather` | positive reserve, 15-min (96 cols) → hourly | MW |

`spread_label.feather` already has 24 columns; the reserve file has 96
15-min columns that are averaged to the hour (`read_hourly`).

## Design (non-degenerate, faithful to the two-phase-paper idea)

| symbol | role | definition | known when |
|---|---|---|---|
| `Y` | outcome | DA−RT spread (day `d`, hour `h`) | settlement |
| `Z_yest` | coarse | reserve, **yesterday** same hour | day-ahead (all days) |
| `Z_roll` | coarse | past-7-day rolling mean of spread | day-ahead (all days) |
| `X` | fine | reserve, **today** same hour | settlement only → Phase II |
| Phase II | ~30% of **days** | 24 hours move together | mimics partial disclosure |

`Z` predicts `X` well (`calibration R² = 0.79` — reserve is strongly
persistent day to day), so regression calibration should be efficient.
`X` has only a weak effect on the spread (`corr(X,Y)=0.05`), which we
report honestly.

## Estimators

- **Phase-II-only**: OLS of `Y ~ (hour, Z_yest, Z_roll, X)` on Phase II days.
- **Two-step (regression calibration)**: `E[X|Z]` estimated on Phase II,
  `X_hat` imputed for *all* days, OLS of `Y ~ (hour, Z_yest, Z_roll, X_hat)`
  on all days.

## Inference

- **i.i.d.** analytic sandwich SE (naive, ignores within-day dependence).
- **day-clustered bootstrap**: resample whole days with replacement; 95% CI
  by percentile.

## Results (all numbers in `output/exp1_results.json`)

### Efficiency gain — day-clustered bootstrap SE (ratio = SE_two_step / SE_phase2)

| coef | SE p2-only | SE two-step | ratio |
|---|---|---|---|
| intercept | 55.59 | 4.87 | **0.09** |
| Z_yest | 0.0055 | 0.0028 | **0.51** |
| Z_roll | 0.088 | 0.053 | **0.60** |
| X | 0.0045 | 0.0029 | **0.63** |

Two-step (full-sample imputation) beats Phase-II-only on every coefficient
for which the coarse covariates carry signal.

### Coverage of 95% CIs (truth = oracle = OLS with true X on all days)

| coef | i.i.d. analytic | day-clustered p2-only | day-clustered two-step |
|---|---|---|---|
| intercept | 0.65 | ✓ | ✗ |
| Z_yest | 0.56 | ✓ | ✓ |
| Z_roll | 0.42 | ✗ | ✓ |
| X | 0.57 | ✓ | ✗ |

The i.i.d. analytic CI **under-covers** (0.42–0.65 vs 0.95) because
within-day dependence (`lag-1 autocorr ≈ 0.26`) inflates the effective
sample size — the time-series vs i.i.d. point in the two-step framework.
Day-clustered inference restores ~nominal coverage where the coefficient is
identified.

## Honest caveats

- `corr(X, Y) = 0.05`: the fine covariate `X` (reserve) has little direct
  effect on the spread. The efficiency gain is on the *coarse* covariates
  (two-step sees all days), and `beta_x` itself is not reliably identified.
  **The estimator is only as good as the information in X** — the same caveat
  as in the two-step research memo.
- A single realized Phase-II mask under-covers `intercept` and `Z_roll` for
  the two-step: with a ~30% subset the coarse load level (baseline) is not
  fully pinned by the imputed data.
- All inference is empirical (bootstrap), not theory — matching the method's
  "empirical observations, not theory" stance.

## What this does NOT reproduce

Two concrete production results — **train/eval gap 12x → 1x** and the
**capture of a sharp market plunge** — come from the author's production
XGBoost pipeline (an internal project log, v5.3 / v5.4-R), not from this
estimator. This script reproduces the *method framing* (day = unit,
two-phase structure, day-clustered inference) on the same data.
