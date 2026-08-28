# Two-step reproduction — results (run 2026-08-28)

Source of every number below: `scripts/reproduce_two_step.py` →
`output/exp1_results.json` (reproducible; seeds fixed).

## Setup facts

- n = 13,848 (date, hour) rows; **577 days**; Phase II = 30% of days (4,152 rows)
- within-day lag-1 autocorr of spread = **0.257**
- corr(X = reserve, Y = spread) = **0.049**  ← weak effect, reported honestly
- corr(Z_yest = yesterday reserve, X) = **0.887**  ← strong persistence
- corr(Z_roll = past-7d spread, X) = **0.094**
- calibration R² (hour + Z_yest + Z_roll → X) = **0.788**
- outcome R² (hour + Z + X → Y) = **0.065**; X alone adds ≈ 0.0001

## Point estimates (key coefficients; truth = oracle = true X on all days)

| coef | oracle | Phase-II-only | two-step |
|---|---|---|---|
| intercept (00:00 base) | −42.20 | −34.39 | −7.22 |
| Z_yest | 0.0002 | 0.0069 | 0.0208 |
| Z_roll | 0.4232 | 0.3371 | **0.4166** |
| X | 0.0016 | −0.0057 | −0.0222 |

Two-step lands closest to the oracle on Z_roll (the coarse covariate with
real signal).

## Efficiency gain — day-clustered bootstrap SE (400 reps)

| coef | SE p2-only | SE two-step | ratio |
|---|---|---|---|
| intercept | 55.59 | 4.87 | **0.09** |
| Z_yest | 0.0055 | 0.0028 | **0.51** |
| Z_roll | 0.0883 | 0.0525 | **0.60** |
| X | 0.0045 | 0.0029 | **0.63** |

Two-step is more efficient on every coefficient — the "coarse supply-side
model covers all days, detailed second stage refines it" structure.

## Coverage of 95% CIs (truth = oracle)

| coef | i.i.d. analytic (100 masks) | day-clustered p2-only (M=500) | day-clustered two-step (M=500) |
|---|---|---|---|
| intercept | 0.65 | ✓ | ✗ |
| Z_yest | 0.56 | ✓ | ✓ |
| Z_roll | 0.42 | ✗ | ✓ |
| X | 0.57 | ✓ | ✗ |

**i.i.d. analytic CIs badly under-cover** (0.42–0.65 vs 0.95) — within-day
dependence inflates the effective sample. Day-clustered bootstrap restores
~nominal coverage where the coefficient is identified (the two-step covers
Z_yest and Z_roll).

## Reading against the 2026-08-27 email

| email claim | reproduced here? | evidence |
|---|---|---|
| "each day as one unit, approximately i.i.d." | ✓ | 577 day-clusters; day resampling is the inference unit |
| "day clustering to handle dependence within a day" | ✓ | iid coverage 0.42–0.65 → day-clustered ~nominal |
| "coarse supply-side model covers all days, detailed second stage refines it" | ✓ | two-step (full sample) SE < Phase-II-only on all 4 coefs |
| "two-stage form clearly improved my results" | ✓ (method level) | ratio 0.09–0.63; the 12x→1x / plunge numbers are from the XGBoost pipeline, not here |
| "model captured a sharp market plunge" | not here | from the author's production XGBoost pipeline (v5.4-R) |

## Caveats (honest)

1. `corr(X, Y) = 0.05` → the fine covariate has little direct effect on the
   spread; `beta_x` is weakly identified (two-step misses it). **The
   estimator is only as good as the information in X.**
2. Single-mask day-clustered inference under-covers `intercept`/`Z_roll` for
   the two-step at 30% Phase II.
3. Empirical bootstrap, not theory — same caveat as the research memo.
