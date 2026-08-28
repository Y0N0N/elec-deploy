# Data — not included (privacy)

The scripts in this repo are self-contained **given** the raw data, but the
data files themselves are **not published** (they are provincial grid
disclosure data the author has access to, and are not public).

## Required input files

Place these under `data/` to run the scripts:

| file | format | used by |
|---|---|---|
| `spread_label.feather` | date × 24 hourly columns, DA−RT price spread (元/MWh) | both scripts |
| `正备用(MW).feather` | date × 96 fifteen-minute columns, positive reserve (MW) | `v10_two_step_predict.py` |
| `factors/*.fea` | date × 24 hourly columns, one factor per file (feather) | `v10_two_step_predict.py` |
| `日前统一结算价.feather` | date × 24, day-ahead settlement price | diagnostics only |
| `实时统一结算价.feather` | date × 24, real-time settlement price | diagnostics only |

`factors/` mirrors the qlib158 factor library layout (each `.fea` is a wide
date×24h frame). The two fine-covariate factors
(`h_正备用_zscore_v10`, `h_价差_前1小时_v10`) are **generated automatically**
by `scripts/v10_two_step_predict.py` from the raw 15-min `正备用(MW).feather`
and `spread_label.feather` — see `build_fine_factors()`.

## How the data is used

- `reproduce_two_step.py`: `spread_label.feather` + `正备用(MW).feather`
  (15-min → hourly) only. Fully reproducible with these two files.
- `v10_two_step_predict.py`: additionally needs the factor library
  `factors/*.fea`. If you only have a few factors, use `--subset N` to run a
  smoke test on the first N features.
