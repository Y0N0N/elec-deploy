#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v10 — two-phase (two-step) prediction of the Guangdong DA-RT spread
===================================================================
Builds on v9's production framework (direction head + magnitude head +
hour prior + walk-forward backtest) and adds the two-phase / two-step
structure from Prof. Wong's two-phase-studies method that we reproduced
in reproduce_two_step.py.

The fine covariate is the REAL positive reserve (正备用), hourly.  v9's
factor library (qlib158) actually ships a BROKEN positive-reserve factor
(h_正备用_zscore is all zeros — see data audit), so v9 never effectively
used it.  v10 builds the factor from the raw 15-min disclosure data and
uses it inside an explicit two-phase structure:

  Phase I   (all days):  coarse, day-ahead features only (673 clean
                         factors + hour one-hot + regime)
  Phase II  (~30% days): also observes the fine covariate X = reserve
                         zscore (settlement-day information)

Three treatment modes (the experiment):

  all         X is observed on EVERY day (uses the fine info perfectly;
              upper-bound reference — not realistic, but oracle)
  phase2_only X is observed only on Phase-II days; NaN on Phase-I days
              (XGBoost handles missing; = v9-style partial-information)
  twostep     regression calibration: E[X | Z] estimated on Phase II,
              X_hat imputed for ALL days, then trained as a feature
              (= the two-step estimator, faithful to Prof. Wong)

Both heads use v9's exact hyper-parameters and thresholds (τ=50 minor,
τ=100 big), the same hour prior, the same C-strategy rule layer, and the
same walk-forward split.  We compare the three modes on the SAME
train/valid windows so the only difference is how the fine covariate is
handled.

Usage:
    python3 scripts/v10_two_step_predict.py [--valid-days 60]
        [--subset N]     # N=use first N feature files (smoke test)
        [--frac 0.30]    # Phase II fraction of days
        [--seed 42]
    Outputs: output/v10_results.json + prints

Data: all under data/ — factors/*.fea (684 incl. our reserve factor),
spread_label.feather (label).  Phase-II mask is BY DAY (24h move together).
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FACTOR_DIR = os.path.join(DATA, "factors")
OUT = os.path.join(HERE, "..", "output")
os.makedirs(OUT, exist_ok=True)

import numpy as np
import pandas as pd
import xgboost as xgb

# ── thresholds & hyper-params inherited from v9 cfg ──
T1 = 50.0          # SPREAD_THRESHOLD
T2 = 100.0         # SPREAD_THRESHOLD_BIG
N_EST = 200
ES = 30
RANDOM_STATE = 42
FIXED = dict(max_depth=8, learning_rate=0.05, subsample=0.8,
             colsample_bytree=1.0, tree_method="hist", n_jobs=8)

FINE_RESERVE = "h_正备用_zscore_v10"   # reserve factor (weak signal — honest control)
FINE_LAG = "h_价差_前1小时_v10"          # intraday lag-1 spread (strong nowcast signal)
FINE_DEFAULT = FINE_LAG

FINE = FINE_DEFAULT   # active fine covariate (set by --fine in main())


def build_fine_factors():
    """Build the fine-covariate factors from raw data (wide, date x 24h).

    - h_正备用_zscore_v10 : real positive reserve, 7-day rolling zscore
                            (the factor v9 ships broken; ours is clean).
    - h_价差_前1小时_v10  : same-day previous-hour spread (nowcast X,
                            only known after the hour is settled). NaN on the
                            first hour of each day (no same-day prev).
    """
    # reserve zscore
    rs = pd.read_feather(os.path.join(DATA, "正备用(MW).feather"))
    rs.index = pd.to_datetime(rs.index)
    rs.columns = [str(c) for c in rs.columns]
    s = rs.stack().astype(float).reset_index()
    s.columns = ["date", "time", "x"]
    s["hour"] = s["time"].astype(str).str[:2]
    rs_h = s.groupby(["date", "hour"])["x"].mean()
    rsw = rs_h.unstack("hour").sort_index()
    mean7 = rsw.rolling(7, min_periods=3).mean()
    std7 = rsw.rolling(7, min_periods=3).std()
    z = ((rsw - mean7) / std7).replace([np.inf, -np.inf], np.nan)
    z.index = z.index.strftime("%Y-%m-%d")
    z.columns = [str(c) for c in z.columns]
    z.to_feather(os.path.join(FACTOR_DIR, f"{FINE_RESERVE}.fea"))

    # intraday lag-1 spread
    sp = pd.read_feather(os.path.join(DATA, "spread_label.feather"))
    sp.index = pd.to_datetime(sp.index).strftime("%Y-%m-%d")
    sp.columns = [str(c) for c in sp.columns]
    w = sp.astype(float)
    lag = w.shift(1, axis=1)          # previous hour, same row (same day)
    lag.iloc[:, 0] = np.nan            # first hour has no same-day predecessor
    lag.to_feather(os.path.join(FACTOR_DIR, f"{FINE_LAG}.fea"))


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────
def load_feature_series(name):
    p = os.path.join(FACTOR_DIR, f"{name}.fea")
    if not os.path.exists(p):
        return None
    df = pd.read_feather(p)
    df.columns = [str(c) for c in df.columns]
    if isinstance(df.index, pd.MultiIndex):
        # long-form (our own factor) already (date, hour)
        s = df[df.columns[0]].astype(float)
        s.name = name
        s.index = s.index.rename(["date", "hour"])
        return s.sort_index()
    # wide convention (date x 24 hours), as in the qlib158 factor library
    df.index = df.index.astype(str)
    s = df.stack().astype(float)
    s.name = name
    s.index = s.index.rename(["date", "hour"])
    return s.sort_index()


def build_feature_list(subset=None):
    """Return feature names (in fixed order). subset=N -> first N existing.

    In the standalone repo there is no deploy_v9 model to inherit the v9
    feature list from, so we build the list from the factor files present
    under data/factors/ (sorted).  Optionally, if a v9 model is available at
    ../../deploy_v9/models (development checkout), inherit its feature list.
    """
    feats = None
    lst = os.path.join(HERE, "..", "..", "deploy_v9", "models")
    import glob as _glob
    v9_models = sorted(_glob.glob(os.path.join(lst, "xgb_v9_*.joblib")))
    if v9_models:
        try:
            sys.path.insert(0, os.path.join(HERE, "..", "..", "deploy_v9", "code", "v9"))
            import v9_wrappers  # noqa: F401
            import joblib
            m = joblib.load(v9_models[-1])
            feats = list(m.get("features", []))
        except Exception:
            feats = None   # fall back to factor files
    if feats is None or not feats:
        feats = sorted(f.replace(".fea", "") for f in os.listdir(FACTOR_DIR))
        feats = [f for f in feats if f != FINE]
    # drop features we don't have files for
    avail = {f.replace(".fea", "") for f in os.listdir(FACTOR_DIR)}
    feats = [f for f in feats if f in avail]
    if FINE not in feats:
        feats.append(FINE)
    if subset:
        feats = feats[:subset]
        if FINE not in feats:
            feats.append(FINE)
    return feats


def load_X(feats):
    cols = []
    for name in feats:
        s = load_feature_series(name)
        if s is None:
            continue
        cols.append(s)
    X = pd.concat(cols, axis=1)
    X = X[~X.index.duplicated()].sort_index()
    return X


def load_y():
    sp = pd.read_feather(os.path.join(DATA, "spread_label.feather"))
    sp.index = pd.to_datetime(sp.index).strftime("%Y-%m-%d")
    sp.columns = [str(c) for c in sp.columns]
    y = sp.stack().astype(float)
    y.index = y.index.rename(["date", "hour"])
    return y.sort_index()


def to_class3(spr, t=T1):
    spr = np.asarray(spr, dtype=float)
    return np.where(spr < -t, 0, np.where(spr <= t, 1, 2))


# ──────────────────────────────────────────────────────────────────────
# Phase II mask (by day)
# ──────────────────────────────────────────────────────────────────────
def phase2_mask(dates, frac=0.30, seed=42):
    rng = np.random.default_rng(seed)
    uniq = np.unique(np.sort(dates))
    k = max(1, int(round(frac * len(uniq))))
    pick = set(uniq[rng.choice(len(uniq), size=k, replace=False)])
    return np.array([d in pick for d in dates])


# ──────────────────────────────────────────────────────────────────────
# Two-step regression calibration
# ──────────────────────────────────────────────────────────────────────
def calibrate_reserve(X_full, p2):
    """Estimate E[X_fine | Z] on Phase II, impute for all days.
    Z = the (available) features, using a small XGBoost regression on
    the Phase-II rows; X_hat returned for every row."""
    from xgboost import XGBRegressor
    fine_col = FINE
    z_cols = [c for c in X_full.columns if c != fine_col]
    sub = X_full[p2]
    # rows with fine present
    ok = sub[fine_col].notna()
    X_tr = sub.loc[ok, z_cols].astype(np.float32)
    y_tr = sub.loc[ok, fine_col].astype(np.float32)
    if len(X_tr) < 100 or np.ptp(y_tr.values) < 1e-9:
        # fall back: impute with the Phase-II mean
        mu = y_tr.mean()
        return pd.Series(mu, index=X_full.index, name=fine_col)
    reg = XGBRegressor(n_estimators=min(120, N_EST), max_depth=6,
                       learning_rate=0.05, subsample=0.8,
                       colsample_bytree=1.0, tree_method="hist",
                       random_state=RANDOM_STATE, verbosity=0, n_jobs=8)
    reg.fit(X_tr, y_tr)
    X_hat = reg.predict(X_full[z_cols].astype(np.float32))
    out = pd.Series(X_hat, index=X_full.index, name=fine_col)
    # keep the true values where observed (Phase II)
    out.loc[X_full[fine_col].notna()] = X_full.loc[X_full[fine_col].notna(), fine_col]
    return out


# ──────────────────────────────────────────────────────────────────────
# Hour prior
# ──────────────────────────────────────────────────────────────────────
def build_hour_prior(y_tr):
    yw = y_tr.unstack()
    frac, vote = {}, {}
    for h in yw.columns:
        v = yw[h].dropna().values
        n_neg = float(np.mean(v < -T1))
        n_pos = float(np.mean(v > T1))
        denom = n_neg + n_pos
        f = n_neg / denom if denom > 1e-9 else 0.5
        frac[str(h)[:2]] = f
        vote[str(h)[:2]] = (-1 if f >= 0.55 else (1 if f <= 0.45 else 0))
    return frac, vote


# ──────────────────────────────────────────────────────────────────────
# Train the two heads (v9 structure)
# ──────────────────────────────────────────────────────────────────────
def train_heads(X_tr, y_tr, X_va, y_va, hour_vote):
    yc_tr = to_class3(y_tr.values, T1)
    yc_va = to_class3(y_va.values, T1)
    dist = np.bincount(yc_tr, minlength=3)
    bal = len(yc_tr) / (3 * np.maximum(dist, 1))
    COST = np.array([2.0, 1.0, 2.0])
    sw_tr = (bal[yc_tr] * COST[yc_tr]).astype(np.float32)
    sw_va = (bal[yc_va] * COST[yc_va]).astype(np.float32)

    def dir_hit(y_true, y_pred):
        return "dir_hit", 0.0   # dummy; we use default for stability

    clf = xgb.XGBClassifier(
        n_estimators=N_EST, early_stopping_rounds=ES,
        eval_metric="mlogloss", random_state=RANDOM_STATE, verbosity=0,
        **FIXED)
    clf.fit(X_tr, yc_tr, sample_weight=sw_tr,
            eval_set=[(X_va, yc_va)], sample_weight_eval_set=[sw_va],
            verbose=False)

    mag = np.abs(y_tr.values)
    sw_mag = np.where(mag > T1, 1.0, 0.2).astype(np.float32)
    target = np.log1p(mag).astype(np.float32)
    reg = xgb.XGBRegressor(n_estimators=N_EST, random_state=RANDOM_STATE,
                           verbosity=0, **FIXED)
    reg.fit(X_tr, target, sample_weight=sw_mag, verbose=False)
    return clf, reg


# ──────────────────────────────────────────────────────────────────────
# C-strategy evaluation (v9 metrics + P&L)
# ──────────────────────────────────────────────────────────────────────
def evaluate(clf, reg, X_va, y_va, hour_vote):
    act = np.asarray(y_va.values, dtype=float)
    hours = np.asarray([h[:2] for h in X_va.index.get_level_values("hour")])
    mag = np.expm1(np.asarray(reg.predict(X_va), dtype=float))
    as_ = np.where(act < -T1, -1, np.where(act > T1, 1, 0))
    prior = np.array([hour_vote.get(h, 0) for h in hours])
    trig = (mag >= T1) & (prior != 0)
    pred_dir = np.where(trig, prior, 0)
    pl = np.zeros(len(act))
    correct = (pred_dir != 0) & (as_ != 0) & (pred_dir == as_)
    wrong = (pred_dir != 0) & (as_ != 0) & (pred_dir != as_)
    pl[correct] = np.abs(act[correct])
    pl[wrong] = -np.abs(act[wrong])
    cum = np.cumsum(pl)
    sel = trig & (as_ != 0)
    hit = float(np.mean(pred_dir[sel] == as_[sel])) if sel.sum() else float("nan")
    nw = correct.sum() + wrong.sum()
    return dict(
        n_hours=int(len(act)),
        n_trigger=int(trig.sum()), trigger_rate=float(trig.mean()),
        dir_hit=hit,
        net_win_rate=float(correct.sum() / nw) if nw else float("nan"),
        n_correct=int(correct.sum()), n_wrong=int(wrong.sum()),
        pnl_total=float(pl.sum()),
        max_drawdown=float((np.maximum.accumulate(cum) - cum).max()),
    )


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────
def main():
    global FINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-days", type=int, default=60)
    ap.add_argument("--subset", type=int, default=0,
                    help="smoke test: use only first N features")
    ap.add_argument("--frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fine", choices=[FINE_LAG, FINE_RESERVE],
                    default=FINE_DEFAULT,
                    help="fine covariate for Phase II (default: intraday lag-1 spread)")
    args = ap.parse_args()
    FINE = args.fine

    print("=" * 72, flush=True)
    print(f"  v10 two-phase prediction | τ={T1} τ_big={T2} | valid={args.valid_days}d "
          f"| phase-II frac={args.frac} | subset={args.subset or 'all'} "
          f"| fine={FINE}", flush=True)
    print("=" * 72, flush=True)

    build_fine_factors()
    print(f"  (re)built fine factors: {FINE_RESERVE}, {FINE_LAG}", flush=True)

    feats = build_feature_list(subset=args.subset)
    print(f"  features: {len(feats)} (incl. fine={FINE})", flush=True)
    X = load_X(feats)
    y = load_y()
    common = X.index.intersection(y.index)
    Xc, yc = X.loc[common], y.loc[common]
    print(f"  X {Xc.shape} | y {len(yc)} | {Xc.index.get_level_values('date').min()} ~ "
          f"{Xc.index.get_level_values('date').max()}", flush=True)

    dates = Xc.index.get_level_values("date").astype(str).values
    uniq_dates = np.unique(dates)
    tr_d = set(uniq_dates[: len(uniq_dates) - args.valid_days])
    va_d = set(uniq_dates[len(uniq_dates) - args.valid_days:])
    tr_m = np.isin(dates, list(tr_d))
    va_m = np.isin(dates, list(va_d))
    assert not (set(dates[tr_m]) & va_d), "valid leaked into train"

    # Phase-II mask on TRAIN days only (Phase-II concept applies to known days)
    tr_dates = uniq_dates[: len(uniq_dates) - args.valid_days]
    p2_tr = phase2_mask(tr_dates, frac=args.frac, seed=args.seed)
    p2_map = dict(zip(tr_dates, p2_tr))
    p2_train = np.array([p2_map.get(d, False) for d in dates])
    print(f"  Phase II (train days): {int(p2_train[tr_m].sum())}/{int(tr_m.sum())} rows "
          f"({100 * p2_train[tr_m].mean():.0f}%)", flush=True)

    # hour prior from train
    frac_p, vote = build_hour_prior(yc.loc[tr_m])
    print(f"  hour prior votes: {vote}", flush=True)

    # ── build the three treatments ──
    results = {}
    for mode in ["all", "phase2_only", "twostep"]:
        print(f"\n[+]{mode}", flush=True)
        Xm = Xc.copy()
        # mode all: fine covariate visible everywhere (oracle upper bound)
        # phase2_only: fine present on Phase-II train rows only; NaN elsewhere
        # twostep: calibrate on Phase-II, impute for all rows
        if mode == "all":
            pass
        elif mode == "phase2_only":
            mask_all = p2_train  # fine observed only on Phase-II train days
            # also set NaN on validation (realistic: valid = future, no fine yet)
            Xm.loc[~mask_all, FINE] = np.nan
        elif mode == "twostep":
            # calibrate on train Phase-II rows; impute over all rows
            Xhat = calibrate_reserve(Xm.loc[tr_m], p2_train[tr_m])
            Xm.loc[tr_m, FINE] = Xhat.values
            # for valid, we have no fine at all -> keep NaN (v9-style missing)
            Xm.loc[va_m, FINE] = np.nan

        X_tr = Xm.loc[tr_m].astype(np.float32)
        y_tr = yc.loc[tr_m]
        X_va = Xm.loc[va_m].astype(np.float32)
        y_va = yc.loc[va_m]

        clf, reg = train_heads(X_tr, y_tr, X_va, y_va, vote)
        ev = evaluate(clf, reg, X_va, y_va, vote)
        results[mode] = ev
        print(f"  valid: n_trig {ev['n_trigger']} | dir_hit {ev['dir_hit']:.3f} | "
              f"net_win {ev['net_win_rate']:.3f} | P&L {ev['pnl_total']:+.0f} | "
              f"MDD {ev['max_drawdown']:.0f}", flush=True)

    # ── summary ──
    print("\n" + "=" * 72, flush=True)
    print("  v10 comparison on same walk-forward valid window", flush=True)
    print("=" * 72, flush=True)
    print(f"  {'mode':<12}{'trig':>6}{'dir_hit':>9}{'net_win':>9}{'P&L':>10}{'MDD':>8}", flush=True)
    for m in ["all", "phase2_only", "twostep"]:
        e = results[m]
        print(f"  {m:<12}{e['n_trigger']:>6}{e['dir_hit']:>9.3f}{e['net_win_rate']:>9.3f}"
              f"{e['pnl_total']:>10.0f}{e['max_drawdown']:>8.0f}", flush=True)

    out = {
        "model": "v10_two_phase",
        "threshold": T1, "threshold_big": T2,
        "fine_covariate": FINE,
        "valid_days": args.valid_days, "frac_phase2": args.frac, "seed": args.seed,
        "n_features": len(feats),
        "feature_subset": args.subset or "all",
        "n_train_rows": int(tr_m.sum()), "n_valid_rows": int(va_m.sum()),
        "hour_prior": vote,
        "valid_window": [min(va_d), max(va_d)],
        "results": results,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "method": "v9 dual-head (dir XGB + mag XGB + hour prior + C rule) with "
                  "two-phase fine covariate (positive reserve): all=visible everywhere, "
                  "phase2_only=NaN outside Phase-II, twostep=regression-calibrated "
                  "E[X|Z] imputed to all days.",
    }
    with open(os.path.join(OUT, "v10_results.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  -> wrote {os.path.join(OUT, 'v10_results.json')}", flush=True)


if __name__ == "__main__":
    main()
