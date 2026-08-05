"""
Leakage & AUC diagnostic script.

Reruns pipeline data-prep (steps 1-7 only -- no model training) to obtain
the real X_train / y_train feature matrix, then runs four diagnostic checks:

  Check 1  -- Single-feature AUC: does any feature alone reach > 0.90 AUC?
  Check 2  -- Spearman/Pearson correlation with label: any feature with |r| > 0.80?
  Check 3  -- Business-process co-definition: bad rate by outstanding-debt status
  Check 4  -- Thin-file vs thick-file population breakdown + scorecard AUC
  Check 4b -- Thin-file LR validation: bootstrap CI, decile table, coefficient signs

Usage (Windows CMD, same args as run_pipeline.py):
    python run_diagnostics.py ^
        --train-file    data\snapshot_20250930.csv ^
        --val-file      data\snapshot_20251115.csv ^
        --repayment-file data\repayments.csv ^
        --train-snapshot-date 20250930 ^
        --val-snapshot-date   20251115 ^
        --train-cutoff  2025-10-31 ^
        --output-dir    artifacts

PowerShell:
    python run_diagnostics.py `
        --train-file    data/snapshot_20250930.csv `
        --val-file      data/snapshot_20251115.csv `
        --repayment-file data/repayments.csv `
        --train-snapshot-date 20250930 `
        --val-snapshot-date   20251115 `
        --train-cutoff  2025-10-31 `
        --output-dir    artifacts
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pd_model.config import feature_config
from pd_model.config.model_config import DEFAULT_CONFIG
from pd_model.data_prep.pipeline_data import prepare_pd_training_and_validation_data
from pd_model.logging_config import get_logger
from pd_model.modeling.scorecard import (
    add_never_loan_scorecard_from_phase_2_1,
    apply_thin_file_lr,
    fit_thin_file_lr,
)
from pd_model.preprocessing.loan_features import (
    add_thin_file_flags,
    classify_agent_loan_status,
    leakage_audit_phase_2_2,
    run_phase_2_2_repayment_pd_features,
)
from pd_model.preprocessing.transaction_features import run_phase_2_1_richer_tx_behaviour
from pd_model.preprocessing.transformations import (
    build_transformed_dataframe,
    get_and_classify_pd_features,
)
from pd_model.scoring.iv_selector import iv_filter_phase_2
from pd_model.tests.test_single_feature_auc import single_feature_aucs
from pd_model.tests.test_feature_target_correlation import feature_label_correlations
from pd_model.tests.test_business_process_codef import business_process_codef_report

logger = get_logger(__name__)

DIVIDER = "=" * 70


def _load_snapshot(path: str, snapshot_date_int: int, split_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["snapshot_dt"] = snapshot_date_int
    df["split"] = split_label
    df["tbl_dt"] = pd.to_numeric(df.get("tbl_dt", snapshot_date_int), errors="coerce")
    logger.info("Loaded %s: %d rows, %d cols", path, len(df), df.shape[1])
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "tbl_dt" in df.columns:
        df["tbl_dt"] = pd.to_datetime(df["tbl_dt"].astype(str), errors="coerce")
    if "snapshot_dt" in df.columns:
        df["snapshot_dt"] = pd.to_datetime(df["snapshot_dt"].astype(str), errors="coerce")
    if "activation_dt" in df.columns:
        df["activation_dt"] = df["activation_dt"].astype(str).str.split(".").str[0].replace("", pd.NA)
        df["activation_dt"] = pd.to_datetime(df["activation_dt"], format="%Y%m%d", errors="coerce")
    for c in ["date_of_birth", "payment_last", "cash_in_last", "cash_out_last"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str), errors="coerce")
    return df


def run_diagnostics(args: argparse.Namespace) -> None:
    cfg = DEFAULT_CONFIG

    # ------------------------------------------------------------------ #
    # Steps 1-3: Load data, run feature engineering, join labels
    # ------------------------------------------------------------------ #
    print(f"\n{DIVIDER}")
    print("STEP 1-3: Loading data and building features ...")
    print(DIVIDER)

    train_df = _load_snapshot(args.train_file, args.train_snapshot_date, "train")
    val_df   = _load_snapshot(args.val_file,   args.val_snapshot_date,   "validation")

    df_pd = pd.concat([train_df, val_df], ignore_index=True, sort=False)
    df_pd = _parse_dates(df_pd)

    df_pd = run_phase_2_1_richer_tx_behaviour(df_pd, cfg=cfg)

    df_repayments = pd.read_csv(args.repayment_file)
    logger.info("Repayments loaded: %d rows", len(df_repayments))

    df_pd, _ = run_phase_2_2_repayment_pd_features(df_pd, df_repayments, cfg=cfg, verbose=True)
    df_pd = classify_agent_loan_status(df_pd)
    df_pd = add_thin_file_flags(df_pd, cfg=cfg)

    leakage_audit_phase_2_2(
        df_pd, hard_fail=True, cfg=cfg,
        pd_feature_blacklist=feature_config.PD_FEATURE_BLACKLIST,
    )

    df_pd = add_never_loan_scorecard_from_phase_2_1(df_pd, cfg=cfg)

    # ------------------------------------------------------------------ #
    # Steps 4-5: Transform features (fit on train only)
    # ------------------------------------------------------------------ #
    train_cutoff = pd.Timestamp(args.train_cutoff)
    split_mask   = pd.to_datetime(df_pd["snapshot_dt"], errors="coerce") <= train_cutoff
    df_train_raw = df_pd[split_mask].copy()
    df_val_raw   = df_pd[~split_mask].copy()

    # Fit thin-file LR on training data only (mirrors Step 5b in run_pipeline.py)
    _thin_lr_pipeline, _thin_lr_features = fit_thin_file_lr(
        df_train_raw, label_col=feature_config.TARGET_COL
    )
    if _thin_lr_pipeline is not None:
        df_train_raw = apply_thin_file_lr(df_train_raw, _thin_lr_pipeline, _thin_lr_features)
        df_val_raw   = apply_thin_file_lr(df_val_raw,   _thin_lr_pipeline, _thin_lr_features)

    pd_features, log_cols, cap_cols, _, signed_log_cols, _ = get_and_classify_pd_features(df_train_raw)

    df_train_trans, _, transform_report = build_transformed_dataframe(
        df_train_raw, pd_features=pd_features, log_cols=log_cols,
        cap_cols=cap_cols, signed_log_cols=signed_log_cols, cfg=cfg,
    )
    fitted_params = {
        row["feature"]: {
            "action": row["action"],
            "lo": row.get("lo") if not pd.isna(row.get("lo", float("nan"))) else None,
            "hi": row.get("hi") if not pd.isna(row.get("hi", float("nan"))) else None,
        }
        for _, row in transform_report.iterrows()
        if pd.notna(row.get("action"))
    }
    df_val_trans, _, _ = build_transformed_dataframe(
        df_val_raw, pd_features=pd_features, log_cols=log_cols,
        cap_cols=cap_cols, signed_log_cols=signed_log_cols, cfg=cfg,
        fitted_params=fitted_params,
    )
    val_cols = [c for c in df_train_trans.columns if c in df_val_trans.columns]
    df_val_trans = df_val_trans[val_cols]

    df_pd_raw        = pd.concat([df_train_raw, df_val_raw],   ignore_index=True)
    df_pd_transformed = pd.concat([df_train_trans, df_val_trans], ignore_index=True)

    # ------------------------------------------------------------------ #
    # Step 6: Train/val split + IV feature selection
    # ------------------------------------------------------------------ #
    (
        X_train_raw, X_train_trans, y_train,
        X_val_raw,   X_val_trans,   y_val,
        candidate_features, *_
    ) = prepare_pd_training_and_validation_data(
        df_pd_raw=df_pd_raw,
        df_pd_transformed=df_pd_transformed,
        target_col=feature_config.TARGET_COL,
        train_cutoff=train_cutoff,
        id_cols=[feature_config.AGENT_KEY],
        protected_cols=list(feature_config.PD_FEATURE_BLACKLIST),
        pd_feature_blacklist=feature_config.PD_FEATURE_BLACKLIST,
        forbidden_feature_patterns=feature_config.LEAKAGE_PATTERNS,
        date_cols=feature_config.DATE_COLS,
    )

    selected_features, _ = iv_filter_phase_2(
        X_train_raw=X_train_raw,
        X_train_transformed=X_train_trans,
        y_train=y_train,
        cfg=cfg,
        pd_feature_blacklist=feature_config.PD_FEATURE_BLACKLIST,
        forbidden_feature_patterns=feature_config.LEAKAGE_PATTERNS,
        target_col=feature_config.TARGET_COL,
    )

    X_train = X_train_trans[selected_features].copy()
    X_val   = X_val_trans[[c for c in selected_features if c in X_val_trans.columns]].copy()

    print(f"\nTraining matrix: {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
    print(f"Bad rate (train): {y_train.mean():.2%}  |  Bad rate (val): {y_val.mean():.2%}")

    # ------------------------------------------------------------------ #
    # CHECK 1: Single-feature AUC
    # ------------------------------------------------------------------ #
    print(f"\n{DIVIDER}")
    print("CHECK 1: Single-feature AUC  (ALERT threshold > 0.90)")
    print(DIVIDER)
    auc_report = single_feature_aucs(X_train, y_train)
    print(auc_report.head(15).to_string(index=False))

    alert_feats = auc_report[auc_report["flag"] == "ALERT"]
    warn_feats  = auc_report[auc_report["flag"] == "WARN"]
    print(f"\n  ALERT features (AUC > 0.90): {len(alert_feats)}")
    print(f"  WARN  features (AUC > 0.85): {len(warn_feats)}")
    if not alert_feats.empty:
        print("\n  *** ALERT -- investigate these features for leakage or co-definition: ***")
        print(alert_feats[["feature", "auc"]].to_string(index=False))

    # ------------------------------------------------------------------ #
    # CHECK 2: Feature-label Spearman / Pearson correlation
    # ------------------------------------------------------------------ #
    print(f"\n{DIVIDER}")
    print("CHECK 2: Feature-label correlation  (ALERT threshold |r| > 0.80)")
    print(DIVIDER)
    corr_report = feature_label_correlations(X_train, y_train)
    print(corr_report.head(15).to_string(index=False))

    alert_corr = corr_report[corr_report["flag"] == "ALERT"]
    print(f"\n  ALERT features (|r| > 0.80): {len(alert_corr)}")
    if not alert_corr.empty:
        print("\n  *** ALERT -- high correlation with label: ***")
        print(alert_corr[["feature", "spearman_r", "pearson_r"]].to_string(index=False))

    # ------------------------------------------------------------------ #
    # CHECK 3: Business-process co-definition
    # ------------------------------------------------------------------ #
    print(f"\n{DIVIDER}")
    print("CHECK 3: Business-process co-definition (net_exposure_6M vs bad_state)")
    print(DIVIDER)

    codef_df = df_train_raw[
        [c for c in ["net_exposure_6M", "currently_outstanding_flag", "bad_state"]
         if c in df_train_raw.columns]
    ].copy()

    if "bad_state" not in codef_df.columns and feature_config.TARGET_COL in df_train_raw.columns:
        codef_df["bad_state"] = df_train_raw[feature_config.TARGET_COL].values

    if "bad_state" in codef_df.columns and "net_exposure_6M" in codef_df.columns:
        codef_report = business_process_codef_report(
            codef_df,
            outstanding_col="currently_outstanding_flag",
            net_exposure_col="net_exposure_6M",
            label_col="bad_state",
        )
        print(codef_report.to_string(index=False))
        outstanding_row = codef_report[codef_report["segment"] == "outstanding"]
        if not outstanding_row.empty:
            out_br   = outstanding_row["bad_rate"].iloc[0]
            out_lift = outstanding_row["lift"].iloc[0]
            print(f"\n  Outstanding agents bad rate : {out_br:.2%}")
            print(f"  Lift vs overall             : {out_lift:.2f}x")
            if out_br > 0.30:
                print("  *** WARNING: bad rate > 30% when outstanding -- strong co-definition ***")
            elif out_br > 0.15:
                print("  NOTE: moderate lift; net_exposure_6M is a strong but not deterministic signal")
            else:
                print("  OK: outstanding status is not near-deterministic of default")
    else:
        print("  SKIP: net_exposure_6M or bad_state not found in training DataFrame")

    # ------------------------------------------------------------------ #
    # CHECK 4: Thin-file vs thick-file population breakdown
    # ------------------------------------------------------------------ #
    print(f"\n{DIVIDER}")
    print("CHECK 4: Thin-file vs thick-file population breakdown")
    print(DIVIDER)

    thin_col = feature_config.THIN_FILE_COL  # "thin_file_flag"
    label_col = feature_config.TARGET_COL    # "bad_state"

    if thin_col in df_train_raw.columns and label_col in df_train_raw.columns:
        from sklearn.metrics import roc_auc_score

        tf = df_train_raw[[thin_col, label_col]].copy()
        tf[thin_col] = pd.to_numeric(tf[thin_col], errors="coerce")
        tf[label_col] = pd.to_numeric(tf[label_col], errors="coerce")
        tf = tf.dropna()

        n_total     = len(tf)
        n_thin      = int((tf[thin_col] == 1).sum())
        n_thick     = int((tf[thin_col] == 0).sum())
        br_thin     = float(tf.loc[tf[thin_col] == 1, label_col].mean()) if n_thin else float("nan")
        br_thick    = float(tf.loc[tf[thin_col] == 0, label_col].mean()) if n_thick else float("nan")
        br_overall  = float(tf[label_col].mean())

        print(f"  {'Segment':<20} {'n':>10} {'share':>8} {'bad_rate':>10}")
        print(f"  {'-'*50}")
        print(f"  {'thin-file (scorecard)':<20} {n_thin:>10,} {n_thin/n_total:>8.1%} {br_thin:>10.2%}")
        print(f"  {'thick-file (PD model)':<20} {n_thick:>10,} {n_thick/n_total:>8.1%} {br_thick:>10.2%}")
        print(f"  {'overall':<20} {n_total:>10,} {'100%':>8} {br_overall:>10.2%}")

        print()
        from sklearn.metrics import roc_auc_score

        thin_mask  = pd.to_numeric(df_train_raw[thin_col], errors="coerce") == 1
        thick_mask = ~thin_mask
        y_all = pd.to_numeric(df_train_raw[label_col], errors="coerce")

        # Thin-file AUC: use the scorecard's own PD estimate (never_loan_pd_like)
        # XGBoost does not score thin-file agents in production -- the scorecard does.
        scorecard_score_col = "never_loan_pd_like"
        if scorecard_score_col in df_train_raw.columns:
            sc_score = pd.to_numeric(df_train_raw[scorecard_score_col], errors="coerce")
            valid_thin = thin_mask & sc_score.notna() & y_all.notna()
            if valid_thin.sum() >= 50 and y_all[valid_thin].nunique() == 2:
                auc_thin = roc_auc_score(y_all[valid_thin], sc_score[valid_thin])
                auc_thin = max(auc_thin, 1 - auc_thin)
                print(f"  Scorecard AUC on thin-file agents  : {auc_thin:.4f}  (n={valid_thin.sum():,})")
            else:
                print(f"  Scorecard AUC on thin-file agents  : insufficient data")
        else:
            print(f"  Scorecard AUC on thin-file agents  : '{scorecard_score_col}' not found in DataFrame")

        # Thick-file AUC: use the top XGBoost feature as a proxy for the model's signal
        # (full model AUC requires running XGBoost -- use top feature as lower-bound indicator)
        top_feat = auc_report.iloc[0]["feature"]
        if top_feat in df_train_raw.columns:
            x_feat = pd.to_numeric(df_train_raw[top_feat], errors="coerce")
            valid_thick = thick_mask & x_feat.notna() & y_all.notna()
            if valid_thick.sum() >= 50 and y_all[valid_thick].nunique() == 2:
                auc_thick = roc_auc_score(y_all[valid_thick], x_feat[valid_thick])
                auc_thick = max(auc_thick, 1 - auc_thick)
                print(f"  Top feature AUC on thick-file only : {auc_thick:.4f}  (n={valid_thick.sum():,})")
                print(f"  (Full XGBoost AUC on thick-file would be higher -- this is a single-feature lower bound)")

        if n_thin > 0:
            print(f"\n  NOTE: {n_thin:,} thin-file agents ({n_thin/n_total:.1%}) go through the scorecard.")
            print(f"  The overall 0.989 AUC combines both populations. Scorecard AUC above")
            print(f"  shows whether the rule-based path discriminates thin-file risk adequately.")
    else:
        print(f"  SKIP: '{thin_col}' or '{label_col}' not found in training DataFrame")

    # ------------------------------------------------------------------ #
    # CHECK 4b: Thin-file LR validation
    #   i)  Bootstrap 95% CI on thin-file val AUC
    #   ii) Decile table (bad rate by score decile, thin-file val)
    #   iii)LR coefficient signs (do directions make business sense?)
    # ------------------------------------------------------------------ #
    print(f"\n{DIVIDER}")
    print("CHECK 4b: Thin-file LR validation (bootstrap CI / decile table / coefficients)")
    print(DIVIDER)

    thin_col  = feature_config.THIN_FILE_COL
    label_col = feature_config.TARGET_COL

    if _thin_lr_pipeline is None:
        print("  SKIP: LR was not fitted (insufficient positives in thin-file train set)")
    elif thin_col not in df_val_raw.columns or "never_loan_pd_like" not in df_val_raw.columns:
        print(f"  SKIP: '{thin_col}' or 'never_loan_pd_like' not in df_val_raw")
    else:
        from sklearn.metrics import roc_auc_score

        _thin_vmask = pd.to_numeric(df_val_raw[thin_col], errors="coerce").eq(1)
        _df_tv      = df_val_raw[_thin_vmask].copy()
        _y_tv       = pd.to_numeric(_df_tv[label_col], errors="coerce").fillna(0).astype(int)
        _s_tv       = pd.to_numeric(_df_tv["never_loan_pd_like"], errors="coerce")
        _valid      = _s_tv.notna() & _y_tv.notna()
        _y_arr      = _y_tv[_valid].values
        _s_arr      = _s_tv[_valid].values

        n_thin_val = len(_y_arr)
        n_pos_val  = int(_y_arr.sum())

        print(f"\n  [i] Bootstrap 95% CI  (n_thin_val={n_thin_val:,}, n_positives={n_pos_val})")
        print(f"  {'-'*50}")

        if n_pos_val < 5:
            print(f"  SKIP: only {n_pos_val} positives in thin-file val -- AUC estimate unreliable")
        else:
            _point_auc = float(roc_auc_score(_y_arr, _s_arr))
            rng = np.random.default_rng(42)
            _boot_aucs = []
            for _ in range(200):
                _idx = rng.integers(0, n_thin_val, n_thin_val)
                _yb, _sb = _y_arr[_idx], _s_arr[_idx]
                if 0 < _yb.sum() < len(_yb):
                    try:
                        _boot_aucs.append(roc_auc_score(_yb, _sb))
                    except Exception:
                        pass

            if _boot_aucs:
                _ci_lo = float(np.percentile(_boot_aucs, 2.5))
                _ci_hi = float(np.percentile(_boot_aucs, 97.5))
                _width = _ci_hi - _ci_lo
                print(f"  Point AUC     : {_point_auc:.4f}")
                print(f"  95% CI        : [{_ci_lo:.4f}, {_ci_hi:.4f}]")
                print(f"  CI width      : {_width:.4f}")
                if _width > 0.15:
                    print("  *** WARNING: CI is wide -- point estimate has high variance."
                          f" Report as ~{_point_auc:.2f} ± {_width/2:.2f}, not a precise figure ***")
                elif _width > 0.08:
                    print("  NOTE: moderate CI width -- treat point estimate as approximate")
                else:
                    print("  OK: CI is tight -- point estimate is stable")
            else:
                print("  WARNING: bootstrap produced no valid samples")

        # -- ii) Decile table --------------------------------------------
        print(f"\n  [ii] Decile table  (thin-file val, sorted by LR score desc)")
        print(f"  {'-'*50}")

        if n_pos_val < 5:
            print(f"  SKIP: too few positives ({n_pos_val}) for a meaningful decile table")
        else:
            _dec_df = pd.DataFrame({"y": _y_arr, "score": _s_arr})
            _dec_df = _dec_df.sort_values("score", ascending=False).reset_index(drop=True)
            _n      = len(_dec_df)
            _dec_df["decile"] = (_dec_df.index * 10 // _n) + 1

            _overall_br = float(_y_arr.mean())
            _dec_tbl = (
                _dec_df.groupby("decile")
                .agg(n=("y", "count"), n_bad=("y", "sum"), bad_rate=("y", "mean"))
                .reset_index()
            )
            _dec_tbl["lift"] = (_dec_tbl["bad_rate"] / _overall_br).round(2)
            _dec_tbl["bad_rate_pct"] = (_dec_tbl["bad_rate"] * 100).round(3)

            print(f"  {'Decile':>7} {'n':>7} {'n_bad':>6} {'bad_rate%':>10} {'lift':>6}")
            print(f"  {'-'*42}")
            for _, row in _dec_tbl.iterrows():
                print(
                    f"  {int(row['decile']):>7} {int(row['n']):>7,} {int(row['n_bad']):>6}"
                    f" {row['bad_rate_pct']:>10.3f} {row['lift']:>6.2f}x"
                )
            print(f"  {'-'*42}")
            print(f"  Overall bad rate: {_overall_br*100:.3f}%")

            _top_lift = float(_dec_tbl.loc[_dec_tbl["decile"] == 1, "lift"].iloc[0]) \
                if 1 in _dec_tbl["decile"].values else float("nan")
            if _top_lift >= 3.0:
                print(f"  OK: top decile lift={_top_lift:.1f}x -- model concentrates risk effectively")
            elif _top_lift >= 1.5:
                print(f"  NOTE: top decile lift={_top_lift:.1f}x -- moderate concentration")
            else:
                print(f"  WARNING: top decile lift={_top_lift:.1f}x -- model barely separates risk")

            # Check monotonicity (top 5 deciles should generally trend down)
            _top5_rates = _dec_tbl.loc[_dec_tbl["decile"] <= 5, "bad_rate"].tolist()
            _monotone = all(_top5_rates[i] >= _top5_rates[i+1] for i in range(len(_top5_rates)-1))
            if not _monotone:
                print("  NOTE: bad rate is not strictly monotone across top 5 deciles"
                      " (expected with small n -- check if directional trend holds)")

        # -- iii) Coefficient signs ---------------------------------------
        print(f"\n  [iii] LR coefficient signs  (are risk directions intuitive?)")
        print(f"  {'-'*50}")

        _lr_step = _thin_lr_pipeline.named_steps["lr"]
        _coef_df = pd.DataFrame({
            "feature":     _thin_lr_features,
            "coefficient": _lr_step.coef_[0],
        }).sort_values("coefficient", key=abs, ascending=False).reset_index(drop=True)
        _coef_df["direction"] = _coef_df["coefficient"].apply(
            lambda c: "(+) riskier" if c > 0 else "(-) safer"
        )

        print(f"  {'Feature':<40} {'Coeff':>8}  Direction")
        print(f"  {'-'*60}")
        for _, row in _coef_df.iterrows():
            print(f"  {row['feature']:<40} {row['coefficient']:>8.4f}  {row['direction']}")

        # Features that should be risk-increasing (positive coeff) by name convention
        _expected_risk = {
            "is_fully_inactive_6m", "is_consecutively_inactive",
            "sharp_volume_drop_flag", "consistent_volume_decline_flag",
            "low_balance_flag", "balance_drawdown_flag",
            "net_cash_flow_negative_flag", "high_peer_dependency_flag",
            "cust_concentration_flag", "commission_without_activity_flag",
            "commission_drop_flag",
        }
        # Features expected to be protective (negative coeff)
        _expected_safe = {"activity_restart_flag", "consistent_volume_growth_flag"}

        _counterintuitive = []
        for _, row in _coef_df.iterrows():
            f = row["feature"]
            c = row["coefficient"]
            if f in _expected_risk and c < 0:
                _counterintuitive.append(f"  {f} (expected +, got {c:.4f})")
            elif f in _expected_safe and c > 0:
                _counterintuitive.append(f"  {f} (expected -, got {c:.4f})")

        print()
        if _counterintuitive:
            print("  *** WARNING: counterintuitive sign(s) -- investigate before presenting: ***")
            for msg in _counterintuitive:
                print(msg)
        else:
            print("  OK: all named risk/protective features have the expected sign")

        # -- iv) Bad-rate-by-flag breakdown --------------------------------
        print(f"\n  [iv] Bad-rate-by-flag breakdown  (thin-file TRAIN, binary features only)")
        print(f"  Does the raw data agree with each LR coefficient direction?")
        print(f"  {'-'*95}")
        print(
            f"  {'Feature':<40} {'n(0)':>7} {'BR%(0)':>7} {'n(1)':>7} {'BR%(1)':>7}"
            f" {'LR says':>8} {'Data says':>10} {'Match':>6}"
        )
        print(f"  {'-'*95}")

        _thin_tr_mask = pd.to_numeric(
            df_train_raw.get(feature_config.THIN_FILE_COL, 0), errors="coerce"
        ).eq(1)
        _df_thin_tr  = df_train_raw[_thin_tr_mask].copy()
        _y_thin_tr   = pd.to_numeric(
            _df_thin_tr.get(feature_config.TARGET_COL, 0), errors="coerce"
        )

        _coef_lookup = dict(zip(_thin_lr_features, _lr_step.coef_[0]))

        for _feat in _thin_lr_features:
            if _feat not in _df_thin_tr.columns:
                continue
            _fvals = pd.to_numeric(_df_thin_tr[_feat], errors="coerce").fillna(0)
            # Only show binary (0/1) features
            if not set(_fvals.unique()).issubset({0, 1, 0.0, 1.0}):
                continue

            _m0 = _fvals == 0
            _m1 = _fvals == 1
            _n0, _n1 = int(_m0.sum()), int(_m1.sum())
            _br0 = float(_y_thin_tr[_m0].mean()) * 100 if _n0 > 0 else float("nan")
            _br1 = float(_y_thin_tr[_m1].mean()) * 100 if _n1 > 0 else float("nan")

            _coef      = _coef_lookup.get(_feat, 0.0)
            _lr_says   = "(+) riskier" if _coef > 0 else "(-) safer"
            _data_says = "(+) riskier" if (
                not np.isnan(_br1) and not np.isnan(_br0) and _br1 > _br0
            ) else "(-) safer"
            _match     = "OK" if _lr_says == _data_says else "WARN"

            print(
                f"  {_feat:<40} {_n0:>7,} {_br0:>7.3f} {_n1:>7,} {_br1:>7.3f}"
                f" {_lr_says:>8} {_data_says:>10} {_match:>6}"
            )

        print(f"  {'-'*95}")
        print(
            "  BR%(0) = bad rate % when flag=0 | BR%(1) = bad rate % when flag=1\n"
            "  WARN = LR coefficient direction contradicts the raw data -- consider dropping that feature"
        )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\n{DIVIDER}")
    print("SUMMARY")
    print(DIVIDER)
    top_auc = auc_report.iloc[0]
    print(f"  Top single-feature AUC : {top_auc['feature']}  =  {top_auc['auc']:.4f}  [{top_auc['flag']}]")
    top_corr = corr_report.iloc[0]
    print(f"  Top label correlation  : {top_corr['feature']}  Spearman r = {top_corr['spearman_r']:.4f}  [{top_corr['flag']}]")
    print(f"  ALERT features (AUC)   : {len(alert_feats)}")
    print(f"  ALERT features (corr)  : {len(alert_corr)}")
    if len(alert_feats) == 0 and len(alert_corr) == 0:
        print("\n  VERDICT: No single feature drives AUC above the leakage threshold.")
        print("  The 0.989 AUC likely reflects a combination of genuine pre-snapshot")
        print("  behavioural signals, not a single leaked or co-defined feature.")
    else:
        print("\n  VERDICT: One or more features require investigation (see ALERT above).")
    print(DIVIDER)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Leakage & AUC diagnostic -- reruns data-prep only, no model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-file",           required=True,  help="Training snapshot CSV")
    p.add_argument("--val-file",             required=True,  help="Validation snapshot CSV")
    p.add_argument("--repayment-file",       required=True,  help="Repayments CSV")
    p.add_argument("--train-snapshot-date",  required=True,  type=int, help="YYYYMMDD int")
    p.add_argument("--val-snapshot-date",    required=True,  type=int, help="YYYYMMDD int")
    p.add_argument("--train-cutoff",         required=True,  help="YYYY-MM-DD upper bound for train rows")
    p.add_argument("--output-dir",           default="artifacts")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    args = _build_parser().parse_args()
    try:
        run_diagnostics(args)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
