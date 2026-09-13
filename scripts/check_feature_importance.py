"""
Reports gain-based feature importance for the XGBoost/LightGBM models
already saved in --artifacts-dir -- entirely post-hoc, no retraining.
xgb_model.joblib/lgbm_model.joblib already contain everything needed
(model.get_booster().get_score(importance_type="gain") for XGBoost,
model.booster_.feature_importance(importance_type="gain") for LightGBM --
same calls evaluate_xgb()/evaluate_lgbm() make, which pd_model.run_pipeline
never actually calls today, so this is the only way to see importance
without changing the training pipeline).

Also cross-checks feature_order.json's selected_features list: a feature
absent from that list never reached either model at all (dropped by IV
selection or the leakage guard), regardless of how strong its standalone
SQL/univariate signal looked.

Usage:
    python scripts\\check_feature_importance.py --artifacts-dir pd_model\\artifacts
    python scripts\\check_feature_importance.py --artifacts-dir pd_model\\artifacts_ablation ^
        --highlight-features prior_max_loan_seq,prior_avg_repayment_ratio,...
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

_DEFAULT_HIGHLIGHT = (
    "prior_max_loan_seq,prior_max_principal_outstanding_ugx,"
    "prior_max_total_outstanding_ugx,prior_avg_repayment_ratio,"
    "prior_principal_unsettled_count,prior_total_late_fee_owed_ugx,"
    "most_recent_prior_days_past_due_within_30d,prior_loan_has_no_history_flag,"
    "prior_loan_confirmed_no_lateness_flag,prior_loan_censored_flag"
)


def _xgb_importance(model) -> pd.Series:
    booster = model.get_booster()
    imp_gain = booster.get_score(importance_type="gain")
    return pd.Series(imp_gain, dtype=float)


def _lgb_importance(model) -> pd.Series:
    return pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=model.booster_.feature_name(),
        dtype=float,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-dir", default="pd_model/artifacts")
    ap.add_argument("--highlight-features", default=_DEFAULT_HIGHLIGHT,
                     help="Comma-separated feature names to flag in the output "
                          "(default: the 10 recently-added prior-loan columns).")
    ap.add_argument("--out", default=None,
                     help="Optional CSV path for the full ranked table (default: "
                          "<artifacts-dir>/feature_importance_report.csv)")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    xgb_path = artifacts_dir / "xgb_model.joblib"
    lgb_path = artifacts_dir / "lgbm_model.joblib"
    order_path = artifacts_dir / "feature_order.json"

    if not xgb_path.exists():
        sys.exit(f"ERROR: {xgb_path} not found")
    if not lgb_path.exists():
        sys.exit(f"ERROR: {lgb_path} not found")

    selected_features = None
    if order_path.exists():
        selected_features = json.loads(order_path.read_text()).get("selected_features")
        print(f"feature_order.json: {len(selected_features):,} features actually selected for training\n")
    else:
        print(f"NOTE: {order_path} not found -- can't cross-check against the selected-feature list.\n")

    xgb_model = joblib.load(xgb_path)
    lgb_model = joblib.load(lgb_path)

    xgb_imp = _xgb_importance(xgb_model)
    lgb_imp = _lgb_importance(lgb_model)

    all_features = sorted(set(xgb_imp.index) | set(lgb_imp.index))
    tbl = pd.DataFrame({"feature": all_features})
    tbl["xgb_gain"] = tbl["feature"].map(xgb_imp).fillna(0.0)
    tbl["lgb_gain"] = tbl["feature"].map(lgb_imp).fillna(0.0)
    tbl["xgb_rank"] = tbl["xgb_gain"].rank(ascending=False, method="min").astype(int)
    tbl["lgb_rank"] = tbl["lgb_gain"].rank(ascending=False, method="min").astype(int)

    if selected_features is not None:
        selected_set = set(selected_features)
        tbl["in_selected_features"] = tbl["feature"].isin(selected_set)

    highlight = {f.strip() for f in args.highlight_features.split(",") if f.strip()}
    tbl["highlighted"] = tbl["feature"].isin(highlight)

    tbl = tbl.sort_values("xgb_gain", ascending=False).reset_index(drop=True)

    print("=" * 78)
    print(f"Full importance table ({len(tbl):,} features seen by at least one model)")
    print("=" * 78)
    print(tbl.to_string(index=False))

    if highlight:
        print("\n" + "=" * 78)
        print(f"Highlighted features ({len(highlight)} requested)")
        print("=" * 78)
        hl_tbl = tbl[tbl["highlighted"]]
        missing_from_models = highlight - set(tbl["feature"])
        print(hl_tbl.drop(columns=["highlighted"]).to_string(index=False) if len(hl_tbl) else "(none matched)")
        if missing_from_models:
            reason = (
                "not in feature_order.json's selected_features (dropped before training)"
                if selected_features is not None
                else "not found in either model's importance table"
            )
            print(f"\n{len(missing_from_models)} highlighted feature(s) never reached either model "
                  f"({reason}): {sorted(missing_from_models)}")

    out_path = Path(args.out) if args.out else artifacts_dir / "feature_importance_report.csv"
    tbl.to_csv(out_path, index=False)
    print(f"\nFull table written to: {out_path}")


if __name__ == "__main__":
    main()
