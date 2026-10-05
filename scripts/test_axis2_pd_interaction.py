"""
test_axis2_pd_interaction.py
==============================
The one focused experiment to decide whether Axis 2 (severity) earns a
place in the credit engine, or stays as portfolio-monitoring/early-warning
information instead of an origination limit adjustment.

Target: SEVERE TAIL, not raw days-aging. Conditional on distress
(fwd_any_bad_3dpd == 1), define

    severe_tail = 1 if fwd_worst_days_aging >= --severe-tail-threshold (default 30) else 0

per the stated reasoning: the operational question is "does distress
persist," not "27 vs. 29 days."

Two things, both out-of-sample (train/test split, stratified on the
target):

1. Nested logistic models, PD only -> PD+volatility -> PD+volatility+trend
   -> PD+volatility+trend+interactions (cal_pd x volatility, cal_pd x
   trend), compared on test-set ROC AUC and lift@20% (share of actual
   severe-tail cases captured in the riskiest 20% of predictions). AUC
   deltas vs. the PD-only baseline are the headline number: if the richer
   models don't beat PD alone by a commercially meaningful margin, Axis 2
   does not deserve an origination-limit role on this evidence, whatever
   the quartile tables suggested.

2. Two non-parametric matrices (PD tercile x volatility tercile, and PD
   tercile x trend tercile), each cell = actual severe-tail rate, on the
   FULL distress subset (not the test split -- these are descriptive, not
   a held-out claim). This is the "before trusting a regression
   interaction" check: do the matrices show real cross-cell escalation
   (e.g. high-PD + high-volatility notably worse than either alone), or
   is the apparent interaction in the logistic model just a modeling
   artifact?

Standardizes cal_pd/volatility/trend (fit on train only) before fitting,
so interaction-term coefficients are interpretable on a common scale and
regularization is well-behaved. No statsmodels in this environment --
coefficients are reported for direction/magnitude only, not inferential
p-values; the AUC/lift comparison is the real evidence here, not
statistical significance of any one coefficient.

Usage:
    python scripts\\test_axis2_pd_interaction.py --forward-outcomes-file data\\persona_k8_forward_outcomes.csv
    python scripts\\test_axis2_pd_interaction.py --forward-outcomes-file data\\persona_k8_forward_outcomes.csv --severe-tail-threshold 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from segmentation.borrower_persona_clustering import digits  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BORROWER_HISTORY_PATH = REPO / "borrower_history_retail_filtered.csv"
ENGINE_OUTPUT_PATH = REPO / "output" / "engine_test_output.csv"
OUT_DIR = REPO / "segmentation_outputs" / "persona_k8_profile"


def _lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.2) -> float:
    """Share of actual positives captured in the top-k fraction of predicted risk."""
    n_top = max(1, int(len(y_score) * k))
    order = np.argsort(-y_score)
    top_idx = order[:n_top]
    total_pos = y_true.sum()
    return float(y_true[top_idx].sum() / total_pos) if total_pos > 0 else float("nan")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--borrower-history-file", type=Path, default=BORROWER_HISTORY_PATH, help=f"default: {BORROWER_HISTORY_PATH}")
    p.add_argument("--engine-output-file", type=Path, default=ENGINE_OUTPUT_PATH, help=f"default: {ENGINE_OUTPUT_PATH}")
    p.add_argument("--forward-outcomes-file", type=Path, required=True,
                   help="CSV exported from data/persona_k8_forward_outcomes_query.sql")
    p.add_argument("--severe-tail-threshold", type=float, default=30.0, help="days aging; default: 30")
    p.add_argument("--test-size", type=float, default=0.3, help="default: 0.3")
    p.add_argument("--random-state", type=int, default=42, help="default: 42")
    args = p.parse_args(argv)

    print("=== Load + merge (borrower_history severity features, cal_pd, forward outcomes) ===")
    bh = pd.read_csv(args.borrower_history_file,
                      usecols=["phonenumber", "lifetime_cure_time_volatility", "cure_time_trend"])
    bh["_id"] = digits(bh["phonenumber"])
    bh = bh.drop(columns=["phonenumber"])

    eng = pd.read_csv(args.engine_output_file, usecols=lambda c: c in {"msisdn", "cal_pd"})
    eng["_id"] = digits(eng["msisdn"])
    eng = eng.drop(columns=["msisdn"]).dropna(subset=["cal_pd"])

    fwd = pd.read_csv(args.forward_outcomes_file)
    fwd["_id"] = digits(fwd["customer_msisdn"])
    needed = {"fwd_any_bad_3dpd", "fwd_worst_days_aging"}
    if not needed <= set(fwd.columns):
        print(f"  ERROR: forward-outcomes file is missing {needed - set(fwd.columns)} -- cannot run this test.")
        return

    merged = bh.merge(eng, on="_id", how="inner").merge(fwd[["_id"] + list(needed)], on="_id", how="inner")
    distress = merged[merged["fwd_any_bad_3dpd"] == 1].copy()
    distress["severe_tail"] = (distress["fwd_worst_days_aging"] >= args.severe_tail_threshold).astype(int)
    distress = distress.dropna(subset=["cal_pd", "lifetime_cure_time_volatility", "cure_time_trend"])
    n = len(distress)
    sev_rate = distress["severe_tail"].mean()
    print(f"  {n:,} distressed borrowers with complete features "
          f"({sev_rate:.1%} reach >= {args.severe_tail_threshold:.0f} days aging -- the severe-tail base rate)")
    if n < 200:
        print("  WARNING: small sample for a train/test split -- treat AUC deltas as directional only.")

    # -- Part 1: nested logistic models, out-of-sample -------------------
    print("\n=== Part 1: nested models, out-of-sample AUC + lift@20% ===")
    X_cols = ["cal_pd", "lifetime_cure_time_volatility", "cure_time_trend"]
    train, test = train_test_split(distress, test_size=args.test_size, stratify=distress["severe_tail"],
                                    random_state=args.random_state)
    scaler = StandardScaler().fit(train[X_cols])
    train_s = pd.DataFrame(scaler.transform(train[X_cols]), columns=X_cols, index=train.index)
    test_s = pd.DataFrame(scaler.transform(test[X_cols]), columns=X_cols, index=test.index)
    for df in (train_s, test_s):
        df["pd_x_vol"] = df["cal_pd"] * df["lifetime_cure_time_volatility"]
        df["pd_x_trend"] = df["cal_pd"] * df["cure_time_trend"]

    model_specs = {
        "M1_pd_only": ["cal_pd"],
        "M2_pd_vol": ["cal_pd", "lifetime_cure_time_volatility"],
        "M3_pd_vol_trend": ["cal_pd", "lifetime_cure_time_volatility", "cure_time_trend"],
        "M4_pd_vol_trend_interactions": ["cal_pd", "lifetime_cure_time_volatility", "cure_time_trend",
                                          "pd_x_vol", "pd_x_trend"],
    }
    y_train, y_test = train["severe_tail"].to_numpy(), test["severe_tail"].to_numpy()
    results = []
    baseline_auc = None
    for name, cols in model_specs.items():
        clf = LogisticRegression(max_iter=1000)
        clf.fit(train_s[cols], y_train)
        pred = clf.predict_proba(test_s[cols])[:, 1]
        auc = roc_auc_score(y_test, pred)
        lift20 = _lift_at_k(y_test, pred, 0.2)
        if baseline_auc is None:
            baseline_auc = auc
        results.append({
            "model": name, "auc": round(auc, 4), "delta_auc_vs_pd_only": round(auc - baseline_auc, 4),
            "lift_at_20pct": round(lift20, 4),
            "coefficients": dict(zip(cols, np.round(clf.coef_[0], 4))),
        })
    res_df = pd.DataFrame(results)
    out1 = OUT_DIR / "axis2_interaction_model_comparison.csv"
    res_df.drop(columns=["coefficients"]).to_csv(out1, index=False)
    print(f"  wrote {out1}")
    with pd.option_context("display.max_colwidth", 120):
        print(res_df.drop(columns=["coefficients"]).to_string(index=False))
    print("\n  Standardized coefficients (direction/magnitude only, no p-values -- statsmodels unavailable here):")
    for r in results:
        print(f"    {r['model']}: {r['coefficients']}")

    print(
        "\n  Reading this: delta_auc_vs_pd_only is the headline number. A small (e.g. <0.01-0.02) or "
        "inconsistent delta across M2-M4 means Axis 2 is not adding commercially meaningful discrimination "
        "over PD alone on this evidence -- treat severity as portfolio-monitoring/early-warning information, "
        "not an origination-limit input. A clear, growing delta from M2 through M4, with lift_at_20pct also "
        "rising, is the evidence an interaction-based overlay actually earns its place."
    )

    # -- Part 2: non-parametric interaction matrices ----------------------
    print("\n=== Part 2: non-parametric severe-tail-rate matrices (full distress subset, descriptive) ===")

    def tercile_matrix(row_col: str, col_col: str, row_label: str, col_label: str, out_name: str) -> None:
        d = distress[[row_col, col_col, "severe_tail"]].dropna()
        d["_row"] = pd.qcut(d[row_col], 3, labels=["Low", "Medium", "High"], duplicates="drop")
        d["_col"] = pd.qcut(d[col_col], 3, labels=["Low", "Medium", "High"], duplicates="drop")
        rate = d.groupby(["_row", "_col"], observed=True)["severe_tail"].mean().unstack() * 100
        cnt = d.groupby(["_row", "_col"], observed=True)["severe_tail"].count().unstack()
        print(f"\n  -- {row_label} (rows) x {col_label} (cols) -- severe-tail rate % --")
        print(rate.round(2).to_string())
        print(f"  -- same cells, n --")
        print(cnt.to_string())
        out_path = OUT_DIR / out_name
        rate.round(2).to_csv(out_path)
        print(f"  wrote {out_path}")

    tercile_matrix("cal_pd", "lifetime_cure_time_volatility", "PD tercile", "Volatility tercile",
                    "axis2_matrix_pd_x_volatility.csv")
    tercile_matrix("cal_pd", "cure_time_trend", "PD tercile", "Trend tercile",
                    "axis2_matrix_pd_x_trend.csv")

    print(
        "\nReading Part 2: look for the high-PD/high-volatility (or high-PD/high-trend) cell being "
        "MATERIALLY worse than either the high-PD/low-X or low-PD/high-X cells alone -- that's the "
        "escalation pattern that would justify an interaction term. If the high-PD row is uniformly bad "
        "regardless of the column, or the pattern isn't monotonic, the apparent interaction in Part 1's "
        "logistic model is more likely a modeling artifact than a real escalation effect."
    )


if __name__ == "__main__":
    main()
