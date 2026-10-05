"""
analyze_axis2_severity_signal.py
==================================
Axis 2 (severity/persistence) screening, per the agreed next step: before
combining any historical features into a severity score, test each
candidate INDIVIDUALLY against two different questions, because conflating
them turns Axis 2 into a second PD model instead of an orthogonal signal:

  Axis 1 question (frequency):   does this feature predict WHETHER
                                  distress happens at all?
  Axis 2 question (severity):    GIVEN distress happens, does this feature
                                  predict HOW SEVERE/PERSISTENT it is?

A feature that only answers the first question belongs in PD, not the
severity overlay. A feature that answers the second belongs in Axis 2 --
but only if it still separates severity AFTER conditioning on calibrated
PD (the incremental-information test), since otherwise it's just a proxy
for PD with the same information, not something orthogonal to it.

Three things this produces, per candidate feature (default: the four
point-in-time-safe candidates from borrower_history_retail_filtered.csv --
lifetime_cure_time_volatility, cure_time_trend, recent_5_default_24h_rate,
lifetime_avg_hours_to_principal_cure):

1. screening_summary.csv -- one row per feature: Spearman correlation with
   future bad frequency (whole population) vs. Spearman correlation with
   future severity CONDITIONAL on distress (fwd_any_bad_3dpd == 1 only).
   This is the "future bad frequency | future severity conditional on bad"
   table from the brief, with real numbers instead of blank cells.

2. severity_by_feature_quartile.csv -- among distressed borrowers only,
   median/n of future worst-days-aging by feature quartile. Simple
   monotonicity check before the harder incremental test.

3. incremental_beyond_pd__<feature>.csv (one per feature) -- among
   distressed borrowers, cal_pd quintile x {low half, high half of this
   feature, split WITHIN each quintile} -> median conditional severity +
   n per cell. This is the actual orthogonality test: does the feature
   still separate severity for borrowers who already look similar on PD?

Deliberately does NOT combine the four candidates into a single severity
score -- that's a later step, only for whichever features survive this
screen with real incremental signal.

Usage:
    python scripts\\analyze_axis2_severity_signal.py --forward-outcomes-file data\\persona_k8_forward_outcomes.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from segmentation.borrower_persona_clustering import digits  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BORROWER_HISTORY_PATH = REPO / "borrower_history_retail_filtered.csv"
ENGINE_OUTPUT_PATH = REPO / "output" / "engine_test_output.csv"
OUT_DIR = REPO / "segmentation_outputs" / "persona_k8_profile"

DEFAULT_CANDIDATES = [
    "lifetime_cure_time_volatility",
    "cure_time_trend",
    "recent_5_default_24h_rate",
    "lifetime_avg_hours_to_principal_cure",
]


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 20:
        return float("nan"), n
    rho, _ = spearmanr(x[mask], y[mask])
    return rho, n


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--borrower-history-file", type=Path, default=BORROWER_HISTORY_PATH, help=f"default: {BORROWER_HISTORY_PATH}")
    p.add_argument("--engine-output-file", type=Path, default=ENGINE_OUTPUT_PATH, help=f"default: {ENGINE_OUTPUT_PATH}")
    p.add_argument("--forward-outcomes-file", type=Path, required=True,
                   help="CSV exported from data/persona_k8_forward_outcomes_query.sql")
    p.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES, help=f"default: {DEFAULT_CANDIDATES}")
    p.add_argument("--pd-quintiles", type=int, default=5, help="default: 5")
    args = p.parse_args(argv)

    print("=== Load borrower_history (severity candidate features) ===")
    available = set(pd.read_csv(args.borrower_history_file, nrows=0).columns)
    cand_present = [c for c in args.candidates if c in available]
    cand_missing = [c for c in args.candidates if c not in available]
    if cand_missing:
        print(f"  WARNING: {cand_missing} not found in {args.borrower_history_file} -- skipping those.")
    bh = pd.read_csv(args.borrower_history_file, usecols=["phonenumber"] + cand_present)
    bh["_id"] = digits(bh["phonenumber"])
    bh = bh.drop(columns=["phonenumber"])
    print(f"  {len(bh):,} borrowers, candidates present: {cand_present}")

    print("\n=== Load engine output (cal_pd) ===")
    eng = pd.read_csv(args.engine_output_file, usecols=lambda c: c in {"msisdn", "cal_pd"})
    eng["_id"] = digits(eng["msisdn"])
    eng = eng.drop(columns=["msisdn"]).dropna(subset=["cal_pd"])

    print("\n=== Load forward outcomes (fwd_any_bad_3dpd, fwd_worst_days_aging) ===")
    fwd = pd.read_csv(args.forward_outcomes_file)
    fwd["_id"] = digits(fwd["customer_msisdn"])
    fwd_cols = [c for c in ["fwd_any_bad_3dpd", "fwd_worst_days_aging"] if c in fwd.columns]
    missing_fwd = {"fwd_any_bad_3dpd", "fwd_worst_days_aging"} - set(fwd_cols)
    if missing_fwd:
        print(f"  ERROR: forward-outcomes file is missing {missing_fwd} -- cannot run this analysis.")
        return

    merged = bh.merge(eng, on="_id", how="inner").merge(fwd[["_id"] + fwd_cols], on="_id", how="inner")
    print(f"\n  {len(merged):,} borrowers with candidate features + cal_pd + forward outcomes")

    distress = merged[merged["fwd_any_bad_3dpd"] == 1].copy()
    print(f"  {len(distress):,} ({len(distress) / len(merged):.1%}) are the distress subset "
          f"(fwd_any_bad_3dpd == 1) -- severity questions below use ONLY this subset")

    # -- Table 1: screening summary --------------------------------------
    print("\n=== Table 1: screening summary (frequency vs. severity-conditional-on-distress) ===")
    rows = []
    for feat in cand_present:
        freq_rho, freq_n = _spearman(merged[feat], merged["fwd_any_bad_3dpd"])
        sev_rho, sev_n = _spearman(distress[feat], distress["fwd_worst_days_aging"])
        rows.append({
            "feature": feat,
            "frequency_corr_spearman": round(freq_rho, 4) if not np.isnan(freq_rho) else None,
            "frequency_n": freq_n,
            "severity_conditional_corr_spearman": round(sev_rho, 4) if not np.isnan(sev_rho) else None,
            "severity_conditional_n": sev_n,
        })
    summary = pd.DataFrame(rows)
    out1 = OUT_DIR / "axis2_screening_summary.csv"
    summary.to_csv(out1, index=False)
    print(f"  wrote {out1}")
    print(summary.to_string(index=False))
    print(
        "\n  Reading this: a feature with |frequency_corr| notably larger than |severity_conditional_corr| "
        "is mostly a frequency/PD signal, not a severity one. A feature with real "
        "|severity_conditional_corr| (and a frequency_corr near zero) is the kind of orthogonal signal "
        "Axis 2 needs -- but still needs the incremental-beyond-PD test below before trusting it."
    )

    # -- Table 2: severity by feature quartile, among distressed only ----
    print("\n=== Table 2: future severity (conditional on distress) by feature quartile ===")
    q2_rows = []
    for feat in cand_present:
        sub = distress[[feat, "fwd_worst_days_aging"]].dropna()
        if len(sub) < 20:
            continue
        try:
            sub["_q"] = pd.qcut(sub[feat], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"], duplicates="drop")
        except ValueError:
            print(f"  NOTE: {feat} has too few distinct values among the distress subset for quartiles -- skipping.")
            continue
        for q, idx in sub.groupby("_q", observed=True).groups.items():
            s = sub.loc[idx]
            q2_rows.append({
                "feature": feat, "quartile": q, "n": len(s),
                "median_fwd_worst_days_aging": s["fwd_worst_days_aging"].median(),
            })
    q2 = pd.DataFrame(q2_rows)
    out2 = OUT_DIR / "axis2_severity_by_feature_quartile.csv"
    q2.to_csv(out2, index=False)
    print(f"  wrote {out2}")
    print(q2.to_string(index=False))

    # -- Table 3: incremental beyond PD, one per feature ------------------
    print("\n=== Table 3: incremental information beyond calibrated PD (per feature) ===")
    distress["_pd_q"] = pd.qcut(distress["cal_pd"], args.pd_quintiles, duplicates="drop")
    for feat in cand_present:
        rows3 = []
        for pd_band, idx in distress.groupby("_pd_q", observed=True).groups.items():
            band = distress.loc[idx, [feat, "fwd_worst_days_aging"]].dropna()
            if len(band) < 20:
                continue
            median_feat = band[feat].median()
            low = band[band[feat] <= median_feat]
            high = band[band[feat] > median_feat]
            rows3.append({
                "cal_pd_band": str(pd_band),
                "n_low": len(low), "median_severity_low": low["fwd_worst_days_aging"].median(),
                "n_high": len(high), "median_severity_high": high["fwd_worst_days_aging"].median(),
                "high_minus_low": (
                    round(high["fwd_worst_days_aging"].median() - low["fwd_worst_days_aging"].median(), 2)
                    if len(low) and len(high) else None
                ),
            })
        tbl3 = pd.DataFrame(rows3)
        out3 = OUT_DIR / f"axis2_incremental_beyond_pd__{feat}.csv"
        tbl3.to_csv(out3, index=False)
        print(f"\n  -- {feat} -- wrote {out3}")
        print(tbl3.to_string(index=False))

    print(
        "\nReading Table 3: 'high_minus_low' consistently positive (or consistently negative) across most "
        "PD bands, with a meaningful magnitude, is the evidence this feature carries real information "
        "BEYOND calibrated PD -- borrowers who already look similar on PD still separate on future "
        "severity by this feature. Inconsistent sign, or a magnitude near zero, means this feature isn't "
        "adding much once PD is already known -- fold it back toward Axis 1 (or drop it) rather than "
        "including it in the severity overlay."
    )


if __name__ == "__main__":
    main()
