"""
analyze_risk_tier_pd_resolution.py
===================================
Answers the question raised against the persona-vs-risk_tier finding: is
risk_tier's modal-tier-per-persona summary (e.g. "C4 and C6 are both
tier_2") hiding real within-tier heterogeneity that the business's 4-bucket
tiering discards, or is it a fair summary? A single mode per persona can't
answer this -- it compresses exactly the information this script restores.

Three things this produces, none of them inferred from aggregates alone:

1. persona_x_risk_tier_crosstab.csv -- the FULL persona x risk_tier
   distribution (counts + %), not just the mode. Confirms or corrects any
   claim of the shape "X% of persona P individually scores into tier Y" --
   that claim needs this table, not a mode + a percentile guess.

2. score_source_coverage_by_persona.csv -- persona x score_source
   (pd_model vs 7_signal_fallback, per run_credit_risk_pipeline.py's own
   column). cal_pd's mean/median/std in the outcome_summary sheet silently
   skip NaN rows (pandas default) -- if some borrowers in a persona were
   scored via the 7-signal fallback (no cal_pd at all, risk_tier derived
   from a DIFFERENT risk_score formula), the cal_pd distribution reported
   for that persona covers only the pd_model-scored subset, not everyone
   risk_tier_mode was computed over. This table makes that coverage gap
   visible instead of assuming 100%.

3. pd_band_breakdown_within_<tier>.csv (default tier_2) and
   pd_band_breakdown_full_population.csv -- bins cal_pd into fixed-width
   bands (default 5 points) and reports, per band: N, % of the
   tier/population, and (if --forward-outcomes-file is given) the real
   forward bad rate and severity (median worst-days-aging) in that band,
   plus which personas make it up. This is the "does realized bad rate
   increase monotonically with PD" check -- the evidence a tier-boundary
   redesign should be based on, not round numbers.

Read-only: does not touch the clustering, the engine run, or any existing
output file.

Usage:
    python scripts\\analyze_risk_tier_pd_resolution.py
    python scripts\\analyze_risk_tier_pd_resolution.py --forward-outcomes-file data\\persona_k8_forward_outcomes.csv
    python scripts\\analyze_risk_tier_pd_resolution.py --tier-to-examine tier_3 --pd-band-width 0.025
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from segmentation.borrower_persona_clustering import digits  # noqa: E402
from scripts.profile_persona_k8 import PERSONA_NAMES  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ASSIGNMENTS_PATH = REPO / "segmentation_outputs" / "persona_k8_profile" / "k8_cluster_assignments.csv"
ENGINE_OUTPUT_PATH = REPO / "output" / "engine_test_output.csv"
OUT_DIR = REPO / "segmentation_outputs" / "persona_k8_profile"


def _pd_band(cal_pd: pd.Series, width: float) -> pd.Series:
    """Fixed-width [lo, hi) bands over [0, 1], labeled "lo-hi" in band order."""
    n_bands = int(np.ceil(1.0 / width))
    edges = [round(i * width, 4) for i in range(n_bands + 1)]
    edges[-1] = 1.0001  # make the top edge inclusive of cal_pd == 1.0
    labels = [f"{edges[i]:.3f}-{edges[i + 1]:.3f}" for i in range(n_bands)]
    return pd.cut(cal_pd, bins=edges, labels=labels, right=False, include_lowest=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assignments-file", type=Path, default=ASSIGNMENTS_PATH, help=f"default: {ASSIGNMENTS_PATH}")
    p.add_argument("--engine-output-file", type=Path, default=ENGINE_OUTPUT_PATH, help=f"default: {ENGINE_OUTPUT_PATH}")
    p.add_argument("--forward-outcomes-file", type=Path, default=None,
                   help="CSV exported from data/persona_k8_forward_outcomes_query.sql -- optional; "
                        "without it, the PD-band tables omit forward bad-rate/severity columns.")
    p.add_argument("--tier-to-examine", default="tier_2", help="default: tier_2")
    p.add_argument("--pd-band-width", type=float, default=0.05, help="default: 0.05 (5 percentage points)")
    args = p.parse_args(argv)

    print("=== Load persona assignments ===")
    assignments = pd.read_csv(args.assignments_file)
    assignments["_id"] = digits(assignments["phonenumber"])
    if "persona_name" not in assignments.columns:
        assignments["persona_name"] = assignments["persona_cluster"].map(PERSONA_NAMES)
    print(f"  {len(assignments):,} borrowers with a persona assignment")

    print("\n=== Load engine output (cal_pd, risk_tier, score_source) ===")
    if not args.engine_output_file.exists():
        print(f"  ERROR: {args.engine_output_file} not found -- nothing to analyze.")
        return
    wanted = ["msisdn", "cal_pd", "risk_tier", "score_source"]
    available = set(pd.read_csv(args.engine_output_file, nrows=0).columns)
    missing = [c for c in wanted if c not in available]
    if missing:
        print(f"  WARNING: {args.engine_output_file} is missing {missing} -- continuing without them.")
    eng = pd.read_csv(args.engine_output_file, usecols=[c for c in wanted if c in available])
    eng["_id"] = digits(eng["msisdn"])
    eng = eng.drop(columns=["msisdn"])
    n_eng_dupes = eng["_id"].dropna().shape[0] - eng["_id"].nunique(dropna=True)
    if n_eng_dupes:
        print(f"  WARNING: {n_eng_dupes:,} duplicate ids in engine output -- keeping first occurrence.")
        eng = eng.drop_duplicates(subset=["_id"], keep="first")

    merged = assignments.merge(eng, on="_id", how="left", validate="one_to_one")
    n_no_engine_row = merged["risk_tier"].isna().sum() if "risk_tier" in merged.columns else len(merged)
    if n_no_engine_row:
        print(f"  WARNING: {n_no_engine_row:,} / {len(merged):,} assigned borrowers have NO matching "
              f"engine-output row -- excluded from the tables below, not silently zero-filled.")

    if args.forward_outcomes_file is not None:
        print("\n=== Load forward-window outcomes ===")
        fwd = pd.read_csv(args.forward_outcomes_file)
        fwd["_id"] = digits(fwd["customer_msisdn"])
        fwd_cols = [c for c in ["fwd_any_bad_3dpd", "fwd_worst_days_aging"] if c in fwd.columns]
        if not fwd_cols:
            print("  WARNING: neither fwd_any_bad_3dpd nor fwd_worst_days_aging found -- "
                  "PD-band tables will have no forward columns.")
        merged = merged.merge(fwd[["_id"] + fwd_cols], on="_id", how="left")
        merged["_had_fwd_activity"] = merged["fwd_any_bad_3dpd"].notna() if "fwd_any_bad_3dpd" in merged.columns else False
    else:
        print("\n=== No --forward-outcomes-file given -- PD-band tables will omit forward columns ===")

    # -- Table 1: full persona x risk_tier crosstab (not just the mode) ----
    print("\n=== Table 1: persona x risk_tier (full distribution) ===")
    cnt = pd.crosstab(merged["persona_name"], merged["risk_tier"])
    pct = pd.crosstab(merged["persona_name"], merged["risk_tier"], normalize="index").round(4) * 100
    pct.columns = [f"{c}_pct" for c in pct.columns]
    crosstab = pd.concat([cnt, pct], axis=1)
    out1 = OUT_DIR / "persona_x_risk_tier_crosstab.csv"
    crosstab.to_csv(out1)
    print(f"  wrote {out1}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(crosstab.to_string())

    # -- Table 2: score_source coverage per persona ------------------------
    if "score_source" in merged.columns:
        print("\n=== Table 2: score_source coverage by persona (resolves the cal_pd NaN-skip gap) ===")
        cov = pd.crosstab(merged["persona_name"], merged["score_source"], dropna=False)
        cov_pct = pd.crosstab(merged["persona_name"], merged["score_source"], normalize="index", dropna=False).round(4) * 100
        cov_pct.columns = [f"{c}_pct" for c in cov_pct.columns]
        cov_tbl = pd.concat([cov, cov_pct], axis=1)
        out2 = OUT_DIR / "score_source_coverage_by_persona.csv"
        cov_tbl.to_csv(out2)
        print(f"  wrote {out2}")
        print(cov_tbl.to_string())
        if "pd_model" in cov.columns and (cov.drop(columns=["pd_model"], errors="ignore").sum(axis=1) > 0).any():
            print("  NOTE: personas with nonzero fallback counts above have cal_pd stats (mean/median/std) "
                  "computed over the pd_model-scored subset ONLY -- not the full persona.")
    else:
        print("\n=== Table 2 skipped: score_source not in engine output ===")

    # -- Table 3: fine PD-band breakdown ------------------------------------
    merged["_pd_band"] = _pd_band(merged["cal_pd"], args.pd_band_width)

    def band_breakdown(df: pd.DataFrame, label: str, out_name: str) -> None:
        print(f"\n=== Table 3: PD-band breakdown -- {label} ===")
        rows = []
        for band, idx in df.groupby("_pd_band", observed=True).groups.items():
            sub = df.loc[idx]
            row = {"pd_band": band, "n": len(sub), "pct_of_group": round(len(sub) / len(df) * 100, 2)}
            if "fwd_any_bad_3dpd" in sub.columns:
                row["fwd_bad_rate_pct"] = round(sub["fwd_any_bad_3dpd"].mean() * 100, 2) if sub["fwd_any_bad_3dpd"].notna().any() else None
            if "fwd_worst_days_aging" in sub.columns:
                row["fwd_worst_days_aging_median"] = sub["fwd_worst_days_aging"].median()
            top_personas = sub["persona_name"].value_counts(normalize=True).head(3)
            row["top_personas"] = "; ".join(f"{name} ({pct:.0%})" for name, pct in top_personas.items())
            rows.append(row)
        band_tbl = pd.DataFrame(rows).sort_values("pd_band")
        out_path = OUT_DIR / out_name
        band_tbl.to_csv(out_path, index=False)
        print(f"  wrote {out_path}")
        with pd.option_context("display.max_colwidth", 60, "display.width", 200):
            print(band_tbl.to_string(index=False))

    tier_subset = merged[merged["risk_tier"] == args.tier_to_examine]
    if len(tier_subset):
        band_breakdown(tier_subset, f"within {args.tier_to_examine} (n={len(tier_subset):,})",
                        f"pd_band_breakdown_within_{args.tier_to_examine}.csv")
    else:
        print(f"\n  WARNING: no borrowers with risk_tier == {args.tier_to_examine!r} -- skipping that table.")

    band_breakdown(merged, f"full population (n={len(merged):,})", "pd_band_breakdown_full_population.csv")

    print(
        "\nReading this: Table 3's fwd_bad_rate_pct column, read band-by-band, is the monotonicity check -- "
        "if it rises steadily with the PD band, the continuous cal_pd score already carries real "
        "information the 4-tier bucketing discards, and tier boundaries should be set where that rate "
        "changes materially, not at round PD numbers. top_personas shows which personas actually make up "
        "each band, rather than assuming a persona's modal tier describes all of it."
    )


if __name__ == "__main__":
    main()
