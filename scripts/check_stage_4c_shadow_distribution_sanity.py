"""
check_stage_4c_shadow_distribution_sanity.py
==============================================
Read-only distribution sanity check on a real Stage 4C Decision Log, run
ONCE against the first real controlled shadow run before freezing it as
the integration baseline. Never tunes k or any C3 parameter -- it only
reports whether the real, already-fixed values produced technically
valid but economically absurd output (row-count/invariant checks in
stage_4c_decision_logger.py cannot catch this class of problem).

Reports, broken out by (risk_tier x capacity_estimable):
  - l_current, capacity_shadow_k0_05, shadow_multiplier_base,
    l_full_shadow_k0_05_base: full distribution (mean/std/min/percentiles/max).
  - disagreement_abs_ugx_shadow_base, disagreement_ratio_shadow_base:
    same distribution, PLUS a direction breakdown (pct where the
    challenger would assign MORE than the live limit, LESS, or the same).
    The Decision Log itself deliberately carries no direction
    classification (frozen schema Section 4.6) -- that classification
    belongs here, in a later analytical layer, never in the logger
    module itself.

risk_tier is NOT a Decision Log column (by frozen schema design) -- it
is read from the engine's own --keep-intermediate output CSV and joined
in via agent_msisdn/msisdn, for this analysis only.

Also prints an ANOMALIES section: rows where shadow_multiplier_base
falls outside the real policy's [m_min, m_max] = [0.40, 1.00] bounds
(SHADOW_SCENARIOS in extrafloat_shadow_risk_multiplier.py), negative
l_current/capacity_shadow_k0_05/l_full_shadow_k0_05_base (should never
occur), and an extreme-disagreement-ratio count (ratio outside
[0.1, 10.0] -- flagged for visibility only, never auto-excluded).

Usage:
    python scripts\\check_stage_4c_shadow_distribution_sanity.py \\
        --decision-log-file output\\stage_4c_shadow\\stage_4c_decision_log_<run_id>.csv \\
        --engine-output-file output\\engine_test_output_stage4c_shadow.csv \\
        --out-prefix output\\stage_4c_shadow\\sanity_check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Real policy bounds -- restated (not imported) per the one-way scripts/
# layering convention. Any shadow_multiplier_base/conservative outside
# this range on a loaded, self-consistent artifact would indicate a bug
# in _policy_3_hybrid or SHADOW_SCENARIOS, not real data variation.
SHADOW_M_MIN = 0.40
SHADOW_M_MAX = 1.00
_BOUNDS_TOL = 1e-9

# Disagreement ratios outside this band are flagged for visual review
# only -- never excluded or treated as errors. A genuinely large
# disagreement (e.g. challenger would assign 20x the live limit) is
# exactly the kind of thing this sanity check exists to surface.
EXTREME_RATIO_LOW = 0.1
EXTREME_RATIO_HIGH = 10.0

DIST_COLS = [
    "l_current",
    "capacity_shadow_k0_05",
    "shadow_multiplier_base",
    "l_full_shadow_k0_05_base",
]
DISAGREEMENT_COLS = [
    "disagreement_abs_ugx_shadow_base",
    "disagreement_ratio_shadow_base",
]

PERCENTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


def _norm_msisdn_for_join(s: pd.Series) -> pd.Series:
    """Minimal digit-string normalization for joining the Decision Log's
    agent_msisdn against the engine output's msisdn -- restated (not
    imported) per the one-way scripts/ layering convention; mirrors
    extrafloat.shadow_logging.stage_4c_decision_logger._norm_msisdn's
    intent without importing a private helper from a non-scripts module."""
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.mask(out.str.lower().isin({"", "nan", "none", "<na>"}))


def load_joined(decision_log_file, engine_output_file) -> pd.DataFrame:
    dlog = pd.read_csv(decision_log_file)
    eng = pd.read_csv(engine_output_file, usecols=lambda c: c in {"msisdn", "risk_tier"})
    if "risk_tier" not in eng.columns:
        raise ValueError(
            f"{engine_output_file} has no risk_tier column -- was it written without "
            f"--keep-intermediate, or from a different run than the Decision Log?"
        )
    dlog["agent_msisdn"] = _norm_msisdn_for_join(dlog["agent_msisdn"])
    eng["msisdn"] = _norm_msisdn_for_join(eng["msisdn"])
    eng = eng.drop_duplicates(subset=["msisdn"]).rename(columns={"msisdn": "agent_msisdn"})
    joined = dlog.merge(eng, on="agent_msisdn", how="left")
    unmatched = joined["risk_tier"].isna().sum()
    if unmatched:
        print(
            f"WARNING: {unmatched}/{len(joined)} Decision Log rows did not find a "
            f"risk_tier match in {engine_output_file} -- likely a mismatched run pair "
            f"(different run_id) rather than this script's own join logic. Those rows "
            f"are grouped under risk_tier=NaN below rather than silently dropped."
        )
    return joined


def _distribution_row(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    n = len(s)
    valid = s.dropna()
    row = {"n": n, "n_non_null": len(valid)}
    if valid.empty:
        for stat in ["mean", "std", "min", "max"] + [f"p{int(p * 100)}" for p in PERCENTILES]:
            row[stat] = np.nan
        return row
    row["mean"] = valid.mean()
    row["std"] = valid.std()
    row["min"] = valid.min()
    row["max"] = valid.max()
    qs = valid.quantile(PERCENTILES)
    for p in PERCENTILES:
        row[f"p{int(p * 100)}"] = qs.loc[p]
    return row


def _direction_row(series: pd.Series, zero_tol: float = 1e-6) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n == 0:
        return {"n_non_null": 0, "pct_challenger_higher": np.nan,
                "pct_challenger_lower": np.nan, "pct_equal": np.nan}
    higher = (s > zero_tol).sum()
    lower = (s < -zero_tol).sum()
    equal = n - higher - lower
    return {
        "n_non_null": n,
        "pct_challenger_higher": 100.0 * higher / n,
        "pct_challenger_lower": 100.0 * lower / n,
        "pct_equal": 100.0 * equal / n,
    }


def build_group_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (risk_tier, capacity_estimable, metric) -- the detailed
    audit table. Grouping deliberately includes an explicit 'OVERALL'
    pseudo-group (risk_tier=None) alongside the real risk_tier breakouts,
    so the full-population view and the per-tier views sit side by side
    without requiring two separate tables."""
    rows = []

    def _add_group(label_tier, label_estimable, sub: pd.DataFrame):
        for col in DIST_COLS:
            if col not in sub.columns:
                continue
            d = _distribution_row(sub[col])
            rows.append({"risk_tier": label_tier, "capacity_estimable": label_estimable,
                         "metric": col, **d})
        for col in DISAGREEMENT_COLS:
            if col not in sub.columns:
                continue
            d = _distribution_row(sub[col])
            rows.append({"risk_tier": label_tier, "capacity_estimable": label_estimable,
                         "metric": col, **d})

    _add_group("OVERALL", "OVERALL", df)
    for estimable, sub in df.groupby("capacity_estimable", dropna=False):
        _add_group("OVERALL", str(estimable), sub)
    for tier, sub in df.groupby("risk_tier", dropna=False):
        _add_group(str(tier), "OVERALL", sub)
        for estimable, sub2 in sub.groupby("capacity_estimable", dropna=False):
            _add_group(str(tier), str(estimable), sub2)

    return pd.DataFrame(rows)


def build_direction_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def _add_group(label_tier, label_estimable, sub: pd.DataFrame):
        for col in DISAGREEMENT_COLS:
            if col == "disagreement_ratio_shadow_base":
                # Direction for the ratio column is read off the signed
                # abs column (ratio=1.0 <=> abs=0; the two must agree by
                # construction) -- computing direction straight off a
                # ratio centered at 1 rather than 0 would need its own
                # tolerance semantics, so reuse the abs column's sign.
                continue
            if col not in sub.columns:
                continue
            d = _direction_row(sub[col])
            rows.append({"risk_tier": label_tier, "capacity_estimable": label_estimable, **d})

    _add_group("OVERALL", "OVERALL", df)
    for estimable, sub in df.groupby("capacity_estimable", dropna=False):
        _add_group("OVERALL", str(estimable), sub)
    for tier, sub in df.groupby("risk_tier", dropna=False):
        _add_group(str(tier), "OVERALL", sub)
        for estimable, sub2 in sub.groupby("capacity_estimable", dropna=False):
            _add_group(str(tier), str(estimable), sub2)

    return pd.DataFrame(rows)


def find_anomalies(df: pd.DataFrame) -> dict:
    anomalies = {}

    if "shadow_multiplier_base" in df.columns:
        m = pd.to_numeric(df["shadow_multiplier_base"], errors="coerce")
        out_of_bounds = m.notna() & ((m < SHADOW_M_MIN - _BOUNDS_TOL) | (m > SHADOW_M_MAX + _BOUNDS_TOL))
        anomalies["shadow_multiplier_base_out_of_policy_bounds"] = int(out_of_bounds.sum())

    for col in ("l_current", "capacity_shadow_k0_05", "l_full_shadow_k0_05_base"):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            anomalies[f"{col}_negative"] = int((v < 0).sum())

    if "disagreement_ratio_shadow_base" in df.columns:
        r = pd.to_numeric(df["disagreement_ratio_shadow_base"], errors="coerce")
        extreme = r.notna() & ((r < EXTREME_RATIO_LOW) | (r > EXTREME_RATIO_HIGH))
        anomalies["disagreement_ratio_extreme_count"] = int(extreme.sum())

    if "capacity_estimable" in df.columns and "capacity_shadow_k0_05" in df.columns:
        est_true_null = (df["capacity_estimable"] == True) & df["capacity_shadow_k0_05"].isna()  # noqa: E712
        est_false_nonnull = (df["capacity_estimable"] == False) & df["capacity_shadow_k0_05"].notna()  # noqa: E712
        anomalies["capacity_estimable_true_but_capacity_null"] = int(est_true_null.sum())
        anomalies["capacity_estimable_false_but_capacity_non_null"] = int(est_false_nonnull.sum())

    return anomalies


TOP_TAIL_COLUMNS = [
    "agent_msisdn", "float_activity_value_1m", "commission",
    "capacity_shadow_k0_05", "shadow_multiplier_base", "l_full_shadow_k0_05_base",
    "l_current", "disagreement_ratio_shadow_base", "risk_tier",
]


def top_tail_rows(df: pd.DataFrame, by_col: str, n: int = 10) -> pd.DataFrame:
    """Top n rows by by_col (descending, NaN last), restricted to
    TOP_TAIL_COLUMNS. A deliberately narrow source-traceability check,
    requested before freezing the first real shadow run as the Stage 4C
    integration baseline: are the extreme outputs (the long right tail in
    capacity_shadow_k0_05 and the resulting large disagreement ratios)
    traceable to genuinely large float_activity_value_1m/commission
    inputs, rather than a malformed source value? No new thresholds, no
    additional segmentation, no recalibration -- purely a row dump
    answering that one binary question."""
    cols = [c for c in TOP_TAIL_COLUMNS if c in df.columns]
    return df.sort_values(by_col, ascending=False, na_position="last").head(n)[cols]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decision-log-file", required=True)
    p.add_argument("--engine-output-file", required=True)
    p.add_argument("--out-prefix", default=None, help="If given, writes _distribution.csv/_direction.csv/_anomalies.csv")
    p.add_argument("--top-n-tail", type=int, default=10,
                    help="Print this many top rows by capacity_shadow_k0_05 and by "
                         "disagreement_ratio_shadow_base (source-traceability check).")
    args = p.parse_args(argv)

    print("=" * 80)
    print("Stage 4C -- real shadow run distribution sanity check")
    print("Observational check only: never tunes k or any C3 parameter. A clean")
    print("result here, combined with the already-passed row-count/invariant checks,")
    print("is what justifies freezing this run as the integration baseline.")
    print("=" * 80)

    df = load_joined(args.decision_log_file, args.engine_output_file)
    print(f"\nJoined rows: {len(df)}")

    dist_table = build_group_distribution_table(df)
    dir_table = build_direction_table(df)
    anomalies = find_anomalies(df)

    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    pd.set_option("display.width", 160)

    print("\n" + "=" * 80)
    print("DISTRIBUTION TABLE (OVERALL + by capacity_estimable + by risk_tier)")
    print("=" * 80)
    print(dist_table.to_string(index=False))

    print("\n" + "=" * 80)
    print("DISAGREEMENT DIRECTION TABLE")
    print("(pct_challenger_higher = Decision Log's challenger would assign MORE")
    print(" than the live limit; pct_challenger_lower = LESS; pct_equal = same)")
    print("=" * 80)
    print(dir_table.to_string(index=False))

    print("\n" + "=" * 80)
    print("ANOMALIES (should all read 0 on a clean run)")
    print("=" * 80)
    for k, v in anomalies.items():
        flag = "  <-- NONZERO" if v else ""
        print(f"  {k:55s} {v}{flag}")

    top_by_capacity = top_tail_rows(df, "capacity_shadow_k0_05", args.top_n_tail)
    top_by_ratio = top_tail_rows(df, "disagreement_ratio_shadow_base", args.top_n_tail)

    print("\n" + "=" * 80)
    print(f"TOP {args.top_n_tail} BY capacity_shadow_k0_05 (source-traceability check)")
    print("=" * 80)
    print(top_by_capacity.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"TOP {args.top_n_tail} BY disagreement_ratio_shadow_base (source-traceability check)")
    print("=" * 80)
    print(top_by_ratio.to_string(index=False))

    if args.out_prefix:
        dist_table.to_csv(f"{args.out_prefix}_distribution.csv", index=False)
        dir_table.to_csv(f"{args.out_prefix}_direction.csv", index=False)
        pd.DataFrame([anomalies]).to_csv(f"{args.out_prefix}_anomalies.csv", index=False)
        top_by_capacity.to_csv(f"{args.out_prefix}_top_by_capacity.csv", index=False)
        top_by_ratio.to_csv(f"{args.out_prefix}_top_by_ratio.csv", index=False)
        print(f"\nWrote {args.out_prefix}_distribution.csv / _direction.csv / _anomalies.csv / "
              f"_top_by_capacity.csv / _top_by_ratio.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
