"""
compute_pd_comparable_capacity_frontier.py
==================================================
Stage 4B/Analysis-4 bridge. Re-examines Analysis 4 Deliverable 1's
already-derived, already-reviewed historically-supported exposure
frontier (scripts/derive_capacity_frontier_from_business_state.py)
through the lens of Stage 4B-R3's reconstructed retrospective_current_
cal_pd, to test -- as PART of this stage, not a separate H1/H2
project -- whether a historical frontier breach reflects a genuine
exposure boundary or risk selection into exposure tier.

TERMINOLOGY (load-bearing, per explicit review): this produces a
PD-COMPARABLE HISTORICAL EXPOSURE FRONTIER, never a "risk-conditioned
frontier" -- matching on one reconstructed PD dimension reduces
observed risk-selection differences, it does not fully control for all
confounding. The drop-in output filenames may say "pd_comparable" for
brevity; the claim itself is always stated this way.

REPRODUCTION GATE, THE MOST IMPORTANT STRUCTURAL CHECK HERE: this
script restates (one-way scripts/ layering convention) the ENTIRE
agent-period/band/pairing pipeline from derive_capacity_frontier_from_
business_state.py, extended to carry PD. Before computing a single
PD-conditioned number, it re-derives every (band, tolerance, candidate
tier) cell ALREADY in the real, frozen classification CSV and asserts
its own numbers match EXACTLY (delta_br_loan_weighted_pp, n_units_
outcome_both, classification). Any mismatch is a HARD EXIT -- never a
warning -- because attributing a difference to PD-conditioning when it
actually came from restatement drift would be a real, dangerous
mistake, not a cosmetic one.

REQUIRES the real Deliverable 1 CSVs (--float-frontier-classification-
csv / --commission-frontier-classification-csv) as input. Never
regenerates them -- if they are missing, this script hard-exits and
names derive_capacity_frontier_from_business_state.py as the separate,
auditable script to run first. Hidden auto-regeneration inside this
script was explicitly rejected.

PD JOIN AND COVERAGE: retrospective_current_cal_pd (Stage 4B-R3's
output) is left-joined onto the episode dataset by disbursement_fid.
Coverage is NOT 100% -- only thick-file, PIT-reconstructable episodes
have a PD (Stage 4A/4A.1 found ~84.5% of episodes PIT-reconstructable;
thin-file rows are separately excluded by R2/R3's own design). Missing
PD is never imputed, backfilled, or silently treated as "similar."

COVERAGE DEFINITIONS (per explicit review -- two distinct questions,
neither one a new gate):
    Coverage_delta^units  = n_similar_delta / n_units_outcome_both
        "Of the paired units that were eligible to support the
        original breach classification, what fraction remain usable
        after requiring comparable reconstructed PD?" n_units_
        outcome_both (NOT the broader n_units_pair) is the denominator
        because it is the population whose SIZE already gated whether
        the original cell was even classified as evidence_gap vs.
        testable -- the natural base to re-test adequacy against.
    Coverage_delta^loans  = (n_eligible_low_delta + n_eligible_high_delta)
                            / (n_eligible_low_original + n_eligible_high_original)
        How much of the LOAN mass underlying the original loan-
        weighted test statistic survives PD-filtering. Reported
        descriptively only -- a cell can retain most of its paired
        units but little of its loan mass, or vice versa, and both
        facts matter for reading the result, but NEITHER becomes a new
        threshold at this stage.

PD-SIMILARITY SWEEP: delta in {0.02, 0.05, 0.10} (PD points -- i.e. 2pp/
5pp/10pp of probability, analogous to but a DIFFERENT scale than the
bad-rate tolerance_pp sweep). No single delta is treated as primary;
stability across all three is itself part of the evidence, exactly as
the existing tolerance_pp sweep is never collapsed to one "correct"
value.

DELTA_PD SIGN CONVENTION, explicit and immutable:
    delta_pd = PD_high_exposure - PD_low_exposure
Positive always means the higher-exposure side was reconstructed as
riskier. pd_low_median and pd_high_median are retained alongside the
signed difference (cheap, and make a surprising sign easy to audit).

VERDICT PER BREACH CELL PER DELTA -- "insufficient evidence never
reopens a breached frontier," the single governing principle here:
    insufficient_to_assess  -- n_similar_delta < EVIDENCE_GAP_MIN_N
                               (reuses the EXISTING threshold -- no new
                               cutoff invented). The risk-conditioned
                               walk treats this EXACTLY like the
                               original risk_breach: it still halts.
    persistent               -- adequate n_similar_delta AND
                               delta_br_similar_delta_pp > tolerance_pp
                               (same tolerance_pp as the cell's own
                               original tolerance). Still halts.
    attenuated                -- adequate n_similar_delta AND
                               delta_br_similar_delta_pp <= tolerance_pp.
                               Only this outcome lets the PD-comparable
                               walk continue past this tier.
Non-breach cells (reference_tier/supported/evidence_gap/not_evaluated_
beyond_breach) pass through UNCHANGED into the PD-comparable walk --
PD-conditioning only ever matters for cells originally classified
risk_breach.

OUTPUT, per fundamental (float, commission) -- SIX drop-in frontier-
summary files total, kept deliberately separate rather than collapsing
to one delta (premature at this stage):
    {out_prefix}_{fund}_pd_comparable_classification_audit.csv
        One row per original (band, tolerance_pp, candidate_tier_ugx)
        cell: the REPRODUCED original columns (for audit), plus
        unconditional pd_low_median/pd_high_median/delta_pd_median/
        n_units_with_usable_pd_both_sides, plus per-delta (x3) n_similar,
        coverage_units, coverage_loans, delta_br_similar_pp, verdict,
        confidence.
    {out_prefix}_{fund}_pd_comparable_frontier_summary_delta{d}.csv  (x3)
        BYTE-FOR-BYTE the same schema as derive_capacity_frontier_from_
        business_state.py's own *_frontier_summary.csv -- a true drop-in
        replacement. derive_capacity_function_and_backtest_frontier.py
        needs ZERO code changes to consume this; point --float-
        frontier-summary-csv / --commission-frontier-summary-csv at it
        and re-run that script UNCHANGED, once per delta, to get the
        k-sensitivity table against the PD-comparable surface.

NEVER in this script: refitting/adjusting the real classification CSV
or the frontier walk's baseline-selection logic; a new PD-coverage
threshold; collapsing float and commission; applying C3, Capacity(F),
or selecting k (that is the separate, next step this script's output
feeds into, run by hand against the unchanged backtest script).

Usage:
    python scripts\\compute_pd_comparable_capacity_frontier.py ^
        --episode-dataset loan_episode_capacity_dataset.csv ^
        --retrospective-cal-scores-csv stage4br3_retrospective_cal_scores_scores.csv ^
        --float-frontier-classification-csv capacity_frontier_float_frontier_classification.csv ^
        --commission-frontier-classification-csv capacity_frontier_commission_frontier_classification.csv ^
        --out-prefix pd_comparable_frontier
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPOSURE_TIERS_UGX = [50_000, 100_000, 250_000, 350_000, 500_000, 750_000, 1_000_000]
FUNDAMENTALS_FOR_BANDS = {"float_activity_value_1m": "float", "commission": "commission"}
N_DECILES_DEFAULT = 10
EVIDENCE_GAP_MIN_N = 10
ROBUST_MIN_N = 100
BASELINE_MIN_N = 100
DEFAULT_TOLERANCES_PP = [2.0, 3.0, 4.0, 5.0]
DEFAULT_PD_SIMILARITY_THRESHOLDS = [0.02, 0.05, 0.10]

CAL_PD_COL = "retrospective_current_cal_pd"

REQUIRED_EPISODE_COLS = [
    "disbursement_fid", "agent_msisdn", "fundamentals_snapshot_date", "disbursement_amount_ugx",
    "float_activity_value_1m", "commission", "bad_state_3dpd_30d", "label_eligible_30d",
]
REQUIRED_CLASSIFICATION_COLS = [
    "band", "band_rank", "baseline_tier_ugx", "baseline_n_units_marginal", "baseline_min_n_used",
    "tolerance_pp", "candidate_tier_ugx", "n_units_pair", "n_units_outcome_both",
    "bad_rate_lower_loan_weighted", "bad_rate_higher_loan_weighted", "delta_br_loan_weighted_pp",
    "classification", "confidence",
]

REPRODUCTION_TOLERANCE = 1e-6


def _sha256_file(path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


# ============================================================================
# Restated verbatim from derive_capacity_frontier_from_business_state.py
# (one-way scripts/ layering convention).
# ============================================================================


def _qcut_safe(s: pd.Series, q: int, prefix: str = "D") -> tuple:
    try:
        codes, bins = pd.qcut(s, q, duplicates="drop", retbins=True, labels=False)
    except ValueError:
        codes, bins = None, None
    n_bins = (len(bins) - 1) if bins is not None else 0
    if n_bins <= 0:
        return pd.Series(f"{prefix}1", index=s.index), 1
    labels = [f"{prefix}{i + 1}" for i in range(n_bins)]
    return codes.map(dict(enumerate(labels))), n_bins


def load_reconstructed_cal_pd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c in {"disbursement_fid", CAL_PD_COL})
    missing = [c for c in ("disbursement_fid", CAL_PD_COL) if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}")
    dup = int(df["disbursement_fid"].duplicated().sum())
    if dup:
        sys.exit(f"ERROR: {dup} duplicate disbursement_fid value(s) in {path}.")
    return df


def load_episode_dataset_with_pd(episode_dataset_path: Path, cal_pd_df: pd.DataFrame) -> pd.DataFrame:
    path = Path(episode_dataset_path)
    if not path.exists():
        sys.exit(f"ERROR: {path} not found -- run build_loan_episode_capacity_dataset.py first.")
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_EPISODE_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}")

    n_before = len(df)
    df = df.merge(cal_pd_df, on="disbursement_fid", how="left")
    if len(df) != n_before:
        sys.exit("ERROR: PD join changed row count -- disbursement_fid is not behaving as a key.")

    df_valid = df[df["fundamentals_snapshot_date"].notna()].copy()
    n_total_valid = len(df_valid)
    n_with_pd = int(df_valid[CAL_PD_COL].notna().sum())
    known_tier = df_valid[df_valid["disbursement_amount_ugx"].isin(EXPOSURE_TIERS_UGX)]
    n_known_tier_with_pd = int(known_tier[CAL_PD_COL].notna().sum())
    print(f"PD coverage: {n_with_pd:,}/{n_total_valid:,} ({100.0 * n_with_pd / n_total_valid:.2f}%) of loans "
          f"with a valid measured business-state anchor have a reconstructed {CAL_PD_COL}; "
          f"{n_known_tier_with_pd:,}/{len(known_tier):,} "
          f"({100.0 * n_known_tier_with_pd / max(len(known_tier), 1):.2f}%) among loans at a known exposure tier. "
          f"Missing PD is never imputed or backfilled.")
    return df_valid


def build_agent_period_summary(df: pd.DataFrame) -> pd.DataFrame:
    key = ["agent_msisdn", "fundamentals_snapshot_date"]
    gb = df.groupby(key, sort=False)
    summary = pd.concat([
        gb.size().rename("n_loans_total"),
        gb["float_activity_value_1m"].first().rename("float_activity_value_1m"),
        gb["commission"].first().rename("commission"),
    ], axis=1).reset_index()
    return summary


def assign_bands(agent_period_summary: pd.DataFrame, fund_col: str, n_bands: int = N_DECILES_DEFAULT) -> tuple:
    valid_mask = agent_period_summary[fund_col].notna() & (agent_period_summary[fund_col] > 0)
    out = pd.Series(np.nan, index=agent_period_summary.index, dtype=object)
    if not valid_mask.any():
        return out, []
    band_series, n_bins = _qcut_safe(agent_period_summary.loc[valid_mask, fund_col], n_bands)
    labels = [f"D{i + 1}" for i in range(n_bins)]
    out.loc[valid_mask] = band_series.values
    return out, labels


def build_tier_marginal_table(df_with_bands: pd.DataFrame, band_col: str) -> pd.DataFrame:
    known = df_with_bands[
        df_with_bands["disbursement_amount_ugx"].isin(EXPOSURE_TIERS_UGX) & df_with_bands[band_col].notna()
    ].copy()
    cols = [band_col, "tier_ugx", "n_units_at_tier"]
    if known.empty:
        return pd.DataFrame(columns=cols)
    known["_unit_key"] = known["agent_msisdn"].astype(str) + "||" + known["fundamentals_snapshot_date"].astype(str)
    marginal = known.groupby([band_col, "disbursement_amount_ugx"], sort=False).agg(
        n_units_at_tier=("_unit_key", "nunique"),
    ).reset_index().rename(columns={"disbursement_amount_ugx": "tier_ugx"})
    marginal["tier_ugx"] = marginal["tier_ugx"].astype(int)
    return marginal[cols]


def select_baseline_tier(tier_marginal_band: pd.DataFrame, baseline_min_n: int):
    lookup = dict(zip(tier_marginal_band["tier_ugx"], tier_marginal_band["n_units_at_tier"], strict=True))
    for tier in EXPOSURE_TIERS_UGX:
        if lookup.get(tier, 0) >= baseline_min_n:
            return tier
    return None


def build_unit_pair_rows_with_pd(df_with_bands: pd.DataFrame) -> pd.DataFrame:
    """Extends the original build_unit_pair_rows with PD alongside the
    existing bad-rate aggregation: for each (unit, tier), pd_mean is the
    unweighted mean of CAL_PD_COL among that tier's loans within that
    unit that have a non-null PD (never zero-filled); n_pd_usable is
    that count. A (unit,tier) side's PD is 'usable' only if n_pd_usable
    >= 1 -- never imputed from the other side or from a population
    average."""
    key = ["agent_msisdn", "fundamentals_snapshot_date"]
    known = df_with_bands[df_with_bands["disbursement_amount_ugx"].isin(EXPOSURE_TIERS_UGX)].copy()
    known["_bad_elig"] = known["bad_state_3dpd_30d"] * known["label_eligible_30d"]
    known["_pd_usable"] = known[CAL_PD_COL].notna().astype(int)
    known["_pd_for_sum"] = known[CAL_PD_COL].fillna(0.0)

    per_tier = known.groupby(key + ["disbursement_amount_ugx"], sort=False).agg(
        n_loans=("disbursement_amount_ugx", "size"),
        n_eligible=("label_eligible_30d", "sum"),
        n_bad=("_bad_elig", "sum"),
        n_pd_usable=("_pd_usable", "sum"),
        _pd_sum=("_pd_for_sum", "sum"),
    ).reset_index()
    per_tier["n_eligible"] = per_tier["n_eligible"].astype(int)
    per_tier["n_bad"] = per_tier["n_bad"].astype(int)
    per_tier["n_pd_usable"] = per_tier["n_pd_usable"].astype(int)
    per_tier["br"] = np.where(per_tier["n_eligible"] > 0, per_tier["n_bad"] / per_tier["n_eligible"], np.nan)
    per_tier["pd"] = np.where(per_tier["n_pd_usable"] > 0, per_tier["_pd_sum"] / per_tier["n_pd_usable"], np.nan)
    per_tier = per_tier.drop(columns=["_pd_sum"])

    multi_tier_units = per_tier.groupby(key, sort=False)["disbursement_amount_ugx"].transform("size") >= 2
    multi = per_tier[multi_tier_units]
    if multi.empty:
        return pd.DataFrame()

    merged = multi.merge(multi, on=key, suffixes=("_low", "_high"))
    merged = merged[merged["disbursement_amount_ugx_low"] < merged["disbursement_amount_ugx_high"]]

    out = merged.rename(columns={
        "disbursement_amount_ugx_low": "tier_low", "disbursement_amount_ugx_high": "tier_high",
    })[key + ["tier_low", "tier_high", "n_loans_low", "n_loans_high", "n_eligible_low", "n_eligible_high",
              "n_bad_low", "n_bad_high", "br_low", "br_high",
              "n_pd_usable_low", "n_pd_usable_high", "pd_low", "pd_high"]].copy()
    out["tier_low"] = out["tier_low"].astype(int)
    out["tier_high"] = out["tier_high"].astype(int)
    out["delta_br"] = np.where(out["br_low"].notna() & out["br_high"].notna(), out["br_high"] - out["br_low"], np.nan)
    out["delta_pd"] = np.where(
        (out["n_pd_usable_low"] > 0) & (out["n_pd_usable_high"] > 0), out["pd_high"] - out["pd_low"], np.nan)
    return out.reset_index(drop=True)


def attach_band_to_pairs(unit_pairs: pd.DataFrame, agent_period_summary: pd.DataFrame, band_col: str) -> pd.DataFrame:
    key = ["agent_msisdn", "fundamentals_snapshot_date"]
    if unit_pairs.empty:
        return unit_pairs.assign(**{band_col: pd.Series(dtype=object)})
    banded = unit_pairs.merge(agent_period_summary[key + [band_col]], on=key, how="left")
    return banded[banded[band_col].notna()]


# ============================================================================
# Reproduction of the ORIGINAL pooled statistic (restated from
# aggregate_baseline_pairs), needed as the reproduction-gate target AND
# as the base for the loan-weighted-coverage denominator.
# ============================================================================


def reproduce_pooled_cell_stats(pairs_for_candidate: pd.DataFrame) -> dict:
    n_units_pair = len(pairs_for_candidate)
    n_loans_lower = int(pairs_for_candidate["n_loans_low"].sum())
    n_loans_higher = int(pairs_for_candidate["n_loans_high"].sum())
    n_eligible_lower = int(pairs_for_candidate["n_eligible_low"].sum())
    n_eligible_higher = int(pairs_for_candidate["n_eligible_high"].sum())
    n_bad_lower = int(pairs_for_candidate["n_bad_low"].sum())
    n_bad_higher = int(pairs_for_candidate["n_bad_high"].sum())
    bad_rate_lower = n_bad_lower / n_eligible_lower if n_eligible_lower else np.nan
    bad_rate_higher = n_bad_higher / n_eligible_higher if n_eligible_higher else np.nan
    delta_br_loan_weighted_pp = (
        (bad_rate_higher - bad_rate_lower) * 100 if pd.notna(bad_rate_lower) and pd.notna(bad_rate_higher) else np.nan
    )
    both = pairs_for_candidate[pairs_for_candidate["br_low"].notna() & pairs_for_candidate["br_high"].notna()]
    n_units_outcome_both = len(both)
    return {
        "n_units_pair": n_units_pair, "n_loans_lower": n_loans_lower, "n_loans_higher": n_loans_higher,
        "n_eligible_lower": n_eligible_lower, "n_eligible_higher": n_eligible_higher,
        "bad_rate_lower_loan_weighted": bad_rate_lower, "bad_rate_higher_loan_weighted": bad_rate_higher,
        "delta_br_loan_weighted_pp": delta_br_loan_weighted_pp, "n_units_outcome_both": n_units_outcome_both,
        "units_outcome_both_df": both,
    }


def reclassify(n_units_outcome_both, delta_br_loan_weighted_pp, tolerance_pp, evidence_gap_min_n, robust_min_n) -> tuple:
    """Restated from classify_candidate_tiers's per-candidate gate logic
    (the reference_tier/not_evaluated_beyond_breach walk bookkeeping
    lives in the caller, which has the ascending-tier context)."""
    if pd.isna(n_units_outcome_both) or n_units_outcome_both < evidence_gap_min_n:
        return "evidence_gap", ""
    if pd.notna(delta_br_loan_weighted_pp) and delta_br_loan_weighted_pp > tolerance_pp:
        return "risk_breach", ("robust" if n_units_outcome_both >= robust_min_n else "exploratory")
    return "supported", ("robust" if n_units_outcome_both >= robust_min_n else "exploratory")


def _empty_pairs_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "n_loans_low", "n_loans_high", "n_eligible_low", "n_eligible_high", "n_bad_low", "n_bad_high",
        "br_low", "br_high", "n_pd_usable_low", "n_pd_usable_high", "pd_low", "pd_high",
    ])


def get_cell_pairs(pairs_band: pd.DataFrame, baseline_tier: int, candidate_tier: int) -> pd.DataFrame:
    if pairs_band.empty or "tier_low" not in pairs_band.columns:
        return _empty_pairs_frame()
    return pairs_band[(pairs_band["tier_low"] == baseline_tier) & (pairs_band["tier_high"] == candidate_tier)]


def verify_reproduction(real_classification: pd.DataFrame, pairs_with_band: pd.DataFrame,
                         band_col: str, fund_name: str, evidence_gap_min_n: int = EVIDENCE_GAP_MIN_N,
                         robust_min_n: int = ROBUST_MIN_N) -> None:
    """THE reproduction gate. Hard-exits on ANY mismatch, before a single
    PD-conditioned number is computed.

    evidence_gap_min_n/robust_min_n MUST be the exact values the real
    classification CSV was itself generated with (the CLI already
    requires this of --n-bands/--baseline-min-n; the same contract
    applies here) -- they default to this module's own constants only
    for direct/standalone calls, never silently ignoring caller-supplied
    values the way a hardcoded reference would.

    For rows ORIGINALLY evaluated (classification != not_evaluated_
    beyond_breach): requires n_units_outcome_both, delta_br_loan_
    weighted_pp, AND classification to match exactly.

    For "not_evaluated_beyond_breach" rows: the stored label is
    PROCEDURAL (a consequence of where the ORIGINAL walk happened to
    halt), not an intrinsic property of that tier's own statistics --
    classify_candidate_tiers never tests these for pass/fail. So this
    gate requires ONLY that the underlying frozen statistics (n_units_
    outcome_both, delta_br_loan_weighted_pp) still match -- never a
    classification-label match, which would be nonsensical here."""
    mismatches = []
    checked = 0
    candidate_rows = real_classification[real_classification["classification"] != "reference_tier"]
    for _, row in candidate_rows.iterrows():
        band = row["band"]
        baseline_tier = int(row["baseline_tier_ugx"])
        candidate_tier = int(row["candidate_tier_ugx"])
        tol = float(row["tolerance_pp"])

        pairs_band = pairs_with_band[pairs_with_band[band_col] == band]
        cell_pairs = get_cell_pairs(pairs_band, baseline_tier, candidate_tier)
        stats = reproduce_pooled_cell_stats(cell_pairs)
        checked += 1

        real_n = row["n_units_outcome_both"]
        real_delta = row["delta_br_loan_weighted_pp"]
        n_match = (pd.isna(real_n) and stats["n_units_outcome_both"] == 0) or (
            pd.notna(real_n) and int(real_n) == stats["n_units_outcome_both"])
        delta_match = (pd.isna(real_delta) and pd.isna(stats["delta_br_loan_weighted_pp"])) or (
            pd.notna(real_delta) and pd.notna(stats["delta_br_loan_weighted_pp"])
            and abs(float(real_delta) - stats["delta_br_loan_weighted_pp"]) < REPRODUCTION_TOLERANCE)

        if row["classification"] == "not_evaluated_beyond_breach":
            ok = n_match and delta_match
        else:
            reclass, _ = reclassify(stats["n_units_outcome_both"], stats["delta_br_loan_weighted_pp"],
                                     tol, evidence_gap_min_n, robust_min_n)
            ok = n_match and delta_match and (reclass == row["classification"])

        if not ok:
            mismatches.append({
                "band": band, "tolerance_pp": tol, "baseline_tier_ugx": baseline_tier,
                "candidate_tier_ugx": candidate_tier, "real_classification": row["classification"],
                "real_n_units_outcome_both": real_n, "reproduced_n_units_outcome_both": stats["n_units_outcome_both"],
                "real_delta_br_loan_weighted_pp": real_delta,
                "reproduced_delta_br_loan_weighted_pp": stats["delta_br_loan_weighted_pp"],
            })

    print(f"Reproduction gate ({fund_name}): checked {checked:,} non-reference classification row(s) "
          f"(stats-only check for not_evaluated_beyond_breach rows, full match otherwise).")
    if mismatches:
        sys.exit(
            f"ERROR: REPRODUCTION GATE FAILED ({fund_name}) -- {len(mismatches)} row(s) of the real "
            f"classification CSV could not be reproduced exactly by this script's restated pipeline. "
            f"Refusing to generate any PD-comparable frontier output, since the difference could be "
            f"pipeline drift rather than PD conditioning. First mismatch: {mismatches[0]}"
        )
    print(f"Reproduction gate ({fund_name}): PASSED -- every checked row's frozen statistics match the "
          f"real classification CSV exactly.")


# ============================================================================
# PD-similarity filtering and re-walk
# ============================================================================


def compute_pd_stats_for_cell(cell_pairs: pd.DataFrame, units_outcome_both: pd.DataFrame) -> dict:
    """Unconditional (not delta-filtered) PD diagnostics for one cell,
    computed over units_outcome_both (the gating population) further
    restricted to usable PD on both sides. delta_pd sign convention:
    PD_high - PD_low (positive = higher-exposure side reconstructed as
    riskier)."""
    pd_both = units_outcome_both[(units_outcome_both["n_pd_usable_low"] > 0)
                                  & (units_outcome_both["n_pd_usable_high"] > 0)]
    n_units_with_usable_pd_both_sides = len(pd_both)
    if n_units_with_usable_pd_both_sides == 0:
        return {
            "pd_low_median": np.nan, "pd_high_median": np.nan, "delta_pd_median": np.nan,
            "n_units_with_usable_pd_both_sides": 0,
        }
    return {
        "pd_low_median": float(pd_both["pd_low"].median()),
        "pd_high_median": float(pd_both["pd_high"].median()),
        "delta_pd_median": float((pd_both["pd_high"] - pd_both["pd_low"]).median()),
        "n_units_with_usable_pd_both_sides": n_units_with_usable_pd_both_sides,
    }


def compute_delta_filtered_stats(units_outcome_both: pd.DataFrame, pooled_stats: dict, delta: float,
                                  tolerance_pp: float, evidence_gap_min_n: int, robust_min_n: int) -> dict:
    """For one (cell, delta): filters units_outcome_both to usable-PD-
    both-sides AND |delta_pd| <= delta, recomputes the loan-weighted
    delta_br WITHIN that subset, and classifies the verdict. Coverage_
    units and Coverage_loans are both reported descriptively -- neither
    gates anything; evidence adequacy is governed solely by n_similar
    vs. EVIDENCE_GAP_MIN_N/ROBUST_MIN_N, reused unchanged."""
    pd_both = units_outcome_both[(units_outcome_both["n_pd_usable_low"] > 0)
                                  & (units_outcome_both["n_pd_usable_high"] > 0)]
    similar = pd_both[(pd_both["pd_high"] - pd_both["pd_low"]).abs() <= delta]
    n_similar = len(similar)

    n_units_outcome_both = pooled_stats["n_units_outcome_both"]
    coverage_units = (n_similar / n_units_outcome_both) if n_units_outcome_both else np.nan

    n_eligible_low_similar = int(similar["n_eligible_low"].sum())
    n_eligible_high_similar = int(similar["n_eligible_high"].sum())
    n_loans_original = pooled_stats["n_eligible_lower"] + pooled_stats["n_eligible_higher"]
    n_loans_similar = n_eligible_low_similar + n_eligible_high_similar
    coverage_loans = (n_loans_similar / n_loans_original) if n_loans_original else np.nan

    n_bad_low_similar = int(similar["n_bad_low"].sum())
    n_bad_high_similar = int(similar["n_bad_high"].sum())
    bad_rate_lower_similar = n_bad_low_similar / n_eligible_low_similar if n_eligible_low_similar else np.nan
    bad_rate_higher_similar = n_bad_high_similar / n_eligible_high_similar if n_eligible_high_similar else np.nan
    delta_br_similar_pp = (
        (bad_rate_higher_similar - bad_rate_lower_similar) * 100
        if pd.notna(bad_rate_lower_similar) and pd.notna(bad_rate_higher_similar) else np.nan
    )

    if n_similar < evidence_gap_min_n:
        verdict, confidence = "insufficient_to_assess", ""
    elif pd.notna(delta_br_similar_pp) and delta_br_similar_pp > tolerance_pp:
        verdict, confidence = "persistent", ("robust" if n_similar >= robust_min_n else "exploratory")
    else:
        verdict, confidence = "attenuated", ("robust" if n_similar >= robust_min_n else "exploratory")

    return {
        "n_similar": n_similar, "coverage_units": coverage_units, "coverage_loans": coverage_loans,
        "n_eligible_low_similar": n_eligible_low_similar, "n_eligible_high_similar": n_eligible_high_similar,
        "bad_rate_lower_similar": bad_rate_lower_similar, "bad_rate_higher_similar": bad_rate_higher_similar,
        "delta_br_similar_pp": delta_br_similar_pp, "verdict": verdict, "confidence": confidence,
    }


def build_audit_and_summaries_for_band(
    band: str, band_rank: int, real_classification_band: pd.DataFrame, pairs_band: pd.DataFrame,
    pd_thresholds: list[float], evidence_gap_min_n: int, robust_min_n: int,
) -> tuple:
    """Full independent re-walk per delta (module docstring, VERDICT PER
    BREACH CELL PER DELTA). For every tier in EXPOSURE_TIERS_UGX above
    baseline_tier, ALWAYS recompute a fresh, delta-independent intrinsic
    classification from the frozen pooled statistics via reclassify() --
    never read a tier's walk-continuation behavior off the real table's
    stored classification label. That label is PATH-DEPENDENT (a
    consequence of where the ORIGINAL walk happened to halt), so a tier
    the original walk never reached past a breach (stored as
    "not_evaluated_beyond_breach", or entirely absent from the real
    table) must still be evaluated on its own merits here, once an
    earlier breach attenuates under PD-comparability.

    original_classification (from the real table, or the sentinel
    "not_present_in_original_table" when the original walk produced no
    row at all for that tier -- e.g. "never observed under any anchor"
    past a breach) is carried on every audit row for comparison only; it
    never drives this function's own walk logic.

    Each of pd_thresholds runs its OWN, fully independent walk: a tier
    whose FRESH intrinsic classification is "risk_breach" is the only
    case that invokes the PD-similarity test for that delta; "attenuated"
    lets that delta's walk continue upward exactly like "supported";
    "persistent" or "insufficient_to_assess" halts that delta's walk --
    and ONLY that delta's -- for every tier above. "evidence_gap" never
    halts any delta's walk; it only breaks that delta's CONTIGUOUS run.

    Returns (audit_rows_for_band, {tolerance_pp: {delta: frontier_summary_row}}).
    """
    audit_rows = []
    summaries: dict[float, dict[float, dict]] = {}

    for tol in sorted(real_classification_band["tolerance_pp"].unique()):
        rows_at_tol = real_classification_band[real_classification_band["tolerance_pp"] == tol]
        ref_row = rows_at_tol[rows_at_tol["classification"] == "reference_tier"].iloc[0]
        baseline_tier = int(ref_row["baseline_tier_ugx"])
        original_lookup = {
            int(r["candidate_tier_ugx"]): r["classification"]
            for _, r in rows_at_tol.iterrows() if r["classification"] != "reference_tier"
        }

        for delta in pd_thresholds:
            summaries.setdefault(tol, {})[delta] = {
                "status": "ok", "band": band, "band_rank": band_rank,
                "baseline_tier_ugx": baseline_tier, "tolerance_pp": tol,
                "contiguous_supported_frontier_tier_ugx": baseline_tier,
                "highest_tier_with_any_supported_evidence_ugx": baseline_tier,
                "first_risk_breach_tier_ugx": np.nan,
            }
        contiguous_open = dict.fromkeys(pd_thresholds, True)
        halted = dict.fromkeys(pd_thresholds, False)

        for candidate_tier in EXPOSURE_TIERS_UGX:
            if candidate_tier <= baseline_tier:
                continue
            if all(halted[d] for d in pd_thresholds):
                break  # every delta's independent walk has already halted

            cell_pairs = get_cell_pairs(pairs_band, baseline_tier, candidate_tier)
            pooled = reproduce_pooled_cell_stats(cell_pairs)
            units_both = pooled["units_outcome_both_df"]
            pd_uncond = compute_pd_stats_for_cell(cell_pairs, units_both)
            intrinsic_class, intrinsic_conf = reclassify(
                pooled["n_units_outcome_both"], pooled["delta_br_loan_weighted_pp"],
                tol, evidence_gap_min_n, robust_min_n)

            original_class = original_lookup.get(candidate_tier, "not_present_in_original_table")

            audit_row = {
                "band": band, "band_rank": band_rank, "baseline_tier_ugx": baseline_tier,
                "tolerance_pp": tol, "candidate_tier_ugx": candidate_tier,
                "original_classification": original_class,
                "intrinsic_classification": intrinsic_class, "intrinsic_confidence": intrinsic_conf,
                "n_units_outcome_both": pooled["n_units_outcome_both"],
                "delta_br_loan_weighted_pp": pooled["delta_br_loan_weighted_pp"],
                "pd_low_median": pd_uncond["pd_low_median"], "pd_high_median": pd_uncond["pd_high_median"],
                "delta_pd_median": pd_uncond["delta_pd_median"],
                "n_units_with_usable_pd_both_sides": pd_uncond["n_units_with_usable_pd_both_sides"],
            }

            for delta in pd_thresholds:
                suffix = f"_delta{delta:.2f}"
                summ = summaries[tol][delta]

                if halted[delta]:
                    audit_row[f"pd_comparable_classification{suffix}"] = "not_evaluated_beyond_breach"
                    audit_row[f"n_similar{suffix}"] = np.nan
                    audit_row[f"coverage_units{suffix}"] = np.nan
                    audit_row[f"coverage_loans{suffix}"] = np.nan
                    audit_row[f"delta_br_similar_pp{suffix}"] = np.nan
                    audit_row[f"verdict{suffix}"] = ""
                    audit_row[f"confidence{suffix}"] = ""
                    continue

                if intrinsic_class == "evidence_gap":
                    contiguous_open[delta] = False
                    audit_row[f"pd_comparable_classification{suffix}"] = "evidence_gap"
                    audit_row[f"n_similar{suffix}"] = np.nan
                    audit_row[f"coverage_units{suffix}"] = np.nan
                    audit_row[f"coverage_loans{suffix}"] = np.nan
                    audit_row[f"delta_br_similar_pp{suffix}"] = np.nan
                    audit_row[f"verdict{suffix}"] = ""
                    audit_row[f"confidence{suffix}"] = ""
                    continue

                if intrinsic_class == "supported":
                    if contiguous_open[delta]:
                        summ["contiguous_supported_frontier_tier_ugx"] = candidate_tier
                    summ["highest_tier_with_any_supported_evidence_ugx"] = candidate_tier
                    audit_row[f"pd_comparable_classification{suffix}"] = "supported"
                    audit_row[f"n_similar{suffix}"] = np.nan
                    audit_row[f"coverage_units{suffix}"] = np.nan
                    audit_row[f"coverage_loans{suffix}"] = np.nan
                    audit_row[f"delta_br_similar_pp{suffix}"] = np.nan
                    audit_row[f"verdict{suffix}"] = ""
                    audit_row[f"confidence{suffix}"] = ""
                    continue

                # intrinsic_class == "risk_breach": a FRESH candidate breach --
                # the only case that invokes the PD-similarity test.
                stats = compute_delta_filtered_stats(
                    units_both, pooled, delta, tol, evidence_gap_min_n, robust_min_n)
                audit_row[f"n_similar{suffix}"] = stats["n_similar"]
                audit_row[f"coverage_units{suffix}"] = stats["coverage_units"]
                audit_row[f"coverage_loans{suffix}"] = stats["coverage_loans"]
                audit_row[f"delta_br_similar_pp{suffix}"] = stats["delta_br_similar_pp"]
                audit_row[f"verdict{suffix}"] = stats["verdict"]
                audit_row[f"confidence{suffix}"] = stats["confidence"]

                if stats["verdict"] == "attenuated":
                    audit_row[f"pd_comparable_classification{suffix}"] = "attenuated_risk_breach"
                    if contiguous_open[delta]:
                        summ["contiguous_supported_frontier_tier_ugx"] = candidate_tier
                    summ["highest_tier_with_any_supported_evidence_ugx"] = candidate_tier
                else:
                    # "persistent" or "insufficient_to_assess" -- halts
                    # this delta's walk, and only this delta's, for good.
                    audit_row[f"pd_comparable_classification{suffix}"] = (
                        "persistent_risk_breach" if stats["verdict"] == "persistent" else "insufficient_to_assess")
                    if pd.isna(summ["first_risk_breach_tier_ugx"]):
                        summ["first_risk_breach_tier_ugx"] = candidate_tier
                    contiguous_open[delta] = False
                    halted[delta] = True

            audit_rows.append(audit_row)

    return audit_rows, summaries


def process_fundamental(
    fund_col: str, fund_name: str, df_valid: pd.DataFrame, real_classification: pd.DataFrame,
    pd_thresholds: list[float], n_bands: int, baseline_min_n: int, evidence_gap_min_n: int, robust_min_n: int,
) -> tuple:
    band_col = f"{fund_name}_band"
    agent_period_summary = build_agent_period_summary(df_valid)
    agent_period_summary[band_col], band_labels = assign_bands(agent_period_summary, fund_col, n_bands)
    if not band_labels:
        sys.exit(f"ERROR: no valid {fund_name} bands -- cannot proceed.")

    df_with_bands = df_valid.merge(
        agent_period_summary[["agent_msisdn", "fundamentals_snapshot_date", band_col]],
        on=["agent_msisdn", "fundamentals_snapshot_date"], how="left",
    )
    unit_pairs = build_unit_pair_rows_with_pd(df_with_bands)
    pairs_with_band = attach_band_to_pairs(unit_pairs, agent_period_summary, band_col)

    verify_reproduction(real_classification, pairs_with_band, band_col, fund_name, evidence_gap_min_n, robust_min_n)

    all_audit_rows = []
    all_summaries: dict[float, list[dict]] = {delta: [] for delta in pd_thresholds}
    for band_rank, band in enumerate(band_labels, start=1):
        real_classification_band = real_classification[real_classification["band"] == band]
        if real_classification_band.empty:
            continue
        pairs_band = pairs_with_band[pairs_with_band[band_col] == band]
        audit_rows, summaries = build_audit_and_summaries_for_band(
            band, band_rank, real_classification_band, pairs_band,
            pd_thresholds, evidence_gap_min_n, robust_min_n,
        )
        all_audit_rows.extend(audit_rows)
        for by_delta in summaries.values():
            for delta, summ in by_delta.items():
                all_summaries[delta].append(summ)

    audit_df = pd.DataFrame(all_audit_rows)
    summary_dfs = {delta: pd.DataFrame(rows) for delta, rows in all_summaries.items()}
    return audit_df, summary_dfs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dataset", default="loan_episode_capacity_dataset.csv")
    ap.add_argument("--retrospective-cal-scores-csv", type=Path, required=True)
    ap.add_argument("--float-frontier-classification-csv", type=Path, required=True,
                     help="Analysis 4 Deliverable 1's real output. Run "
                          "derive_capacity_frontier_from_business_state.py separately first if missing "
                          "-- this script never regenerates it.")
    ap.add_argument("--commission-frontier-classification-csv", type=Path, required=True)
    ap.add_argument("--pd-similarity-thresholds", type=str,
                     default=",".join(str(d) for d in DEFAULT_PD_SIMILARITY_THRESHOLDS))
    ap.add_argument("--n-bands", type=int, default=N_DECILES_DEFAULT,
                     help="Must match the --n-bands used for the real classification CSVs, or the "
                          "reproduction gate will correctly fail.")
    ap.add_argument("--baseline-min-n", type=int, default=BASELINE_MIN_N)
    ap.add_argument("--evidence-gap-min-n", type=int, default=EVIDENCE_GAP_MIN_N)
    ap.add_argument("--robust-min-n", type=int, default=ROBUST_MIN_N)
    ap.add_argument("--out-prefix", type=str, default="pd_comparable_frontier")
    args = ap.parse_args(argv)

    for p, label in [(args.float_frontier_classification_csv, "--float-frontier-classification-csv"),
                      (args.commission_frontier_classification_csv, "--commission-frontier-classification-csv")]:
        if not Path(p).exists():
            sys.exit(f"ERROR: {label} {p} does not exist. This script never regenerates Deliverable 1's "
                      f"output -- run scripts/derive_capacity_frontier_from_business_state.py separately "
                      f"first (using the SAME --n-bands/--baseline-min-n/--evidence-gap-min-n/--robust-min-n "
                      f"this script was given), then re-run this script against its real output.")

    print(f"\n{'#' * 100}\nPD-COMPARABLE HISTORICAL EXPOSURE FRONTIER\n"
          f"Matching on one reconstructed PD dimension reduces observed risk-selection differences -- it "
          f"does NOT fully control for confounding. This is NOT Capacity(F), NOT C3, and does not select k.\n"
          f"{'#' * 100}")

    pd_thresholds = sorted({float(x) for x in args.pd_similarity_thresholds.split(",") if x.strip()})
    print(f"PD-similarity sweep: {pd_thresholds} (PD points). No single value is primary -- stability "
          f"across all three is itself part of the evidence.")

    cal_pd_df = load_reconstructed_cal_pd(args.retrospective_cal_scores_csv)
    df_valid = load_episode_dataset_with_pd(args.episode_dataset, cal_pd_df)

    for fund_col, fund_name, classification_csv in [
        ("float_activity_value_1m", "float", args.float_frontier_classification_csv),
        ("commission", "commission", args.commission_frontier_classification_csv),
    ]:
        print(f"\n{'#' * 100}\n# {fund_name}-banded\n{'#' * 100}")
        real_classification = pd.read_csv(classification_csv)
        missing = [c for c in REQUIRED_CLASSIFICATION_COLS if c not in real_classification.columns]
        if missing:
            sys.exit(f"ERROR: {classification_csv} is missing required column(s): {missing}")

        audit_df, summary_dfs = process_fundamental(
            fund_col, fund_name, df_valid, real_classification, pd_thresholds,
            args.n_bands, args.baseline_min_n, args.evidence_gap_min_n, args.robust_min_n,
        )

        audit_path = f"{args.out_prefix}_{fund_name}_pd_comparable_classification_audit.csv"
        audit_df.to_csv(audit_path, index=False)
        print(f"Wrote {audit_path} ({len(audit_df):,} row(s)).")

        n_original_breach_rows = int((audit_df["original_classification"] == "risk_breach").sum()) if len(audit_df) else 0
        n_intrinsic_breach_rows = int((audit_df["intrinsic_classification"] == "risk_breach").sum()) if len(audit_df) else 0
        n_newly_surfaced_breach_rows = int(
            ((audit_df["intrinsic_classification"] == "risk_breach")
             & (audit_df["original_classification"] != "risk_breach")).sum()
        ) if len(audit_df) else 0
        print(f"{n_original_breach_rows:,} cell(s) originally classified risk_breach; "
              f"{n_intrinsic_breach_rows:,} cell(s) freshly (re-)identified as risk_breach by this script's "
              f"independent per-tier re-walk ({n_newly_surfaced_breach_rows:,} of those were NOT originally "
              f"classified risk_breach -- e.g. a tier the original walk never reached past an earlier breach, "
              f"now evaluated on its own merits).")
        for delta in pd_thresholds:
            suffix = f"_delta{delta:.2f}"
            summary_path = f"{args.out_prefix}_{fund_name}_pd_comparable_frontier_summary{suffix}.csv"
            summary_dfs[delta].to_csv(summary_path, index=False)
            print(f"Wrote {summary_path}")
            if n_intrinsic_breach_rows:
                breach_rows = audit_df[audit_df["intrinsic_classification"] == "risk_breach"]
                verdict_counts = breach_rows[f"verdict{suffix}"].value_counts().to_dict()
                print(f"  delta={delta}: breach verdicts -> {verdict_counts}")

    print(f"\n{'#' * 100}\nTERMINOLOGY REMINDER\n{'#' * 100}")
    print(
        "This is a PD-COMPARABLE historical exposure frontier, not a fully risk-conditioned one -- "
        "matching on retrospective_current_cal_pd alone reduces, but does not eliminate, risk-selection "
        "confounding. Insufficient PD-comparable evidence NEVER reopens a breached frontier.\n\n"
        "NEXT STEP (not performed here, by design): re-run scripts/derive_capacity_function_and_backtest_"
        "frontier.py UNCHANGED, once per fundamental x delta (6 runs total), pointing --float-frontier-"
        "summary-csv / --commission-frontier-summary-csv at the *_pd_comparable_frontier_summary_delta*.csv "
        "files written here, to get the k-sensitivity table against the PD-comparable surface."
    )


if __name__ == "__main__":
    main()
