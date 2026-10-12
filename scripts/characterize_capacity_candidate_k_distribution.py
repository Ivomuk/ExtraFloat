"""
characterize_capacity_candidate_k_distribution.py
==================================================
Stage 4B follow-on. A DIFFERENT question from derive_capacity_function_
and_backtest_frontier.py's k-sensitivity table: that script asked "at what
k does Capacity(F) start exceeding the PD-comparable historical frontier
50%/75% of the time" (an exceedance-crossing question). The six-surface
result from that analysis found the global k is NOT point-identified by
the historical PD-comparable frontier alone (median crossings moved from
~0.073 at the tightest PD-similarity tolerance down to ~0.045-0.050 at the
loosest, against a ~0.03 unconditioned-frontier benchmark) -- a design
sensitivity, not a precise estimate. That result is FROZEN and this script
does not revisit it, add more PD tolerances, or compute any additional
crossing point.

This script instead asks: for a small, policy-relevant SET of candidate k
values, what does Capacity(F) actually produce in absolute UGX terms, and
how does that relate to business scale? A k could sit comfortably inside
the historical sensitivity envelope above and still produce economically
nonsensical capacities for some business profiles -- or the reverse. This
is a DESCRIPTIVE characterization of Capacity(F)'s own output distribution,
never another backtest against any historical frontier.

NOT A BACKTEST: this script takes no --float-frontier-summary-csv / --
commission-frontier-summary-csv argument at all, and never reads or joins
against Analysis 4's frontier CSVs. It restates the SAME usecols-
whitelisted episode-dataset read, band assignment, weight loading, and
Capacity(F) formula from derive_capacity_function_and_backtest_frontier.py
(one-way scripts/ layering convention -- never imports that module) --
REQUIRED_COLS below is the entire usecols whitelist; no outcome, exposure,
assigned_limit, or cal_pd column is ever named anywhere this script reads.

C3/PD COMPLETELY OUT OF SCOPE: no cal_pd column is read, no shadow/C3
multiplier is applied, L_full = Capacity_k(F) x M_C3(PD) is explicitly
deferred to a separate, later, already-planned step. This script only
ever computes and describes Capacity_k(F) in isolation.

CAPACITY_NOT_ESTIMABLE, NEVER CAPACITY=0: an agent-period with
float_activity_value_1m<=0 or commission<=0 (the same validity bar
assign_bands already uses) does not have a measured zero economic
capacity -- we simply lack usable fundamentals for it. These rows are
masked to NaN before any table is built, so overall/decile/tier-bucket/
concordance stats are computed only over the genuinely estimable
population; their TRUE count is still reported everywhere (never
silently dropped to n=0), just with NaN stats rather than a fabricated
near-zero capacity.

CANDIDATE k's ARE A POLICY-CONSEQUENCE CHARACTERIZATION, NOT A SEARCH:
--candidate-ks defaults to "0.03,0.04,0.05,0.06" (0.03 = the pre-Stage-4
unconditioned benchmark; 0.04/0.05/0.06 = the policy region emerging from
the frozen six-surface sensitivity result). This script never selects,
recommends, or optimizes a k -- it only reports what each candidate implies
so a human can make that choice informed by economic plausibility, not
just historical exceedance.

OUTPUT, per --out-prefix (five tables, one row per k in every table except
the tier-bucket one which is one row per (k, tier_bucket)):
    {out_prefix}_overall_stats.csv
        One row per k: n, median/p25/p75/mean/min/max capacity_ugx_at_k.
    {out_prefix}_by_float_decile.csv / _by_commission_decile.csv
        One row per (k, band): n, median/p25/p75 capacity_ugx_at_k --
        answers "proposed capacity by Float/Commission decile."
    {out_prefix}_tier_bucket_proportions.csv
        One row per (k, tier_bucket_upper_ugx): pct_in_bucket (the
        proportion of agent-periods whose capacity_ugx_at_k falls strictly
        into that half-open UGX bucket) and pct_at_or_below (cumulative) --
        answers "proportion below 50K/100K/etc." Buckets are
        EXPOSURE_TIERS_UGX's own boundaries plus an open-ended
        ">1,000,000" bucket; this is purely descriptive binning against
        the SAME ladder Analysis 4 used, never a backtest against it.
    {out_prefix}_by_concordance.csv
        One row per (k, concordance in {{concordant, middle, discordant,
        unknown}}): n, median/p25/p75 capacity_ugx_at_k -- the concordant/
        discordant split the sibling script already uses, restated here
        against the CANDIDATE k values rather than an exceedance ratio.

Usage:
    python scripts\\characterize_capacity_candidate_k_distribution.py ^
        --episode-dataset loan_episode_capacity_dataset.csv ^
        --weights-file capacity_artifacts\\capacity_score_weights.json ^
        --candidate-ks 0.03,0.04,0.05,0.06 ^
        --out-prefix capacity_candidate_k_characterization
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPOSURE_TIERS_UGX = [50_000, 100_000, 250_000, 350_000, 500_000, 750_000, 1_000_000]
N_DECILES_DEFAULT = 10
FUNDAMENTALS_FOR_SCORE = ["float_activity_value_1m", "commission"]
DEFAULT_WEIGHTS = {"float_activity_value_1m": 0.5, "commission": 0.5}
DEFAULT_CANDIDATE_KS = [0.03, 0.04, 0.05, 0.06]
DISCORDANCE_DECILE_THRESHOLD = 3

REQUIRED_COLS = ["agent_msisdn", "fundamentals_snapshot_date", "float_activity_value_1m", "commission"]


# ============================================================================
# Restated verbatim from derive_capacity_function_and_backtest_frontier.py
# (one-way scripts/ layering convention -- never imported).
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


def load_agent_period_fundamentals(episode_dataset_path: Path) -> pd.DataFrame:
    """usecols-whitelisted read (REQUIRED_COLS only -- no outcome,
    exposure, assigned_limit, or cal_pd column is ever named here),
    deduped to one row per (agent_msisdn, fundamentals_snapshot_date)
    agent-period unit."""
    key = ["agent_msisdn", "fundamentals_snapshot_date"]
    df = pd.read_csv(episode_dataset_path, usecols=lambda c: c in set(REQUIRED_COLS))
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: episode dataset is missing required column(s): {missing}")
    df = df[df["fundamentals_snapshot_date"].notna()]
    return df.groupby(key, sort=False)[["float_activity_value_1m", "commission"]].first().reset_index()


def assign_bands(df: pd.DataFrame, fund_col: str, n_bands: int = N_DECILES_DEFAULT) -> tuple:
    valid_mask = df[fund_col].notna() & (df[fund_col] > 0)
    out = pd.Series(np.nan, index=df.index, dtype=object)
    if not valid_mask.any():
        return out, []
    band_series, n_bins = _qcut_safe(df.loc[valid_mask, fund_col], n_bands)
    labels = [f"D{i + 1}" for i in range(n_bins)]
    out.loc[valid_mask] = band_series.values
    return out, labels


def load_weights(weights_file: Path | None, fundamentals: list[str]) -> dict:
    if weights_file is None:
        print(f"WARNING: no --weights-file given -- using DEFAULT_WEIGHTS {DEFAULT_WEIGHTS}. "
              f"Run derive_capacity_score_weights.py first to produce a governed artifact.")
        weights = dict(DEFAULT_WEIGHTS)
    else:
        import json
        meta = json.loads(Path(weights_file).read_text())
        weights = dict(meta["recommended_weights"])
        missing = [c for c in fundamentals if c not in weights]
        if missing:
            sys.exit(f"ERROR: --weights-file has no weight for {missing}.")
        weights = {c: weights[c] for c in fundamentals}

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        print(f"WARNING: weights sum to {total}, not 1.0 -- renormalizing.")
        weights = {c: w / total for c, w in weights.items()}
    return weights


def compute_capacity_ugx(df: pd.DataFrame, cols: list[str], weights: dict, k: float) -> pd.Series:
    """Capacity(F) = k * (exp(sum_i w_i*log1p(x_i)) - 1)."""
    valid_mask = pd.Series(True, index=df.index)
    for c in cols:
        valid_mask &= df[c].notna() & (df[c] >= 0)
    out = pd.Series(np.nan, index=df.index)
    log1p_df = np.log1p(df.loc[valid_mask, cols])
    s_log = sum(weights[c] * log1p_df[c] for c in cols)
    out.loc[valid_mask] = k * (np.exp(s_log) - 1.0)
    return out


def _discordance_label(row) -> str:
    """'capacity_not_estimable' (never 'unknown' -- see module docstring)
    means float_activity_value_1m <= 0 or commission <= 0 (or missing) for
    this agent-period, the SAME validity bar assign_bands already uses.
    This is categorically different from a measured, low-but-real capacity:
    we lack usable fundamentals, we did not measure a zero-capacity
    business."""
    if pd.isna(row["float_band_rank"]) or pd.isna(row["commission_band_rank"]):
        return "capacity_not_estimable"
    diff = abs(row["float_band_rank"] - row["commission_band_rank"])
    if diff <= 1:
        return "concordant"
    if diff >= DISCORDANCE_DECILE_THRESHOLD:
        return "discordant"
    return "middle"


# ============================================================================
# New: distribution characterization (never a backtest against any
# historical frontier).
# ============================================================================


def compute_distribution_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"n": 0, "median": np.nan, "p25": np.nan, "p75": np.nan,
                "mean": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(len(s)), "median": float(s.median()), "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)), "mean": float(s.mean()), "min": float(s.min()), "max": float(s.max()),
    }


def compute_tier_bucket_table(capacity_series: pd.Series, tiers: list[int]) -> pd.DataFrame:
    """One row per half-open UGX bucket (prev_tier, tier], plus a final
    open-ended '>max(tiers)' bucket. pct_in_bucket is the share of the
    (non-null) population strictly within that bucket; pct_at_or_below is
    the cumulative share at or below that bucket's upper edge. Purely
    descriptive binning against Analysis 4's own exposure ladder -- never
    a backtest against it."""
    s = capacity_series.dropna()
    n_total = len(s)
    rows = []
    boundaries = [0] + list(tiers)
    cumulative_n = 0
    if n_total == 0:
        for i in range(len(boundaries) - 1):
            rows.append({"tier_bucket_upper_ugx": boundaries[i + 1], "n_in_bucket": 0,
                         "pct_in_bucket": np.nan, "pct_at_or_below": np.nan})
        rows.append({"tier_bucket_upper_ugx": f">{tiers[-1]}", "n_in_bucket": 0,
                     "pct_in_bucket": np.nan, "pct_at_or_below": np.nan})
        return pd.DataFrame(rows)

    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        # The first bucket's lower edge is inclusive (capacity_ugx_at_k is
        # never negative, and a capacity of EXACTLY 0.0 -- e.g. both
        # fundamentals at 0 -- is a legitimate value that must land in a
        # bucket, not silently fall through every condition).
        in_bucket = (s >= lo) if i == 0 else (s > lo)
        n_in_bucket = int((in_bucket & (s <= hi)).sum())
        cumulative_n += n_in_bucket
        rows.append({
            "tier_bucket_upper_ugx": hi, "n_in_bucket": n_in_bucket,
            "pct_in_bucket": 100.0 * n_in_bucket / n_total, "pct_at_or_below": 100.0 * cumulative_n / n_total,
        })
    n_above = int((s > tiers[-1]).sum())
    cumulative_n += n_above
    rows.append({
        "tier_bucket_upper_ugx": f">{tiers[-1]}", "n_in_bucket": n_above,
        "pct_in_bucket": 100.0 * n_above / n_total, "pct_at_or_below": 100.0 * cumulative_n / n_total,
    })
    return pd.DataFrame(rows)


def compute_decile_table(df: pd.DataFrame, band_col: str, capacity_col: str) -> pd.DataFrame:
    rows = []
    for band, g in df.groupby(band_col, sort=False):
        stats = compute_distribution_stats(g[capacity_col])
        stats["n"] = len(g)  # true group size -- never silently shrunk by a dropna() inside the stats helper.
        rows.append({"band": band, **stats})
    rows.sort(key=lambda r: int(str(r["band"])[1:]))
    return pd.DataFrame(rows)


def compute_concordance_table(df: pd.DataFrame, capacity_col: str) -> pd.DataFrame:
    """For the capacity_not_estimable segment, capacity_col is NaN for
    every row by construction (see main()'s capacity_estimable masking),
    so median/p25/p75/mean/min/max correctly come back NaN -- but n must
    still report the TRUE group size, not 0, or this population's
    existence silently disappears from the table."""
    labelled = df.assign(concordance=df.apply(_discordance_label, axis=1))
    rows = []
    for concordance, g in labelled.groupby("concordance", sort=False):
        stats = compute_distribution_stats(g[capacity_col])
        stats["n"] = len(g)
        rows.append({"concordance": concordance, **stats})
    order = {"concordant": 0, "middle": 1, "discordant": 2, "capacity_not_estimable": 3}
    rows.sort(key=lambda r: order.get(r["concordance"], 99))
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dataset", type=Path, default=Path("loan_episode_capacity_dataset.csv"))
    ap.add_argument("--weights-file", type=Path, default=None)
    ap.add_argument("--candidate-ks", type=str, default=",".join(str(k) for k in DEFAULT_CANDIDATE_KS))
    ap.add_argument("--n-bands", type=int, default=N_DECILES_DEFAULT)
    ap.add_argument("--out-prefix", type=str, default="capacity_candidate_k_characterization")
    args = ap.parse_args(argv)

    if not args.episode_dataset.exists():
        sys.exit(f"ERROR: {args.episode_dataset} not found.")

    candidate_ks = sorted({float(x) for x in args.candidate_ks.split(",") if x.strip()})

    print(f"\n{'#' * 100}\nNOT A BACKTEST\n{'#' * 100}")
    print(
        "This script does not join against, or re-examine, any historical exposure frontier -- that\n"
        "question (at what k does Capacity(F) start exceeding the PD-comparable historical frontier)\n"
        "is answered by derive_capacity_function_and_backtest_frontier.py and is FROZEN. This script\n"
        "asks a different question: what does each candidate k actually imply for proposed capacity,\n"
        "in absolute UGX terms and relative to business scale."
    )
    print(f"\n{'#' * 100}\nC3/PD COMPLETELY OUT OF SCOPE\n{'#' * 100}")
    print(
        "No cal_pd column is read, no shadow/C3 multiplier is applied. L_full = Capacity_k(F) x\n"
        "M_C3(PD) is a separate, later, already-planned step -- this script only ever characterizes\n"
        "Capacity_k(F) in isolation."
    )
    print(f"\n{'#' * 100}\nCANDIDATE k's ARE A POLICY-CONSEQUENCE CHARACTERIZATION, NOT A SEARCH\n{'#' * 100}")
    print(f"Candidate k values: {candidate_ks}. This script never selects, recommends, or optimizes a k.")

    weights = load_weights(args.weights_file, FUNDAMENTALS_FOR_SCORE)
    print(f"\nWeights in use: {weights}")

    df = load_agent_period_fundamentals(args.episode_dataset)
    print(f"{len(df):,} agent-period unit(s) across {df['agent_msisdn'].nunique():,} agent(s).")

    df["float_band"], float_labels = assign_bands(df, "float_activity_value_1m", args.n_bands)
    df["commission_band"], commission_labels = assign_bands(df, "commission", args.n_bands)
    df["float_band_rank"] = df["float_band"].map(lambda b: np.nan if pd.isna(b) else int(str(b)[1:]))
    df["commission_band_rank"] = df["commission_band"].map(lambda b: np.nan if pd.isna(b) else int(str(b)[1:]))

    df["capacity_ugx_at_k1"] = compute_capacity_ugx(df, FUNDAMENTALS_FOR_SCORE, weights, 1.0)

    # CAPACITY_NOT_ESTIMABLE, never Capacity=0: an agent-period with
    # float_activity_value_1m<=0 or commission<=0 (the SAME validity bar
    # assign_bands already uses) did not have its economic capacity
    # measured as zero -- we simply lack usable fundamentals for it. Mask
    # capacity to NaN for these rows so every downstream table (overall,
    # decile, tier-bucket, concordance) excludes them from numeric stats
    # by construction, rather than contaminating those stats with a
    # formula artifact near zero.
    capacity_estimable = (df["float_activity_value_1m"] > 0) & (df["commission"] > 0)
    n_not_estimable = int((~capacity_estimable).sum())
    df.loc[~capacity_estimable, "capacity_ugx_at_k1"] = np.nan
    print(f"\n{n_not_estimable:,} of {len(df):,} agent-period(s) ({100.0 * n_not_estimable / len(df):.2f}%) "
          f"are CAPACITY_NOT_ESTIMABLE (float_activity_value_1m<=0 or commission<=0) -- excluded from every "
          f"numeric capacity table below, reported with their true count and NaN stats where applicable, "
          f"never as a fabricated capacity of 0.")

    overall_rows, float_decile_rows, commission_decile_rows = [], [], []
    tier_bucket_rows, concordance_rows = [], []

    for k in candidate_ks:
        df["capacity_ugx_at_k"] = df["capacity_ugx_at_k1"] * k

        overall_rows.append({"k": k, **compute_distribution_stats(df["capacity_ugx_at_k"])})

        ft = compute_decile_table(df.dropna(subset=["float_band"]), "float_band", "capacity_ugx_at_k")
        ft.insert(0, "k", k)
        float_decile_rows.append(ft)

        ct = compute_decile_table(df.dropna(subset=["commission_band"]), "commission_band", "capacity_ugx_at_k")
        ct.insert(0, "k", k)
        commission_decile_rows.append(ct)

        tb = compute_tier_bucket_table(df["capacity_ugx_at_k"], EXPOSURE_TIERS_UGX)
        tb.insert(0, "k", k)
        tier_bucket_rows.append(tb)

        cc = compute_concordance_table(df, "capacity_ugx_at_k")
        cc.insert(0, "k", k)
        concordance_rows.append(cc)

    overall_df = pd.DataFrame(overall_rows)
    float_decile_df = pd.concat(float_decile_rows, ignore_index=True)
    commission_decile_df = pd.concat(commission_decile_rows, ignore_index=True)
    tier_bucket_df = pd.concat(tier_bucket_rows, ignore_index=True)
    concordance_df = pd.concat(concordance_rows, ignore_index=True)

    with pd.option_context("display.width", 180, "display.float_format", "{:.3f}".format):
        print("\n-- Overall capacity_ugx_at_k distribution, by k --")
        print(overall_df.to_string(index=False))
        print("\n-- By float decile --")
        print(float_decile_df.to_string(index=False))
        print("\n-- By commission decile --")
        print(commission_decile_df.to_string(index=False))
        print("\n-- Tier-bucket proportions (proportion of capacity_ugx_at_k below each exposure tier) --")
        print(tier_bucket_df.to_string(index=False))
        print("\n-- By concordance (concordant / middle / discordant float-commission decile pairing) --")
        print(concordance_df.to_string(index=False))

    overall_df.to_csv(f"{args.out_prefix}_overall_stats.csv", index=False)
    float_decile_df.to_csv(f"{args.out_prefix}_by_float_decile.csv", index=False)
    commission_decile_df.to_csv(f"{args.out_prefix}_by_commission_decile.csv", index=False)
    tier_bucket_df.to_csv(f"{args.out_prefix}_tier_bucket_proportions.csv", index=False)
    concordance_df.to_csv(f"{args.out_prefix}_by_concordance.csv", index=False)
    print(f"\nWrote {args.out_prefix}_overall_stats.csv, _by_float_decile.csv, _by_commission_decile.csv, "
          f"_tier_bucket_proportions.csv, _by_concordance.csv")


if __name__ == "__main__":
    main()
