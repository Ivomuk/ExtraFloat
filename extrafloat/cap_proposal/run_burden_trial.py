"""
extrafloat/cap_proposal/run_burden_trial.py
===========================================
Side-by-side trial of the "burden" limit design (limit_design.py) against the
live engine, on the same agents. Read-only with respect to the live code:

  * the engine source is NOT modified -- limit_design.py swaps two engine
    functions in for the duration of its own call and restores them, and this
    runner checks they are restored after every design run;
  * the CURRENT limit is produced by run_extrafloat_limit_engine() exactly as
    run_engine.py runs it in production (no loan-history snapshot, no PD,
    no wrapper corrections) unless --current-with-loan-history is given;
  * nothing is written outside --out-dir.

Unlike run_engine_wrapper.py (the local Windows pipeline with hard-coded
F:\\ paths, PD join and coverage corrections), this runner needs only the
repo itself and the warehouse extracts, so it can be run anywhere.

Usage
-----
    # real extracts (loan history snapshot is required by the design's ladder)
    python -m extrafloat.cap_proposal.run_burden_trial \\
        --transaction  data/agent_profile_snapshot.csv \\
        --borrower     data/borrower_history.csv \\
        --loan-summary data/loan_summary.csv \\
        --loan-history data/loan_history_snapshot.csv \\
        --out-dir      output/burden_trial

    # every candidate settings file in extrafloat/cap_proposal/ in one go
    python -m extrafloat.cap_proposal.run_burden_trial ... --all-candidates

    # dry run on synthetic agents (checks the plumbing, not the numbers)
    python -m extrafloat.cap_proposal.run_burden_trial --synthetic 5000 --out-dir output/burden_trial_synth

Outputs (per settings file, in --out-dir/<settings name>/)
------------------------------------------------------------
    agents.csv     one row per agent: current vs proposed limit and the design audit columns
    summary.md     population totals, change distribution, stage and binding-constraint tables
    summary.json   the same figures, machine-readable
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import extrafloat.engine.run_extrafloat_limit_engine as R
from extrafloat.cap_proposal import limit_design as ld
from extrafloat.engine.extrafloat_limit_engine_features import build_extrafloat_limit_engine_features
from extrafloat.io.extrafloat_data_loaders import (
    load_borrower_limit_features,
    load_loan_history_snapshot_features,
    load_loan_summary_recent_features,
    load_transaction_capacity_features,
)

logger = logging.getLogger("burden_trial")

HERE = Path(__file__).resolve().parent
DEFAULT_SETTINGS = HERE / "limit_design_proposal.json"
LIVE_CEILING = 1_000_000.0

# Columns the design reads that the live engine does not need, by design setting.
_LADDER_COLS = ["commission", "closed_loan_count", "lifetime_default_rate", "on_time_repayment_rate",
                "lifetime_worst_days_to_principal_cure", "consecutive_on_time_loans", "max_prior_loan_size"]
_ARREARS_COLS = {
    "legacy": ["active_loan_days_aging_at_snapshot", "anomaly_open_at_snapshot"],
    "snapshot": ["current_dpd_at_snapshot", "written_off_at_snapshot"],
    "any_loan": ["worst_dpd_material_any_loan_at_snapshot", "written_off_at_snapshot"],
}

AGENT_COLUMNS = [
    "msisdn", "limit_current", "limit_proposed", "limit_change", "change_direction",
    "design_stage", "design_k_applied", "design_arrears_rule", "design_binding_constraint",
    "design_monthly_commission", "design_ladder_cap", "design_limit_before_governor",
    "design_governor_cap", "design_limit_at_reference_ceiling", "max_prior_loan_size",
    "is_thin_file", "risk_tier",
    "design_agent_category", "design_agent_tier_max", "design_agent_tier_guard",
    "design_limit_before_tier_guard", "design_agent_tier_guard_binding",
]
CATEGORY_ORDER = ["Below Threshold", "new bronze", "bronze", "silver", "gold", "platinum", "titanium", "diamond"]
GUARD_MODES = ("enforce", "enforce_below_diamond", "off")


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------


def load_features(transaction: str, borrower: str, loan_summary: str, loan_history: str | None,
                  as_of_date: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build engine features twice from the same extracts, with the live engine's own builder:
    (production_features, design_features). Production is what run_engine.py builds today
    (no loan history); the design needs the loan history snapshot for its ladder and arrears rules."""
    txn = load_transaction_capacity_features(transaction)
    bor = load_borrower_limit_features(borrower)
    ls = load_loan_summary_recent_features(loan_summary)
    lh = load_loan_history_snapshot_features(loan_history) if loan_history else None
    asof = pd.Timestamp(as_of_date) if as_of_date else None
    production = build_extrafloat_limit_engine_features(bor, txn, ls, None, as_of_date=asof)
    design = build_extrafloat_limit_engine_features(bor, txn, ls, lh, as_of_date=asof)
    return production, design


def adapt_for_design(features: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Fill design inputs this repo's feature builder names differently, then check every input exists.

    The only adaptation: the repo's borrower history carries the worst cure time in HOURS
    (lifetime_worst_hours_to_principal_cure); the design reads DAYS. Converted as hours / 24.
    Anything else missing is an error -- the design must never silently fill an input."""
    f = features.copy()
    if "lifetime_worst_days_to_principal_cure" not in f.columns and "lifetime_worst_hours_to_principal_cure" in f.columns:
        f["lifetime_worst_days_to_principal_cure"] = f["lifetime_worst_hours_to_principal_cure"] / 24.0
        logger.info("design input: lifetime_worst_days_to_principal_cure = lifetime_worst_hours_to_principal_cure / 24")
    if (settings["ladder"].get("clean_on_time_source") == "lifetime" and "lifetime_on_time_rate" not in f.columns
            and "lifetime_on_time_24h_rate" in f.columns):
        # same 24h on-time definition the engine's on_time_repayment_rate uses (from prior_on_time_24h_rate),
        # over every matured loan instead of loans before the latest one
        f["lifetime_on_time_rate"] = f["lifetime_on_time_24h_rate"]
        logger.info("design input: lifetime_on_time_rate = lifetime_on_time_24h_rate")
    needed = list(_LADDER_COLS) + _ARREARS_COLS[settings["arrears"].get("source", "legacy")]
    if settings["ladder"].get("clean_on_time_source") == "lifetime":
        needed.append("lifetime_on_time_rate")
    if (settings.get("reactivation") or {}).get("enabled"):
        needed.append("days_since_last_repayment_any_loan_at_snapshot")
    missing = [c for c in needed if c not in f.columns]
    if missing:
        raise ValueError(
            f"burden design inputs missing from the features: {missing}. "
            "All of them come from the loan history snapshot (--loan-history); current_dpd_at_snapshot, "
            "written_off_at_snapshot, worst_dpd_material_any_loan_at_snapshot and "
            "days_since_last_repayment_any_loan_at_snapshot only from the newer extract, not from "
            "data/loan_history_snapshot_query.txt."
        )
    return f


# -----------------------------------------------------------------------------
# Runs
# -----------------------------------------------------------------------------


def run_current(features: pd.DataFrame) -> pd.DataFrame:
    """The live engine, unchanged."""
    return R.run_extrafloat_limit_engine(features)


def run_proposed(features: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """The burden design, with a check that the engine functions it swaps are put back."""
    originals = (R.compute_capacity_cap, R.compute_risk_cap)
    try:
        return ld.run_burden_design(adapt_for_design(features, settings), settings)
    finally:
        if (R.compute_capacity_cap, R.compute_risk_cap) != originals:
            R.compute_capacity_cap, R.compute_risk_cap = originals
            raise RuntimeError("limit_design left the live engine patched -- restored; results discarded.")


def compare(current: pd.DataFrame, proposed: pd.DataFrame) -> pd.DataFrame:
    """One row per agent: current vs proposed limit plus the design's audit columns."""
    cur = current[["msisdn", "assigned_limit"]].rename(columns={"assigned_limit": "limit_current"})
    cur["msisdn"] = cur["msisdn"].astype(str)
    prop = proposed.rename(columns={"assigned_limit": "limit_proposed"}).copy()
    prop["msisdn"] = prop["msisdn"].astype(str)
    out = prop.merge(cur, on="msisdn", how="left", validate="one_to_one")
    out["limit_change"] = out["limit_proposed"] - out["limit_current"]
    out["change_direction"] = np.select([out["limit_current"].isna(), out["limit_change"] > 0.5, out["limit_change"] < -0.5],
                                        ["not scored by current", "up", "down"], "same")
    return out[[c for c in AGENT_COLUMNS if c in out.columns]]


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------


def category_table(agents: pd.DataFrame) -> pd.DataFrame:
    """Per agent category: the category maximum and how many proposed limits sit above it."""
    a = agents.assign(over=agents["limit_proposed"] > agents["design_agent_tier_max"] + 1,
                      excess=(agents["limit_proposed"] - agents["design_agent_tier_max"]).clip(lower=0))
    g = a.groupby("design_agent_category", dropna=False)
    t = pd.DataFrame({
        "agents": g.size(),
        "category_max": g["design_agent_tier_max"].max(),
        "max_current": g["limit_current"].max(),
        "max_proposed": g["limit_proposed"].max(),
        "total_current_m": g["limit_current"].sum() / 1e6,
        "total_proposed_m": g["limit_proposed"].sum() / 1e6,
        "above_category_max": g["over"].sum().astype(int),
        "excess_above_max_m": g["excess"].sum() / 1e6,
        "capped_by_guard": g["design_agent_tier_guard_binding"].sum().astype(int),
    })
    order = [c for c in CATEGORY_ORDER if c in t.index] + [c for c in t.index if c not in CATEGORY_ORDER]
    return t.reindex(order)


GUARD_LABELS = {
    "enforce": "category_enforced",
    "off": "category_not_enforced",
    "enforce_below_diamond": "enforce_below_diamond",
}


def guard_comparison(agents_by_mode: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Side by side for several agent_tier_guard modes of ONE settings file.
    Returns (per-agent wide table, per-category totals)."""
    modes = [m for m in ("enforce", "off", "enforce_below_diamond") if m in agents_by_mode]
    first = agents_by_mode[modes[0]]
    wide = first[["msisdn", "design_agent_category", "design_agent_tier_max", "design_stage", "limit_current"]].copy()
    wide = wide.rename(columns={"design_agent_category": "agent_category", "design_agent_tier_max": "category_max",
                                "design_stage": "experience_stage"})
    for m in modes:
        lim = agents_by_mode[m].set_index("msisdn")["limit_proposed"]
        wide[f"limit_{GUARD_LABELS[m]}"] = wide["msisdn"].map(lim).values
    if "off" in modes:
        wide["above_category_max_if_not_enforced"] = (wide["limit_category_not_enforced"] > wide["category_max"] + 1).astype(int)

    order = [c for c in CATEGORY_ORDER if c in set(wide["agent_category"])]
    order += [c for c in wide["agent_category"].dropna().unique() if c not in order]
    g = wide.groupby("agent_category", dropna=False)
    cat = pd.DataFrame({"agents": g.size(), "category_max": g["category_max"].max(),
                        "current_total_m": g["limit_current"].sum() / 1e6})
    for m in modes:
        col = f"limit_{GUARD_LABELS[m]}"
        cat[f"{GUARD_LABELS[m]}_total_m"] = g[col].sum() / 1e6
        cat[f"{GUARD_LABELS[m]}_max"] = g[col].max()
    if "off" in modes:
        cat["above_max_if_not_enforced"] = g["above_category_max_if_not_enforced"].sum()
    cat = cat.reindex(order)
    total = cat.sum(numeric_only=True)
    for c in cat.columns:
        if c.endswith("_max") or c == "category_max":
            total[c] = cat[c].max()
    cat.loc["ALL AGENTS"] = total
    return wide, cat


def render_guard_markdown(cat: pd.DataFrame, settings_name: str, synthetic: bool) -> str:
    note = ("\n> **Synthetic data.** Checks the mechanics only; says nothing about the real population.\n"
            if synthetic else "")
    t = cat.reset_index().rename(columns={"index": "agent_category"})
    return (f"# Agent-category guard comparison: `{settings_name}`\n{note}\n"
            "Same agents, same design; only the agent-category guard differs. Totals in UGX millions.\n\n"
            + _md_table(t.round(1).to_dict("records")))


def _group_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, dropna=False)
    t = pd.DataFrame({
        "agents": g.size(),
        "total_current_m": g["limit_current"].sum() / 1e6,
        "total_proposed_m": g["limit_proposed"].sum() / 1e6,
        "median_current": g["limit_current"].median(),
        "median_proposed": g["limit_proposed"].median(),
        "share_up": g["change_direction"].apply(lambda s: (s == "up").mean()),
        "share_down": g["change_direction"].apply(lambda s: (s == "down").mean()),
    })
    t["share_of_agents"] = t["agents"] / len(df)
    return t.sort_values("agents", ascending=False)


def summarize(agents: pd.DataFrame, settings: dict, settings_name: str) -> dict:
    cur, prop = agents["limit_current"].fillna(0), agents["limit_proposed"]

    def pop(s: pd.Series) -> dict:
        return {
            "total_m": round(float(s.sum()) / 1e6, 1),
            "mean": round(float(s.mean()), 0),
            "median": round(float(s.median()), 0),
            "p90": round(float(s.quantile(0.9)), 0),
            "max": round(float(s.max()), 0),
            "zero": int((s <= 0).sum()),
            "above_1m": int((s > LIVE_CEILING).sum()),
        }

    total_cur = float(cur.sum())
    return {
        "settings": settings_name,
        "agents": int(len(agents)),
        "current": pop(cur),
        "proposed": pop(prop),
        "total_change_pct": round(100.0 * (float(prop.sum()) - total_cur) / total_cur, 1) if total_cur > 0 else None,
        "direction": agents["change_direction"].value_counts().to_dict(),
        "by_stage": _group_table(agents, "design_stage").round(3).reset_index().to_dict("records"),
        "by_binding_constraint": _group_table(agents, "design_binding_constraint").round(3).reset_index().to_dict("records"),
        "by_agent_category": category_table(agents).round(3).reset_index().to_dict("records"),
        "above_category_max": int((agents["limit_proposed"] > agents["design_agent_tier_max"] + 1).sum()),
        "key_settings": {
            "capacity_ceiling_multiple": settings["capacity_ceiling_multiple"],
            "engine_ceiling": settings["engine_ceiling"],
            "governor": {k: v for k, v in settings["governor"].items() if not str(k).startswith("_")},
            "ladder": {k: v for k, v in settings["ladder"].items() if not str(k).startswith("_")},
            "arrears_source": settings["arrears"].get("source", "legacy"),
            "reactivation_enabled": bool((settings.get("reactivation") or {}).get("enabled")),
            "pd_tier_cutoffs": (settings.get("pd") or {}).get("tier_pd_cutoffs"),
            "agent_tier_guard": settings.get("agent_tier_guard", "enforce"),
        },
    }


def _md_table(records: list[dict]) -> str:
    if not records:
        return "_none_\n"
    cols = list(records[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in records:
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:,.3f}" if isinstance(v, float) and abs(v) < 10 else f"{v:,.0f}" if isinstance(v, (int, float)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_markdown(s: dict, synthetic: bool) -> str:
    rows = [{"metric": k, "current": s["current"][k], "proposed": s["proposed"][k]} for k in s["current"]]
    note = ("\n> **Synthetic data.** These figures check that the design runs and that its rules fire as written. "
            "They say nothing about the real population.\n") if synthetic else ""
    return (
        f"# Burden design trial: `{s['settings']}`\n{note}\n"
        f"Agents scored: **{s['agents']:,}**. Total limit change vs the live engine: **{s['total_change_pct']}%**.\n\n"
        f"## Population\n\n{_md_table(rows)}\n"
        f"Direction of change: {s['direction']}\n\n"
        f"## By agent category (guard: `{s['key_settings']['agent_tier_guard']}`)\n\n"
        f"Proposed limits above the agent's category maximum: **{s['above_category_max']:,}**.\n\n"
        f"{_md_table(s['by_agent_category'])}\n"
        f"## By experience stage\n\n{_md_table(s['by_stage'])}\n"
        f"## By binding constraint\n\n{_md_table(s['by_binding_constraint'])}\n"
        f"## Settings used\n\n```json\n{json.dumps(s['key_settings'], indent=2)}\n```\n"
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _settings_files(args) -> list[Path]:
    if args.all_candidates:
        return sorted(HERE.glob("limit_design_*.json"))
    return [Path(p) for p in (args.design_config or [DEFAULT_SETTINGS])]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Trial the burden limit design against the live engine (live code untouched).")
    src = ap.add_argument_group("inputs (warehouse extracts) -- or use --synthetic")
    src.add_argument("--transaction")
    src.add_argument("--borrower")
    src.add_argument("--loan-summary")
    src.add_argument("--loan-history", help="Loan history snapshot. Required by the design's experience ladder and arrears rules.")
    src.add_argument("--synthetic", type=int, metavar="N", help="Generate N synthetic agents instead of reading extracts.")
    ap.add_argument("--as-of-date", help="As-of date for feature engineering (YYYY-MM-DD); default today.")
    ap.add_argument("--design-config", nargs="+", help=f"Settings JSON(s). Default {DEFAULT_SETTINGS.name}.")
    ap.add_argument("--all-candidates", action="store_true", help="Run every limit_design_*.json in extrafloat/cap_proposal/.")
    ap.add_argument("--agent-tier-guard", nargs="+", choices=GUARD_MODES,
                    help="Override the settings' agent_tier_guard; several values run each settings file once per mode.")
    ap.add_argument("--current-with-loan-history", action="store_true",
                    help="Feed the loan history to the CURRENT run too (activates the engine's unresolved-loan haircut). "
                         "Default off = the engine exactly as run_engine.py runs it in production.")
    ap.add_argument("--out-dir", default="output/burden_trial")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        from extrafloat.cap_proposal.synthetic_inputs import SNAPSHOT_DT, make_synthetic_inputs

        raw_dir = out_dir / "synthetic_inputs"
        raw_dir.mkdir(exist_ok=True)
        for name, frame in make_synthetic_inputs(args.synthetic).items():
            frame.to_csv(raw_dir / f"{name}.csv", index=False)
        paths = {k: str(raw_dir / f"{k}.csv") for k in ("transaction", "borrower", "loan_summary", "loan_history")}
        as_of = args.as_of_date or SNAPSHOT_DT
    else:
        need = {"--transaction": args.transaction, "--borrower": args.borrower, "--loan-summary": args.loan_summary,
                "--loan-history": args.loan_history}
        missing = [k for k, v in need.items() if not v]
        if missing:
            ap.error(f"missing {', '.join(missing)} (or use --synthetic N)")
        paths = {"transaction": args.transaction, "borrower": args.borrower,
                 "loan_summary": args.loan_summary, "loan_history": args.loan_history}
        as_of = args.as_of_date

    production, design_features = load_features(**paths, as_of_date=as_of)
    current = run_current(design_features if args.current_with_loan_history else production)

    index = []
    runs = []
    for path in _settings_files(args):
        base = ld.load_design_config(path)
        if not path.exists():
            logger.warning("settings file %s not found -- using limit_design.py DEFAULTS", path)
        if args.agent_tier_guard:
            for mode in args.agent_tier_guard:
                runs.append((f"{path.stem}__guard_{mode}" if len(args.agent_tier_guard) > 1 else path.stem,
                             path.stem, {**base, "agent_tier_guard": mode}))
        else:
            runs.append((path.stem, path.stem, base))

    by_settings: dict[str, dict[str, pd.DataFrame]] = {}
    for name, stem, settings in runs:
        try:
            proposed = run_proposed(design_features, settings)
        except ValueError as exc:
            logger.error("%s: SKIPPED -- %s", name, exc)
            index.append({"settings": name, "skipped": str(exc)})
            continue
        agents = compare(current, proposed)
        by_settings.setdefault(stem, {})[settings.get("agent_tier_guard", "enforce")] = agents
        summary = summarize(agents, settings, name)
        run_dir = out_dir / name
        run_dir.mkdir(exist_ok=True)
        agents.to_csv(run_dir / "agents.csv", index=False)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        (run_dir / "summary.md").write_text(render_markdown(summary, synthetic=bool(args.synthetic)))
        index.append({"settings": name, "agent_tier_guard": settings.get("agent_tier_guard", "enforce"), "agents": summary["agents"],
                      "total_current_m": summary["current"]["total_m"], "total_proposed_m": summary["proposed"]["total_m"],
                      "total_change_pct": summary["total_change_pct"], "above_1m": summary["proposed"]["above_1m"],
                      "zero_proposed": summary["proposed"]["zero"], "above_category_max": summary["above_category_max"]})
        logger.info("%s: total %.1fM -> %.1fM (%s%%), above 1M: %d", name, summary["current"]["total_m"],
                    summary["proposed"]["total_m"], summary["total_change_pct"], summary["proposed"]["above_1m"])

    for stem, agents_by_mode in by_settings.items():
        if len(agents_by_mode) < 2:
            continue
        wide, cat = guard_comparison(agents_by_mode)
        wide.to_csv(out_dir / f"{stem}__guard_comparison_agents.csv", index=False)
        cat.reset_index().rename(columns={"index": "agent_category"}).to_csv(
            out_dir / f"{stem}__guard_comparison_by_category.csv", index=False)
        (out_dir / f"{stem}__guard_comparison.md").write_text(render_guard_markdown(cat, stem, bool(args.synthetic)))
        print(f"\n== {stem}: agent-category guard comparison (UGX millions) ==")
        print(cat.round(1).to_string())

    idx = pd.DataFrame(index)
    idx.to_csv(out_dir / "index.csv", index=False)
    ran = idx[idx["skipped"].isna()].drop(columns="skipped") if "skipped" in idx else idx
    print(ran.to_string(index=False))
    if "skipped" in idx:
        for _, row in idx[idx["skipped"].notna()].iterrows():
            print(f"SKIPPED {row['settings']}: {row['skipped']}")
    print(f"\nWritten to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
