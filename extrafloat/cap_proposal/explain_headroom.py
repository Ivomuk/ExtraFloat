"""
extrafloat/cap_proposal/explain_headroom.py
===========================================
Why is the new limit at or above what agents borrowed, and who gets the room above it? (read-only)

Joins the disbursement comparison (compare_limits_vs_disbursed) with the trial's agents.csv, which records
for every agent the rules that set the limit, and reports for agents whose August disbursement fits under
the new limit ("covered"):

  * headroom = new limit - disbursed, in total and per agent
  * by risk tier and by experience stage: how many agents, how much headroom, their PD
  * what the limit is anchored to: limit / monthly commission (the ladder multiple) and
    limit / largest past loan (the governor)
  * which rule set the limit (design_binding_constraint)

    python -m extrafloat.cap_proposal.explain_headroom ^
        --comparison output\\burden_trial\\vs_disbursed_pd_3_12_30\\limits_vs_disbursed_agents.csv ^
        --agents output\\burden_trial\\limit_design_candidate_pd_3_12_30__guard_enforce\\agents.csv ^
        --out-dir output\\burden_trial\\headroom_pd_3_12_30

Writes headroom_by_tier.csv, headroom_by_stage.csv, headroom_by_rule.csv, headroom_by_category.csv,
headroom_agents.csv (one row per covered agent; contains phone numbers -- keep internal) and headroom.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIMIT_COL = "limit_category_enforced"
AGENT_COLS = ["msisdn", "risk_tier", "cal_pd", "design_stage", "design_k_applied", "design_monthly_commission",
              "max_prior_loan_size", "design_binding_constraint", "design_agent_category"]


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.replace(r"[^0-9]", "", regex=True)


def build(comparison: Path, agents: Path, limit_col: str = LIMIT_COL) -> pd.DataFrame:
    d = pd.read_csv(comparison, dtype={"msisdn": str}, low_memory=False)
    a = pd.read_csv(agents, dtype={"msisdn": str}, low_memory=False)
    if limit_col not in d.columns:
        raise ValueError(f"{comparison} has no {limit_col} column")
    keep = [c for c in AGENT_COLS if c in a.columns]
    d["msisdn"], a["msisdn"] = _norm(d["msisdn"]), _norm(a["msisdn"])
    m = d.merge(a[keep], on="msisdn", how="left", suffixes=("", "_design"))
    m["new_limit"] = m[limit_col].astype(float)
    m["covered"] = m["disbursed_amount"] <= m["new_limit"] + 0.5
    m["headroom"] = (m["new_limit"] - m["disbursed_amount"]).clip(lower=0)
    m["limit_over_disbursed"] = np.where(m["disbursed_amount"] > 0, m["new_limit"] / m["disbursed_amount"], np.nan)
    mc = m.get("design_monthly_commission")
    m["limit_over_monthly_commission"] = np.where(mc > 0, m["new_limit"] / mc, np.nan) if mc is not None else np.nan
    mp = m.get("max_prior_loan_size")
    m["limit_over_largest_past_loan"] = np.where(mp > 0, m["new_limit"] / mp, np.nan) if mp is not None else np.nan
    return m


def group_table(m: pd.DataFrame, by: str) -> pd.DataFrame:
    if by not in m.columns:
        return pd.DataFrame()
    g = m.groupby(by, dropna=False)
    cov = m[m["covered"]].groupby(by, dropna=False)
    t = pd.DataFrame({
        "agents": g.size(),
        "covered_share": g["covered"].mean(),
        "median_disbursed": g["disbursed_amount"].median(),
        "median_new_limit": g["new_limit"].median(),
        "median_limit_over_disbursed": cov["limit_over_disbursed"].median(),
        "headroom_total_m": cov["headroom"].sum() / 1e6,
        "median_limit_over_monthly_commission": cov["limit_over_monthly_commission"].median(),
        "median_limit_over_largest_past_loan": cov["limit_over_largest_past_loan"].median(),
    })
    if "cal_pd" in m.columns:
        t["median_pd"] = g["cal_pd"].median()
    total = t["headroom_total_m"].sum()
    t["share_of_headroom"] = t["headroom_total_m"] / total if total else np.nan
    return t


def _fmt(t: pd.DataFrame) -> pd.DataFrame:
    out = t.copy().astype(object)
    for c in t.columns:
        for i in t.index:
            v = t.loc[i, c]
            if pd.isna(v):
                out.loc[i, c] = ""
            elif c in ("covered_share", "share_of_headroom", "median_pd"):
                out.loc[i, c] = f"{v:.1%}"
            elif c.startswith("median_limit_over"):
                out.loc[i, c] = f"{v:.2f}"
            elif c == "headroom_total_m":
                out.loc[i, c] = f"{v:,.1f}"
            else:
                out.loc[i, c] = f"{v:,.0f}"
    return out


def _md(t: pd.DataFrame) -> str:
    if t.empty:
        return "_not available_\n"
    f = _fmt(t).reset_index()
    cols = list(f.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in f.itertuples(index=False)]) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Who gets the room above what they borrowed, and why (read-only).")
    ap.add_argument("--comparison", required=True, help="limits_vs_disbursed_agents.csv from compare_limits_vs_disbursed")
    ap.add_argument("--agents", required=True, help="agents.csv from the trial run of the same settings (guard enforce)")
    ap.add_argument("--limit-col", default=LIMIT_COL)
    ap.add_argument("--out-dir", default="output/burden_trial/headroom")
    args = ap.parse_args(argv)

    m = build(Path(args.comparison), Path(args.agents), args.limit_col)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cov = m[m["covered"]]
    tables = {
        "risk tier": ("risk_tier", "headroom_by_tier.csv"),
        "experience stage": ("design_stage", "headroom_by_stage.csv"),
        "rule that set the limit": ("design_binding_constraint", "headroom_by_rule.csv"),
        "agent category": ("agent_category", "headroom_by_category.csv"),
    }
    md = ["# Room above what agents borrowed: who gets it and why\n",
          f"Matched agents: **{len(m):,}**. Covered (new limit at or above August disbursement): **{len(cov):,} "
          f"({len(cov) / len(m):.1%})**. Total room above disbursed for covered agents: **{cov['headroom'].sum() / 1e6:,.1f}M**; "
          f"median limit / disbursed **{cov['limit_over_disbursed'].median():.2f}**.\n"]
    pd.set_option("display.width", 240)
    print(md[1].replace("**", ""))
    for title, (col, fname) in tables.items():
        t = group_table(m, col)
        if t.empty:
            continue
        t.to_csv(out / fname)
        md.append(f"## By {title}\n\n{_md(t)}")
        print(f"== By {title} ==")
        print(_fmt(t).to_string())
        print()
    agent_cols = ["msisdn", "agent_category", "risk_tier", "cal_pd", "design_stage", "design_k_applied",
                  "design_monthly_commission", "max_prior_loan_size", "design_binding_constraint", "disbursed_amount",
                  "new_limit", "headroom", "limit_over_disbursed", "limit_over_monthly_commission", "limit_over_largest_past_loan"]
    cov[[c for c in agent_cols if c in cov.columns]].sort_values("headroom", ascending=False).to_csv(out / "headroom_agents.csv", index=False)
    (out / "headroom.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
