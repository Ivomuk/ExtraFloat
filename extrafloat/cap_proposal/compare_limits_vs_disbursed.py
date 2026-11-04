"""
extrafloat/cap_proposal/compare_limits_vs_disbursed.py
======================================================
Compares the trial's limits with what agents ACTUALLY received.

Inputs
------
  --agents   <out-dir>/<settings>__guard_comparison_agents.csv   (from run_burden_trial.py with
             --agent-tier-guard enforce off enforce_below_diamond): one row per agent with
             limit_current, limit_category_enforced, limit_category_not_enforced, limit_enforce_below_diamond
  --disbursed monthly_disbursement_summary.csv (from scripts/build_monthly_disbursement_summary.py):
             msisdn, profile, month, disbursed_amount, n_transactions -- only agent/months where every
             disbursement was the SAME amount, so disbursed_amount is the loan size the agent took.

By default each agent is compared on their LATEST month in the file; --month YYYY-MM picks one month.

For every limit column it reports, per agent category and overall:
  * covered       disbursed_amount <= limit (the agent could still take what they took)
  * cut           disbursed_amount >  limit (the new limit is BELOW what the agent actually took),
                  with the UGX shortfall (disbursed - limit)
  * ratio bands   disbursed / limit, in the same bands as scripts/check_monthly_summary_vs_assigned_limit.py
  * headroom      limit / disbursed, median

Outputs (in --out-dir)
----------------------
  limits_vs_disbursed_agents.csv        one row per matched agent: limits, disbursed, ratios, cut flags
  limits_vs_disbursed_by_category.csv   coverage / cut / shortfall per category x limit
  limits_vs_disbursed_bands.csv         ratio-band distribution per limit
  limits_vs_disbursed.md                all of the above as readable tables

Read-only: never touches the engine or the trial outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIMIT_COLUMNS = {
    "limit_frozen": "frozen engine",
    "limit_current": "current engine",
    "limit_category_enforced": "category enforced",
    "limit_category_not_enforced": "category not enforced",
    "limit_enforce_below_diamond": "enforce below diamond",
    "limit_enforce_except_graduates": "enforce except graduates",
}
CATEGORY_ORDER = ["Below Threshold", "new bronze", "bronze", "silver", "gold", "platinum", "titanium", "diamond"]

# Same bands as scripts/check_monthly_summary_vs_assigned_limit.py (disbursed / limit).
BAND_EDGES = [-0.01, 0.25, 0.50, 0.75, 0.90, 1.10, 1.25, 1.50, 2.00, np.inf]
BAND_LABELS = [
    "0-25% (well under)", "25-50% (under)", "50-75% (under)",
    "75-90% (near, under)", "90-110% (agreement)", "110-125% (near, over)",
    "125-150% (over)", "150-200% (well over)", "200%+ (far over)",
]


def normalize_msisdn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.replace(r"[^0-9]", "", regex=True).replace("", pd.NA)


def _resolve(path: str) -> Path:
    """Accept a bare file name that lives in data/."""
    p = Path(path)
    if not p.exists() and not p.is_absolute() and (Path("data") / p).exists():
        return Path("data") / p
    if not p.exists():
        sys.exit(f"ERROR: file not found: {path} (also looked in data\\)")
    return p


def load_disbursed(path: Path, month: str | None) -> tuple[pd.DataFrame, list[str]]:
    d = pd.read_csv(path, dtype={"msisdn": str})
    d.columns = [c.strip() for c in d.columns]
    need = {"msisdn", "month", "disbursed_amount"}
    if not need <= set(d.columns):
        sys.exit(f"ERROR: {path} needs columns {sorted(need)}; has {list(d.columns)}")
    d["msisdn"] = normalize_msisdn(d["msisdn"])
    d["month"] = d["month"].astype(str).str[:7]
    d["disbursed_amount"] = pd.to_numeric(d["disbursed_amount"], errors="coerce")
    d = d.dropna(subset=["msisdn", "disbursed_amount"])
    months = sorted(d["month"].unique())
    if month:
        d = d[d["month"] == month]
        if d.empty:
            sys.exit(f"ERROR: no rows for --month {month}. Months in the file: {months}")
    else:
        d = d.sort_values(["msisdn", "month"]).drop_duplicates("msisdn", keep="last")
    keep = [c for c in ("msisdn", "month", "profile", "disbursed_amount", "n_transactions") if c in d.columns]
    return d[keep].drop_duplicates("msisdn", keep="last"), months


def build_agents(limits: pd.DataFrame, disbursed: pd.DataFrame) -> pd.DataFrame:
    a = limits.copy()
    a["msisdn"] = normalize_msisdn(a["msisdn"])
    m = a.merge(disbursed, on="msisdn", how="inner", validate="one_to_one")
    for col in LIMIT_COLUMNS:
        if col not in m.columns:
            continue
        lim = m[col].astype(float)
        m[f"{col}__ratio"] = np.where(lim > 0, m["disbursed_amount"] / lim, np.inf)   # disbursed / limit
        m[f"{col}__cut"] = (m["disbursed_amount"] > lim + 0.5).astype(int)
        m[f"{col}__shortfall"] = (m["disbursed_amount"] - lim).clip(lower=0)
    return m


def by_category(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cats = [c for c in CATEGORY_ORDER if c in set(m["agent_category"])] + \
           [c for c in m["agent_category"].dropna().unique() if c not in CATEGORY_ORDER] + ["ALL AGENTS"]
    for cat in cats:
        g = m if cat == "ALL AGENTS" else m[m["agent_category"] == cat]
        for col, label in LIMIT_COLUMNS.items():
            if col not in g.columns or g.empty:
                continue
            lim = g[col].astype(float)
            headroom = (lim / g["disbursed_amount"]).replace([np.inf, -np.inf], np.nan)
            rows.append({
                "agent_category": cat,
                "limit": label,
                "agents_matched": len(g),
                "median_disbursed": g["disbursed_amount"].median(),
                "median_limit": lim.median(),
                "median_limit_over_disbursed": headroom.median(),
                "covered_share": 1 - g[f"{col}__cut"].mean(),
                "cut_agents": int(g[f"{col}__cut"].sum()),
                "cut_share": g[f"{col}__cut"].mean(),
                "shortfall_total_m": g[f"{col}__shortfall"].sum() / 1e6,
                "zero_limit_but_disbursed": int(((lim <= 0) & (g["disbursed_amount"] > 0)).sum()),
            })
    return pd.DataFrame(rows)


def bands(m: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for col, label in LIMIT_COLUMNS.items():
        if col not in m.columns:
            continue
        b = pd.cut(m[f"{col}__ratio"], BAND_EDGES, labels=BAND_LABELS)
        out[label] = b.value_counts(normalize=True).reindex(BAND_LABELS).fillna(0)
    return pd.DataFrame(out).rename_axis("disbursed / limit")


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"{v:.1%}" if ("share" in c or c in LIMIT_COLUMNS.values()) else f"{v:,.2f}" if abs(v) < 10 else f"{v:,.0f}")
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare trial limits with actual disbursements.")
    ap.add_argument("--agents", default="output/burden_trial/limit_design_proposal__guard_comparison_agents.csv",
                    help="The trial's side-by-side per-agent file.")
    ap.add_argument("--disbursed", default="monthly_disbursement_summary.csv",
                    help="monthly_disbursement_summary.csv (a bare name is also looked for in data\\).")
    ap.add_argument("--month", help="YYYY-MM to compare on. Default: each agent's latest month in the file.")
    ap.add_argument("--out-dir", default="output/burden_trial/vs_disbursed")
    args = ap.parse_args(argv)

    limits = pd.read_csv(_resolve(args.agents), dtype={"msisdn": str})
    disbursed, months = load_disbursed(_resolve(args.disbursed), args.month)
    m = build_agents(limits, disbursed)
    if m.empty:
        sys.exit("ERROR: no agent in the disbursement file matches an agent in the trial output (check msisdn format).")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cat = by_category(m)
    bnd = bands(m)
    m.to_csv(out / "limits_vs_disbursed_agents.csv", index=False)
    cat.to_csv(out / "limits_vs_disbursed_by_category.csv", index=False)
    bnd.to_csv(out / "limits_vs_disbursed_bands.csv")

    month_note = args.month or f"each agent's latest month (months in file: {', '.join(months)})"
    overall = cat[cat["agent_category"] == "ALL AGENTS"].drop(columns="agent_category")
    wide_cut = cat.pivot(index="agent_category", columns="limit", values="cut_share")
    wide_cut = wide_cut.reindex(list(cat["agent_category"].unique()))[[v for v in LIMIT_COLUMNS.values() if v in wide_cut.columns]]
    wide_short = cat.pivot(index="agent_category", columns="limit", values="shortfall_total_m")
    wide_short = wide_short.reindex(wide_cut.index)[wide_cut.columns]

    md = (
        "# Trial limits vs actual disbursements\n\n"
        f"Agents in trial: **{len(limits):,}**. Matched to a disbursement: **{len(m):,}**. Month: {month_note}.\n\n"
        "*Cut* = the agent actually took more than this limit would allow. *Shortfall* = UGX they took above it.\n\n"
        f"## Overall\n\n{_md(overall)}\n"
        f"## Share of agents cut, by category\n\n{_md(wide_cut.reset_index())}\n"
        f"## Shortfall (UGX millions), by category\n\n{_md(wide_short.reset_index())}\n"
        f"## Disbursed / limit bands (share of matched agents)\n\n{_md(bnd.reset_index())}\n"
    )
    (out / "limits_vs_disbursed.md").write_text(md, encoding="utf-8")

    pd.set_option("display.width", 200)
    print(f"Agents in trial: {len(limits):,} | matched to a disbursement: {len(m):,} | month: {month_note}\n")
    print("== Overall ==")
    print(overall.round(3).to_string(index=False))
    print("\n== Share of agents whose actual disbursement is ABOVE the limit (cut) ==")
    print((wide_cut * 100).round(1).to_string())
    print("\n== Shortfall: UGX millions disbursed above the limit ==")
    print(wide_short.round(1).to_string())
    print("\n== disbursed / limit bands (share of matched agents) ==")
    print((bnd * 100).round(1).to_string())
    print(f"\nWritten to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
