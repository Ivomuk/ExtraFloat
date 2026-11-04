"""
extrafloat/cap_proposal/compare_scenarios.py
============================================
Compare limit-design scenarios with a base scenario, agent by agent (read-only).

Reads the trial's agents.csv of each scenario (same agents, same category setting) and reports against the base:
  * totals: total limit, change, agents graduated, agents whose limit went up, exposure added
  * by agent category and by PD risk tier: agents up, exposure added, median increase
  * graduated agents: which rule finally set their limit -- i.e. how many actually received the higher multiple
    and how many were still held by the past-loan governor, engine caps or the category maximum
  * graduated agents by experience: completed loans and on-time streak

    python -m extrafloat.cap_proposal.compare_scenarios ^
        --base "A 4x=output\\burden_trial\\limit_design_proposal__guard_off\\agents.csv" ^
        --scenario "B 5x=output\\burden_trial\\limit_design_scenario_B_ladder_5x__guard_off\\agents.csv" ^
        --scenario "C 6x T1=output\\burden_trial\\limit_design_scenario_C_ladder_6x_tier1__guard_off\\agents.csv" ^
        --out-dir output\\burden_trial\\scenarios_off

This does NOT validate safety: it shows who would get more and what stops them. Default rates at the new
loan-to-commission levels need historical loan outcomes (not in these files).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORY_ORDER = ["Below Threshold", "new bronze", "bronze", "silver", "gold", "platinum", "titanium", "diamond"]
KEEP = ["msisdn", "limit_proposed", "design_agent_category", "risk_tier", "cal_pd", "design_stage", "design_graduation",
        "design_k_capacity", "design_binding_constraint", "closed_loan_count", "consecutive_on_time_loans",
        "design_monthly_commission", "max_prior_loan_size", "design_agent_tier_max",
        "design_ladder_cap", "design_limit_before_governor", "design_limit_before_tier_guard"]
ENGINE_CEILING = 5_000_000.0


def parse_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH, got {text!r}")
    label, path = text.split("=", 1)
    return label.strip(), Path(path.strip())


def load(path: Path) -> pd.DataFrame:
    a = pd.read_csv(path, dtype={"msisdn": str}, low_memory=False)
    return a[[c for c in KEEP if c in a.columns]].drop_duplicates("msisdn")


def compare(base: pd.DataFrame, scen: pd.DataFrame) -> pd.DataFrame:
    m = scen.merge(base[["msisdn", "limit_proposed"]].rename(columns={"limit_proposed": "limit_base"}), on="msisdn", how="inner")
    m["change"] = m["limit_proposed"] - m["limit_base"]
    m["up"] = m["change"] > 0.5
    m["graduated"] = m.get("design_graduation", pd.Series("", index=m.index)).fillna("").astype(str).ne("")
    m["above_category_max"] = m["limit_proposed"] > m.get("design_agent_tier_max", np.inf) + 1
    return m


def _group(m: pd.DataFrame, by: str, order: list[str] | None = None) -> pd.DataFrame:
    g = m.groupby(by, dropna=False)
    t = pd.DataFrame({
        "agents": g.size(),
        "graduated": g["graduated"].sum().astype(int),
        "limit_up": g["up"].sum().astype(int),
        "share_up": g["up"].mean(),
        "base_total_m": g["limit_base"].sum() / 1e6,
        "scenario_total_m": g["limit_proposed"].sum() / 1e6,
        "added_m": g["change"].sum() / 1e6,
        "median_increase_if_up": m[m["up"]].groupby(by, dropna=False)["change"].median(),
    })
    if order:
        t = t.reindex([o for o in order if o in t.index] + [i for i in t.index if i not in order])
    return t


def waterfall(a: pd.DataFrame) -> dict:
    """Where potential exposure is held back, in UGX millions, applying the design's caps in order:
    commission capacity (ladder multiple x monthly commission, at most the engine ceiling)
      - engine caps and risk tier (usage, exposure, risk cap, tier multiplier)  -> limit before governor
      - past-loan governor                                                       -> limit before category guard
      - category maximum (and any arrears block)                                 -> final limit"""
    need = {"design_ladder_cap", "design_limit_before_governor", "design_limit_before_tier_guard", "limit_proposed"}
    if not need <= set(a.columns):
        return {}
    cap = a["design_ladder_cap"].clip(lower=0, upper=ENGINE_CEILING)
    s1 = a["design_limit_before_governor"].clip(lower=0)
    s2 = a["design_limit_before_tier_guard"].clip(lower=0)
    fin = a["limit_proposed"].clip(lower=0)
    return {
        "agents": len(a),
        "commission_capacity_m": round(cap.sum() / 1e6, 1),
        "held_by_engine_caps_and_tier_m": round((cap - s1).clip(lower=0).sum() / 1e6, 1),
        "held_by_governor_m": round((s1 - s2).clip(lower=0).sum() / 1e6, 1),
        "held_by_category_max_m": round((s2 - fin).clip(lower=0).sum() / 1e6, 1),
        "final_limits_m": round(fin.sum() / 1e6, 1),
    }


def pd_spread(gr: pd.DataFrame) -> pd.DataFrame:
    """PD distribution of graduated agents by risk tier."""
    if gr.empty or "cal_pd" not in gr or "risk_tier" not in gr:
        return pd.DataFrame()
    q = gr.groupby("risk_tier")["cal_pd"].describe(percentiles=[0.1, 0.5, 0.9])
    return q[["count", "min", "10%", "50%", "90%", "max"]]


def summarize(label: str, m: pd.DataFrame) -> dict:
    gr = m[m["graduated"]]
    return {
        "scenario": label,
        "agents": len(m),
        "graduated": int(m["graduated"].sum()),
        "graduated_share_of_all_agents": round(m["graduated"].mean(), 4),
        "limit_up": int(m["up"].sum()),
        "limit_down": int((m["change"] < -0.5).sum()),
        "base_total_m": round(m["limit_base"].sum() / 1e6, 1),
        "scenario_total_m": round(m["limit_proposed"].sum() / 1e6, 1),
        "added_m": round(m["change"].sum() / 1e6, 1),
        "added_pct": round(100 * m["change"].sum() / m["limit_base"].sum(), 2) if m["limit_base"].sum() else None,
        "graduated_but_not_up": int((m["graduated"] & ~m["up"]).sum()),
        "above_category_max": int(m["above_category_max"].sum()),
        "graduated_median_pd": round(float(gr["cal_pd"].median()), 4) if "cal_pd" in gr and len(gr) else None,
    }


def _fmt(t: pd.DataFrame) -> str:
    out = t.copy().astype(object)
    for c in t.columns:
        for i in t.index:
            v = t.loc[i, c]
            out.loc[i, c] = ("" if pd.isna(v) else f"{v:.1%}" if c.startswith("share") else
                             f"{v:,.1f}" if c.endswith("_m") else f"{v:,.0f}")
    return out.to_string()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare limit-design scenarios with a base, agent by agent (read-only).")
    ap.add_argument("--base", required=True, type=parse_spec, metavar="LABEL=agents.csv")
    ap.add_argument("--scenario", required=True, action="append", type=parse_spec, metavar="LABEL=agents.csv")
    ap.add_argument("--focus", default="silver,gold,platinum,titanium",
                    help="Comma-separated agent categories to report separately (default Silver to Titanium).")
    ap.add_argument("--out-dir", default="output/burden_trial/scenarios")
    args = ap.parse_args(argv)

    base_label, base_path = args.base
    base = load(base_path)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    focus = [x.strip().lower() for x in args.focus.split(",") if x.strip()]
    rows, md = [], [f"# Scenarios vs {base_label}\n"]
    waterfalls = []
    wb = waterfall(base)
    if wb:
        waterfalls.append({"scenario": base_label, "subset": "all agents", **wb})
    pd.set_option("display.width", 220)
    for label, path in args.scenario:
        m = compare(base, load(path))
        s = summarize(label, m)
        fm = m[m["design_agent_category"].astype(str).str.lower().isin(focus)] if "design_agent_category" in m else m.iloc[0:0]
        s.update({"focus_agents": len(fm), "focus_graduated": int(fm["graduated"].sum()),
                  "focus_graduated_share": round(fm["graduated"].mean(), 4) if len(fm) else None,
                  "focus_limit_up": int(fm["up"].sum()), "focus_added_m": round(fm["change"].sum() / 1e6, 1)})
        rows.append(s)
        slug = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")
        by_cat = _group(m, "design_agent_category", CATEGORY_ORDER) if "design_agent_category" in m else pd.DataFrame()
        by_tier = _group(m, "risk_tier") if "risk_tier" in m else pd.DataFrame()
        gr = m[m["graduated"]]
        by_rule = (gr.groupby(["design_graduation", "design_binding_constraint"]).size().rename("graduated_agents").to_frame()
                   if len(gr) and "design_binding_constraint" in gr else pd.DataFrame())
        by_cat.to_csv(out / f"{slug}_by_category.csv")
        by_tier.to_csv(out / f"{slug}_by_tier.csv")
        by_rule.to_csv(out / f"{slug}_graduated_by_rule.csv")
        spread = pd_spread(gr)
        spread.to_csv(out / f"{slug}_graduated_pd_by_tier.csv")
        sm = load(path)
        for subset, frame in (("all agents", sm), ("graduated agents", sm[sm["msisdn"].isin(gr["msisdn"])])):
            w = waterfall(frame)
            if w:
                waterfalls.append({"scenario": label, "subset": subset, **w})
        m.to_csv(out / f"{slug}_agents.csv", index=False)
        block = (f"\n=== {label} vs {base_label} ===\n"
                 f"graduated {s['graduated']:,} | limit up for {s['limit_up']:,} | total {s['base_total_m']:,.1f}M -> "
                 f"{s['scenario_total_m']:,.1f}M (+{s['added_m']:,.1f}M, {s['added_pct']}%) | graduated but no increase "
                 f"{s['graduated_but_not_up']:,} | {', '.join(focus)}: up {s['focus_limit_up']:,} of {s['focus_agents']:,}, "
                 f"+{s['focus_added_m']:,.1f}M\n")
        print(block)
        if not by_cat.empty:
            print("-- by agent category --\n" + _fmt(by_cat))
        if not by_tier.empty:
            print("\n-- by PD risk tier --\n" + _fmt(by_tier))
        if not by_rule.empty:
            print("\n-- graduated agents: rule that finally set the limit --\n" + by_rule.to_string())
        if not spread.empty:
            print("\n-- graduated agents: PD by risk tier --\n" + spread.round(4).to_string())
        md.append(block.replace("===", "##") + "\n```\n" + (_fmt(by_cat) if not by_cat.empty else "") + "\n```\n")
    if waterfalls:
        wf = pd.DataFrame(waterfalls)
        wf.to_csv(out / "exposure_waterfall.csv", index=False)
        print("\n== Where potential exposure is held back (UGX millions) ==\n"
              "commission capacity - engine caps & tier - governor - category max = final limits\n" + wf.to_string(index=False))
        md.append("## Where potential exposure is held back (UGX millions)\n\n```\n" + wf.to_string(index=False) + "\n```\n")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "scenarios_summary.csv", index=False)
    (out / "scenarios.md").write_text("\n".join(md), encoding="utf-8")
    print("\n== Summary ==\n" + summary.to_string(index=False))
    print(f"\nWritten to {out.resolve()}  (the *_agents.csv files contain phone numbers -- keep internal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
