"""
extrafloat/cap_proposal/summarize_vs_disbursed.py
=================================================
One side-by-side summary of several compare_limits_vs_disbursed runs (read-only).

Each run folder (written by compare_limits_vs_disbursed --out-dir) has the frozen engine, the current engine
and the design's limits under the three category settings. This puts the frozen engine, the current engine
(taken from the first run) and the chosen design limit of EVERY run in one table each for:
  overall fit, share of agents cut by agent category, shortfall by category, and disbursed / limit bands.

    python -m extrafloat.cap_proposal.summarize_vs_disbursed ^
        --run "PD 2/5/10=output\\burden_trial\\vs_disbursed" ^
        --run "PD 3/12/30=output\\burden_trial\\vs_disbursed_pd_3_12_30" ^
        --run "PD 5/15/35=output\\burden_trial\\vs_disbursed_pd_5_15_35" ^
        --out-dir output\\burden_trial\\vs_disbursed_summary

Outputs in --out-dir: summary_overall.csv, summary_cut_share_by_category.csv,
summary_shortfall_by_category.csv, summary_bands.csv, summary.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BASELINES = ["frozen engine", "current engine"]
OVERALL_COLS = ["agents_matched", "median_limit", "median_limit_over_disbursed", "covered_share", "cut_agents",
                "cut_share", "shortfall_total_m", "zero_limit_but_disbursed"]


def parse_run(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--run must be LABEL=FOLDER, got {text!r}")
    label, folder = text.split("=", 1)
    return label.strip(), Path(folder.strip())


def load_run(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat_p, band_p = folder / "limits_vs_disbursed_by_category.csv", folder / "limits_vs_disbursed_bands.csv"
    for p in (cat_p, band_p):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found -- run compare_limits_vs_disbursed with --out-dir {folder} first")
    return pd.read_csv(cat_p), pd.read_csv(band_p, index_col=0)


def build(runs: list[tuple[str, Path]], design_limit: str):
    columns: dict[str, tuple[pd.DataFrame, str]] = {}   # column label -> (by_category frame, limit name in it)
    bands: dict[str, pd.Series] = {}
    for i, (label, folder) in enumerate(runs):
        cat, bnd = load_run(folder)
        if i == 0:
            for b in BASELINES:
                if b in set(cat["limit"]):
                    columns[b] = (cat, b)
                    bands[b] = bnd[b]
        if design_limit not in set(cat["limit"]):
            raise ValueError(f"{folder}: no '{design_limit}' limit (has {sorted(set(cat['limit']))})")
        columns[label] = (cat, design_limit)
        bands[label] = bnd[design_limit]

    def pick(metric: str, category: str | None = None) -> pd.DataFrame:
        out = {}
        for col, (cat, lim) in columns.items():
            g = cat[cat["limit"] == lim]
            if category is not None:
                g = g[g["agent_category"] == category]
                out[col] = g.iloc[0][metric] if len(g) else float("nan")
            else:
                out[col] = g.set_index("agent_category")[metric]
        return pd.Series(out) if category is not None else pd.DataFrame(out)

    overall = pd.DataFrame({m: pick(m, "ALL AGENTS") for m in OVERALL_COLS}).T
    cut = pick("cut_share") * 100
    shortfall = pick("shortfall_total_m")
    band_tbl = pd.DataFrame(bands) * 100
    return overall, cut, shortfall, band_tbl


def _md(df: pd.DataFrame, fmt: str) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [("" if pd.isna(v) else fmt.format(v)) for v in row]) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Side-by-side summary of compare_limits_vs_disbursed runs.")
    ap.add_argument("--run", action="append", type=parse_run, required=True, metavar="LABEL=FOLDER",
                    help="A compare_limits_vs_disbursed --out-dir folder and the label to show it under. Repeat.")
    ap.add_argument("--design-limit", default="category enforced",
                    choices=["category enforced", "category not enforced", "enforce below diamond"],
                    help="Which of the design's limits to take from each run (default: category enforced).")
    ap.add_argument("--out-dir", default="output/burden_trial/vs_disbursed_summary")
    args = ap.parse_args(argv)

    overall, cut, shortfall, band_tbl = build(args.run, args.design_limit)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    overall.to_csv(out / "summary_overall.csv")
    cut.to_csv(out / "summary_cut_share_by_category.csv")
    shortfall.to_csv(out / "summary_shortfall_by_category.csv")
    band_tbl.to_csv(out / "summary_bands.csv")

    ov = overall.copy()
    ov.index.name = "metric"
    ov_fmt = ov.astype(object)
    for m in ov.index:
        for c in ov.columns:
            v = ov.loc[m, c]
            ov_fmt.loc[m, c] = ("" if pd.isna(v) else f"{v:.1%}" if m in ("covered_share", "cut_share")
                                else f"{v:.2f}" if m == "median_limit_over_disbursed" else f"{v:,.1f}" if m == "shortfall_total_m"
                                else f"{v:,.0f}")
    md = (
        f"# Limits vs actual disbursements: side by side\n\nDesign limit shown for each run: **{args.design_limit}**. "
        "Frozen and current engine taken from the first run. *Cut* = the agent borrowed more than the limit.\n\n"
        f"## Overall\n\n{_md(ov_fmt, '{}')}\n"
        f"## Share of agents cut (%), by category\n\n{_md(cut, '{:.1f}')}\n"
        f"## Shortfall (UGX millions), by category\n\n{_md(shortfall, '{:,.1f}')}\n"
        f"## Disbursed / limit bands (% of matched agents)\n\n{_md(band_tbl, '{:.1f}')}\n"
        "Runs: " + "; ".join(f"{lab} = {fold}" for lab, fold in args.run) + "\n"
    )
    (out / "summary.md").write_text(md, encoding="utf-8")

    pd.set_option("display.width", 220)
    print(f"Design limit shown for each run: {args.design_limit}\n\n== Overall ==")
    print(ov_fmt.to_string())
    print("\n== Share of agents cut (%), by category ==")
    print(cut.round(1).to_string())
    print("\n== Shortfall (UGX millions), by category ==")
    print(shortfall.round(1).to_string())
    print("\n== Disbursed / limit bands (% of matched agents) ==")
    print(band_tbl.round(1).to_string())
    print(f"\nWritten to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
