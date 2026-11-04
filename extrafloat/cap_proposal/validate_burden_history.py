"""
extrafloat/cap_proposal/validate_burden_history.py
==================================================
Historical default validation of the loan-to-commission multiples (read-only).

Question: how did loans perform at different realised loan / monthly-commission multiples, by repayment experience
at disbursement? This is the evidence for (or against) graduating proven agents from 4x to 5x and 6x.

Inputs
------
--loans   loan-level label file from data/loan_state_query_updated_materialized.txt (e.g. state_data_202609101323):
          msisdn, loan_date, disbursement_amount_ugx, bad_state_3dpd_30d, label_eligible_30d, split,
          prior_closed_loan_count, consecutive_on_time_loans, active_loan_days_aging_at_scoring, anomaly_open_at_scoring
--mart    agent mart with several snapshot dates (e.g. data/mfs_daily_agent_mart_202607.csv): agent_msisdn (or msisdn),
          tbl_dt (or snapshot_dt), commission (6-month commission, as the design uses it)

Method
------
* Only matured loans: label_eligible_30d = 1. Outcome = bad_state_3dpd_30d (more than 3 days past due within 30 days).
* Commission at disbursement = the agent's commission on the latest mart snapshot STRICTLY BEFORE the loan date
  (no look-ahead), divided by 6 = monthly commission, as in the design. Loans with no earlier snapshot, or with
  commission 0, are reported separately, never dropped silently.
* Multiple = disbursement_amount_ugx / monthly commission, in bands.
* Experience stage at disbursement, mirroring the design's ladder and graduation:
      S0  no completed loan                     S1  1-2 completed
      S2  3+ completed (not graduated)          G5  5+ completed and 5+ on-time streak (not G6)
      G6  8+ completed and 8+ on-time streak
  Agents in arrears at disbursement (days aging > 0 or an open rollover) are a separate group, as graduation
  requires no current arrears.
* Each cell: loans, bad loans, bad rate and its 95% Wilson interval, so small cells show as uncertain.

Outputs (in --out-dir): burden_by_band.csv, burden_by_band_and_stage.csv, burden_matching.csv, burden_validation.md

Limits: realised multiples reflect past limits (selection: larger loans went to agents the old policy trusted);
the outcome is lateness (3+ days past due), not loss; no PD tier at disbursement (current cal_pd would leak).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BAND_EDGES = [0, 1, 2, 3, 4, 5, 6, np.inf]
BAND_LABELS = ["0-1x", "1-2x", "2-3x", "3-4x", "4-5x", "5-6x", ">6x"]
STAGE_ORDER = ["S0 no completed loan", "S1 1-2 completed", "S2 3+ completed", "G5 5+ completed, 5+ streak",
               "G6 8+ completed, 8+ streak", "in arrears at disbursement"]
LOAN_COLS = ["msisdn", "loan_date", "disbursement_amount_ugx", "bad_state_3dpd_30d", "label_eligible_30d", "split",
             "prior_closed_loan_count", "consecutive_on_time_loans", "active_loan_days_aging_at_scoring",
             "anomaly_open_at_scoring"]
LOAN_REQUIRED = ["msisdn", "loan_date", "disbursement_amount_ugx", "bad_state_3dpd_30d", "label_eligible_30d"]


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.replace(r"[^0-9]", "", regex=True)


def _sep(path: Path) -> str:
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        head = fh.readline()
    return max([",", "\t", "|", ";"], key=head.count)


def _read(path: Path, wanted, required) -> pd.DataFrame:
    sep = _sep(path)
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        header = [c.strip().strip('"').lower() for c in fh.readline().rstrip("\r\n").split(sep)]
    missing = [c for c in required if c not in header]
    if missing:
        sys.exit(f"ERROR: {path} is missing {missing}. Columns found: {header[:40]}{' ...' if len(header) > 40 else ''}")
    use = [c for c in wanted if c in header]
    df = pd.read_csv(path, sep=sep, usecols=lambda c: c.strip().strip('"').lower() in use, low_memory=False,
                     encoding="utf-8-sig")
    df.columns = [c.strip().strip('"').lower() for c in df.columns]
    return df


def _date(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    out = pd.to_datetime(t, format="%Y%m%d", errors="coerce")
    return out.fillna(pd.to_datetime(t, errors="coerce"))


def load_mart(path: Path) -> pd.DataFrame:
    sep = _sep(path)
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        header = [c.strip().strip('"').lower() for c in fh.readline().rstrip("\r\n").split(sep)]
    key = next((c for c in ("agent_msisdn", "msisdn") if c in header), None)
    dt = next((c for c in ("tbl_dt", "snapshot_dt", "snapshot_date", "dt") if c in header), None)
    if key is None or dt is None or "commission" not in header:
        sys.exit(f"ERROR: {path} needs agent_msisdn/msisdn, tbl_dt/snapshot_dt and commission. "
                 f"Found key={key}, date={dt}, commission={'commission' in header}.")
    m = _read(path, [key, dt, "commission"], [key, dt, "commission"]).rename(columns={key: "msisdn", dt: "snapshot_dt"})
    m["msisdn"] = _norm(m["msisdn"])
    m["snapshot_dt"] = _date(m["snapshot_dt"])
    m["commission"] = pd.to_numeric(m["commission"], errors="coerce")
    return m.dropna(subset=["snapshot_dt"]).drop_duplicates(["msisdn", "snapshot_dt"], keep="last")


def load_loans(path: Path, split: str | None) -> pd.DataFrame:
    df = _read(path, LOAN_COLS, LOAN_REQUIRED)
    df["msisdn"] = _norm(df["msisdn"])
    df["loan_date"] = _date(df["loan_date"])
    for c in ("disbursement_amount_ugx", "bad_state_3dpd_30d", "label_eligible_30d", "prior_closed_loan_count",
              "consecutive_on_time_loans", "active_loan_days_aging_at_scoring", "anomaly_open_at_scoring"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if split and "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == split.lower()]
    return df


def stage(df: pd.DataFrame) -> pd.Series:
    c = df.get("prior_closed_loan_count", pd.Series(np.nan, index=df.index)).fillna(0)
    s = df.get("consecutive_on_time_loans", pd.Series(np.nan, index=df.index)).fillna(0)
    arrears = (df.get("active_loan_days_aging_at_scoring", pd.Series(0, index=df.index)).fillna(0) > 0) | \
              (df.get("anomaly_open_at_scoring", pd.Series(0, index=df.index)).fillna(0) > 0)
    st = np.select([arrears, (c >= 8) & (s >= 8), (c >= 5) & (s >= 5), c >= 3, c >= 1],
                   [STAGE_ORDER[5], STAGE_ORDER[4], STAGE_ORDER[3], STAGE_ORDER[2], STAGE_ORDER[1]], STAGE_ORDER[0])
    return pd.Series(st, index=df.index)


def attach_commission(loans: pd.DataFrame, mart: pd.DataFrame) -> pd.DataFrame:
    """Commission from the latest snapshot strictly before the loan date (merge_asof, no look-ahead)."""
    lo = loans.dropna(subset=["loan_date"]).sort_values("loan_date").copy()
    lo["lookup_date"] = lo["loan_date"] - pd.Timedelta(days=1)
    ma = mart.sort_values("snapshot_dt")
    out = pd.merge_asof(lo, ma[["msisdn", "snapshot_dt", "commission"]], left_on="lookup_date", right_on="snapshot_dt",
                        by="msisdn", direction="backward")
    out["monthly_commission"] = out["commission"] / 6.0
    out["multiple"] = np.where(out["monthly_commission"] > 0, out["disbursement_amount_ugx"] / out["monthly_commission"], np.nan)
    out["band"] = pd.cut(out["multiple"], BAND_EDGES, labels=BAND_LABELS, right=True, include_lowest=True)
    out["commission_age_days"] = (out["loan_date"] - out["snapshot_dt"]).dt.days
    return out


def wilson(bad: pd.Series, n: pd.Series, z: float = 1.96) -> tuple[pd.Series, pd.Series]:
    p = np.where(n > 0, bad / n, np.nan)
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return pd.Series(centre - half, index=n.index), pd.Series(centre + half, index=n.index)


def rate_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by, observed=False)
    t = pd.DataFrame({"loans": g.size(), "bad": g["bad_state_3dpd_30d"].sum()})
    t["bad_rate"] = np.where(t["loans"] > 0, t["bad"] / t["loans"], np.nan)
    lo, hi = wilson(t["bad"], t["loans"].replace(0, np.nan))
    t["ci95_low"], t["ci95_high"] = lo, hi
    t["share_of_loans"] = t["loans"] / t["loans"].sum()
    return t


def _pct(x):
    return "" if pd.isna(x) else f"{x:.2%}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bad rate by realised loan / monthly commission multiple and experience stage.")
    ap.add_argument("--loans", required=True, help="Loan-level label file (e.g. data\\state_data_202609101323)")
    ap.add_argument("--mart", required=True, help="Agent mart with several snapshot dates (e.g. data\\mfs_daily_agent_mart_202607.csv)")
    ap.add_argument("--split", help="Only this split (train / validation). Default: all matured loans.")
    ap.add_argument("--max-commission-age-days", type=int, default=45,
                    help="Ignore a commission snapshot older than this many days before the loan (default 45).")
    ap.add_argument("--out-dir", default="output/burden_validation")
    args = ap.parse_args(argv)

    mart = load_mart(Path(args.mart))
    dates = sorted(mart["snapshot_dt"].dt.strftime("%Y-%m-%d").unique())
    print(f"mart: {len(mart):,} agent-snapshots, {len(dates)} snapshot dates: {', '.join(dates[:12])}{' ...' if len(dates) > 12 else ''}")
    if len(dates) < 2:
        print("WARNING: only one mart snapshot -- commission cannot be measured at each loan's disbursement date. "
              "Loans before it get no commission (no look-ahead); results cover only loans after it.")

    loans = load_loans(Path(args.loans), args.split)
    n_all = len(loans)
    matured = loans[loans["label_eligible_30d"] == 1].copy()
    print(f"loans: {n_all:,} read, {len(matured):,} matured (label_eligible_30d = 1)"
          + (f", split = {args.split}" if args.split else ""))
    m = attach_commission(matured, mart)
    m["stage"] = stage(m)

    reasons = np.select(
        [m["snapshot_dt"].isna(), m["commission_age_days"] > args.max_commission_age_days, m["commission"].fillna(0) <= 0],
        ["no mart snapshot before the loan", f"commission snapshot older than {args.max_commission_age_days} days",
         "commission 0 or missing"], "used")
    m["matching"] = reasons
    matching = m["matching"].value_counts().rename_axis("matching").rename("loans").to_frame()
    matching["share"] = matching["loans"] / len(m)
    used = m[m["matching"] == "used"].copy()
    print("\n== Commission matching ==")
    print(matching.assign(share=matching["share"].map(_pct)).to_string())

    by_band = rate_table(used, ["band"])
    by_stage = rate_table(used, ["stage", "band"])
    by_stage = by_stage.reindex(pd.MultiIndex.from_product([STAGE_ORDER, BAND_LABELS], names=["stage", "band"]))
    by_stage = by_stage[by_stage["loans"].fillna(0) > 0]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_band.to_csv(out / "burden_by_band.csv")
    by_stage.to_csv(out / "burden_by_band_and_stage.csv")
    matching.to_csv(out / "burden_matching.csv")

    def fmt(t: pd.DataFrame) -> pd.DataFrame:
        f = t.copy().astype(object)
        for c in ("bad_rate", "ci95_low", "ci95_high", "share_of_loans"):
            f[c] = t[c].map(_pct)
        for c in ("loans", "bad"):
            f[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
        return f

    pd.set_option("display.width", 200)
    print(f"\n== Bad rate by loan / monthly commission ({len(used):,} matured loans with commission) ==")
    print(fmt(by_band).to_string())
    print("\n== Bad rate by experience stage at disbursement and multiple ==")
    print(fmt(by_stage).to_string())
    pivot = by_stage["bad_rate"].unstack("band").reindex(columns=BAND_LABELS)
    counts = by_stage["loans"].unstack("band").reindex(columns=BAND_LABELS)
    print("\n== Bad rate (loans) -- stage x multiple ==")
    cells = pivot.map(_pct) + " (" + counts.map(lambda v: "" if pd.isna(v) else f"{int(v):,}") + ")"
    print(cells.reindex([s for s in STAGE_ORDER if s in cells.index]).to_string())
    md = ("# Historical default validation: loan / monthly commission\n\n"
          f"Matured loans with commission at disbursement: **{len(used):,}** of {len(m):,} matured. Outcome: more than 3 days "
          "past due within 30 days. Commission: latest mart snapshot before the loan, 6-month / 6.\n\n"
          "## Commission matching\n\n```\n" + matching.to_string() + "\n```\n\n"
          "## By multiple\n\n```\n" + fmt(by_band).to_string() + "\n```\n\n"
          "## Stage x multiple: bad rate (loans)\n\n```\n" + cells.to_string() + "\n```\n\n"
          "Limits: realised multiples reflect past limits (selection); outcome is lateness, not loss; no PD tier at disbursement.\n")
    (out / "burden_validation.md").write_text(md, encoding="utf-8")
    print(f"\nWritten to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
