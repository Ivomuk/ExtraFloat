"""
Traces whitelist agents through every named stage of the retail-filtered
pipeline (run_retail_filtered.bat), reporting presence/absence at each
stage -- not just the final scored output.

Answers "where exactly did this whitelist agent drop out" instead of just
"are they in the final output": were they excluded by the retail-agent
filter itself (Step 2 of run_retail_filtered.bat), missing from
borrower_history_retail_filtered.csv (already covered separately by
check_whitelist_blacklist_eval.py's PART A, but included here for a
one-stop trace), missing from loan_summary_retail_filtered.csv or
loan_history_snapshot_..._retail_filtered.csv, or never in the raw source
data at all (i.e. absent even before any filtering)?

Auto-detects the agent-key column per stage file among
agent_msisdn/msisdn/phonenumber, since different files in this pipeline
use different column names for the same identifier.

Usage (order of --stage matters only for display; presence is checked
independently per stage, not assumed monotonic):
    python scripts\\check_whitelist_not_scored.py ^
        --whitelist-file data\\whitelist_aug_20260804.csv ^
        --stage "raw_agent_mart=data\\mfs_daily_agent_mart_20260731.csv" ^
        --stage "retail_filtered=retail_agents_filtered.csv" ^
        --stage "borrower_raw=data\\borrower_history.csv" ^
        --stage "borrower_filtered=borrower_history_retail_filtered.csv" ^
        --stage "loan_summary_filtered=data\\loan_summary_retail_filtered.csv" ^
        --stage "loan_history_filtered=data\\loan_history_snapshot_20260817_retail_filtered.csv" ^
        --stage "scored_output=output\\engine_test_output.csv" ^
        --final-stage scored_output ^
        --loan-summary-file data\\loan_summary.csv ^
        --out-missing whitelist_agents_not_scored.csv ^
        --out-real-gap whitelist_real_gap_agents.csv ^
        --out-borrower-filter-drop whitelist_borrower_filter_drop_overlap.csv

Two extra, more targeted analyses (both optional, both answer questions
raised while interpreting a real run of this script against
check_whitelist_blacklist_eval.py's PART A):

  --loan-summary-file : with --borrower-raw-stage (default "borrower_raw"),
      computes the REAL GAP list directly -- whitelist agents absent from
      the raw borrower_history stage who nonetheless have a real
      disbursement on record in --loan-summary-file. PART A only prints a
      category-count *summary* of this group; this exports the actual
      agent_msisdn list to --out-real-gap.

  --borrower-filter-drop-check : with --borrower-raw-stage,
      --borrower-filtered-stage (default "borrower_filtered"), and
      --retail-filter-stage (default "retail_filtered"), computes which
      whitelist agents are present at the raw borrower stage but absent
      at the filtered borrower stage, then cross-checks how many of THOSE
      are also excluded at the retail-filter stage -- answers "is this
      drop just the same retail-filter exclusions showing up again, or a
      separate gap" directly instead of by inference from stage counts.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

_KEY_CANDIDATES = ("agent_msisdn", "msisdn", "phonenumber")


def _parse_stage(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--stage must be NAME=PATH, got {raw!r}")
    name, path = raw.split("=", 1)
    return name.strip(), path.strip()


def _load_keys(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    key_col = next((c for c in _KEY_CANDIDATES if c in df.columns), None)
    if key_col is None:
        print(f"  WARNING: none of {_KEY_CANDIDATES} found in {path} "
              f"(columns: {list(df.columns)[:10]}...) -- treating as unavailable.")
        return None
    return set(df[key_col].astype(str).str.strip())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--whitelist-file", required=True, metavar="PATH")
    p.add_argument(
        "--stage", type=_parse_stage, action="append", required=True, metavar="NAME=PATH",
        help="A pipeline stage to check presence against. May be given multiple times, in "
             "pipeline order (raw source -> filtered -> scored output).",
    )
    p.add_argument(
        "--final-stage", default=None, metavar="NAME",
        help="Which --stage name is the final scored output (drives the summary line and "
             "--out-missing). Defaults to the last --stage given.",
    )
    p.add_argument("--out-missing", default=None, metavar="PATH",
                    help="Optional: write the full per-agent presence trace to this CSV.")
    p.add_argument("--loan-summary-file", default=None, metavar="PATH",
                    help="Enables the REAL GAP export: whitelist agents absent from "
                         "--borrower-raw-stage but with a real disbursement here.")
    p.add_argument("--borrower-raw-stage", default="borrower_raw", metavar="NAME")
    p.add_argument("--borrower-filtered-stage", default="borrower_filtered", metavar="NAME")
    p.add_argument("--retail-filter-stage", default="retail_filtered", metavar="NAME")
    p.add_argument("--out-real-gap", default=None, metavar="PATH",
                    help="Where to write the REAL GAP agent list (requires --loan-summary-file).")
    p.add_argument("--out-borrower-filter-drop", default=None, metavar="PATH",
                    help="Where to write the borrower-raw-present/borrower-filtered-absent "
                         "agents, with a column flagging retail-filter overlap.")
    args = p.parse_args(argv)

    wl_path = Path(args.whitelist_file)
    if not wl_path.exists():
        sys.exit(f"ERROR: file not found: {wl_path}")
    wl = pd.read_csv(wl_path)
    if "agent_msisdn" not in wl.columns:
        sys.exit(f"ERROR: 'agent_msisdn' column not found in {wl_path} (columns: {list(wl.columns)})")

    wl = wl.copy()
    wl["_key"] = wl["agent_msisdn"].astype(str).str.strip()
    n_wl = len(wl)
    print(f"Loaded {n_wl:,} whitelist agents from {wl_path}\n")

    final_stage = args.final_stage or args.stage[-1][0]
    stage_names = [name for name, _ in args.stage]
    if final_stage not in stage_names:
        sys.exit(f"ERROR: --final-stage {final_stage!r} is not among --stage names: {stage_names}")

    presence = pd.DataFrame({"agent_msisdn": wl["agent_msisdn"], "_key": wl["_key"]})
    for name, path_str in args.stage:
        path = Path(path_str)
        keys = _load_keys(path)
        if keys is None:
            print(f"[{name}] MISSING/UNREADABLE: {path} -- marking all agents as unknown (NaN) for this stage.")
            presence[name] = pd.NA
        else:
            in_stage = presence["_key"].isin(keys)
            presence[name] = in_stage
            print(f"[{name}] {path}: {int(in_stage.sum()):,}/{n_wl:,} whitelist agents present "
                  f"({int(in_stage.sum()) / n_wl:.1%})")

    print()
    final_present = presence[final_stage]
    n_missing_final = int((final_present == False).sum())  # noqa: E712 -- must distinguish False from NA
    print(f"{n_missing_final:,} of {n_wl:,} whitelist agents ({n_missing_final / n_wl:.1%}) "
          f"NOT present at the final stage ({final_stage!r}).")

    missing = presence.loc[presence[final_stage] == False].drop(columns="_key")  # noqa: E712
    if len(missing) > 0:
        sample_n = min(20, len(missing))
        print(f"\nPer-agent presence trace -- showing {sample_n:,} of {len(missing):,} missing agents "
              f"on screen (True/False per stage; NaN = stage file unavailable). "
              f"{'See --out-missing for the full list.' if args.out_missing else 'Pass --out-missing to save the full list to a CSV instead of scrolling.'}")
        with pd.option_context("display.width", 200):
            print(missing.head(sample_n).to_string(index=False))

        if args.out_missing:
            out_path = Path(args.out_missing)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            missing.to_csv(out_path, index=False)
            print(f"\nWrote full presence trace for {len(missing):,} agents to {out_path}")

        # Earliest stage (in --stage order) where each missing agent is absent --
        # answers "where did they first drop out" directly, rather than making
        # the reader scan the whole True/False row by eye.
        print("\nFirst stage each missing agent drops out at (in --stage order given):")
        first_drop = []
        for _, row in missing.iterrows():
            dropped_at = next((name for name in stage_names if row.get(name) == False), "unknown")  # noqa: E712
            first_drop.append(dropped_at)
        print(pd.Series(first_drop).value_counts().to_string())

    # =========================================================================
    # Analysis 1: REAL GAP export -- whitelist agents absent from the raw
    # borrower stage who nonetheless have a real disbursement on record.
    # =========================================================================
    if args.loan_summary_file:
        print("\n" + "=" * 70)
        print("REAL GAP: absent from borrower-raw stage but has a real disbursement")
        print("=" * 70)
        if args.borrower_raw_stage not in stage_names:
            print(f"SKIPPED: --borrower-raw-stage {args.borrower_raw_stage!r} is not among "
                  f"the --stage names given ({stage_names}).")
        else:
            ls_path = Path(args.loan_summary_file)
            ls_keys = _load_keys(ls_path)
            if ls_keys is None:
                print(f"SKIPPED: could not load keys from --loan-summary-file {ls_path}.")
            else:
                absent_from_raw = presence[args.borrower_raw_stage] == False  # noqa: E712
                has_disbursement = presence["_key"].isin(ls_keys)
                real_gap = presence.loc[absent_from_raw & has_disbursement, ["agent_msisdn"]]
                n_absent = int(absent_from_raw.sum())
                print(f"{n_absent:,} whitelist agents absent from {args.borrower_raw_stage!r}; "
                      f"{len(real_gap):,} of those ({len(real_gap) / max(1, n_absent):.1%}) "
                      f"have a real disbursement in {ls_path} -- the REAL GAP.")
                if len(real_gap) > 0:
                    print(real_gap.head(20).to_string(index=False))
                    if args.out_real_gap:
                        out_path = Path(args.out_real_gap)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        real_gap.to_csv(out_path, index=False)
                        print(f"\nWrote {len(real_gap):,} REAL GAP agents to {out_path}")
                    elif len(real_gap) > 20:
                        print(f"(showing 20 of {len(real_gap):,} -- pass --out-real-gap to save the full list)")

    # =========================================================================
    # Analysis 2: borrower-filter-drop overlap -- agents present in the raw
    # borrower stage but absent from the filtered one, cross-checked against
    # the retail-filter exclusion stage.
    # =========================================================================
    if args.borrower_raw_stage in stage_names and args.borrower_filtered_stage in stage_names:
        print("\n" + "=" * 70)
        print("Borrower-filter drop vs. retail-filter exclusion overlap")
        print("=" * 70)
        present_raw = presence[args.borrower_raw_stage] == True  # noqa: E712
        absent_filtered = presence[args.borrower_filtered_stage] == False  # noqa: E712
        dropped = presence.loc[present_raw & absent_filtered].copy()
        print(f"{len(dropped):,} whitelist agents present at {args.borrower_raw_stage!r} but "
              f"absent at {args.borrower_filtered_stage!r}.")

        if len(dropped) > 0:
            if args.retail_filter_stage in stage_names:
                dropped["also_excluded_by_retail_filter"] = (
                    dropped[args.retail_filter_stage] == False  # noqa: E712
                )
                n_overlap = int(dropped["also_excluded_by_retail_filter"].sum())
                print(f"  Of those {len(dropped):,}: {n_overlap:,} "
                      f"({n_overlap / len(dropped):.1%}) are ALSO excluded at "
                      f"{args.retail_filter_stage!r} -- i.e. already-known retail-filter "
                      f"exclusions, not a separate/new gap.")
                n_unexplained = len(dropped) - n_overlap
                if n_unexplained > 0:
                    print(f"  {n_unexplained:,} are present at {args.retail_filter_stage!r} "
                          f"too -- a genuine, UNEXPLAINED drop during borrower-history "
                          f"retail-filtering specifically, worth a direct look.")
            else:
                print(f"  (--retail-filter-stage {args.retail_filter_stage!r} not among "
                      f"--stage names -- cannot check overlap.)")

            out_cols = [c for c in ("agent_msisdn", "also_excluded_by_retail_filter") if c in dropped.columns]
            print(dropped[out_cols].head(20).to_string(index=False))
            if args.out_borrower_filter_drop:
                out_path = Path(args.out_borrower_filter_drop)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                dropped[out_cols].to_csv(out_path, index=False)
                print(f"\nWrote {len(dropped):,} agents to {out_path}")
            elif len(dropped) > 20:
                print(f"(showing 20 of {len(dropped):,} -- pass --out-borrower-filter-drop to save the full list)")


if __name__ == "__main__":
    main()
