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
        --out-missing whitelist_agents_not_scored.csv
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


if __name__ == "__main__":
    main()
