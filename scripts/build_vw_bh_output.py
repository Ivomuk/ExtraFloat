"""
Assembles the seven SQL statements used by
data/borrower_history_validation_queries.sql's GATE 1 final-output checks,
from the checked-in data/borrower_history.txt -- so validation always runs
against the exact production query, not a manually pasted (and potentially
stale or hand-edited) copy.

borrower_history.txt is split by its
##BORROWER_HISTORY_CHECKPOINT_0##/_1##/_2## markers into four statements
instead of one, because the unsplit query exceeds the warehouse's
query-plan stage-count ceiling (100 max; ~280 observed unsplit, ~142
observed after one split, 140 observed on checkpoint 1 alone after two
splits). Part A0 (everything before marker 0: dedup, attribution, cure-event
classification), Part A1 (between marker 0 and marker 1: cure timing,
loan-level flags, reading Part A0's checkpoint table instead of re-deriving
`classified`), and Part A2 (between marker 1 and marker 2: the
borrower-level window-function cascade) are each materialized as physical
checkpoint tables in turn; Part A3 (everything after marker 2) is the final
CREATE OR REPLACE VIEW, built from the third checkpoint table instead of
re-deriving anything inline. See the comments at each marker in
borrower_history.txt for the full rationale.

Stamps the git commit SHA of data/borrower_history.txt into the output as a
comment, and warns if the working tree has uncommitted changes to it -- both
so the SHA can be recorded in GATE 6 as evidence of exactly what was
validated.

Usage:
    python scripts/build_vw_bh_output.py <validation_schema> > vw_bh_output.sql
    # then run vw_bh_output.sql in Athena/Trino -- it contains seven
    # statements in order: DROP TABLE/CREATE TABLE ... AS SELECT for
    # checkpoint 0, DROP TABLE/CREATE TABLE ... AS SELECT for checkpoint 1
    # (built from checkpoint 0), DROP TABLE/CREATE TABLE ... AS SELECT for
    # checkpoint 2 (built from checkpoint 1), then CREATE OR REPLACE VIEW
    # for vw_bh_output (built from checkpoint 2).
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "borrower_history.txt"

CHECKPOINT_MARKER_0 = "-- ##BORROWER_HISTORY_CHECKPOINT_0##"
CHECKPOINT_MARKER_1 = "-- ##BORROWER_HISTORY_CHECKPOINT_1##"
CHECKPOINT_MARKER_2 = "-- ##BORROWER_HISTORY_CHECKPOINT_2##"
CHECKPOINT_TABLE_NAME_0 = "tbl_bh_classified"
CHECKPOINT_TABLE_NAME_1 = "tbl_bh_loan_final"
CHECKPOINT_TABLE_NAME_2 = "tbl_bh_loan_level"
CHECKPOINT_PLACEHOLDER_0 = "{{CHECKPOINT_TABLE_0}}"
CHECKPOINT_PLACEHOLDER_1 = "{{CHECKPOINT_TABLE_1}}"
CHECKPOINT_PLACEHOLDER_2 = "{{CHECKPOINT_TABLE_2}}"

# Athena/Trino unquoted identifier: letters/digits/underscore, not starting
# with a digit. schema is interpolated directly into `CREATE TABLE
# {schema}.tbl_bh_loan_final` / `{schema}.tbl_bh_loan_level` /
# `CREATE OR REPLACE VIEW {schema}.vw_bh_output` below -- reject anything
# else rather than emit malformed or unintended SQL from a typo'd or
# pasted-wrong argument.
_VALID_SCHEMA = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Matches the three literals in borrower_history.txt's `snapshots` CTE, e.g.
#   cast(20260531 AS bigint) AS snapshot_dt,
#   cast('2026-06-01 00:00:00.000' AS timestamp) AS as_of_load_ts,
#   cast('2026-06-01 00:00:00.000' AS timestamp) AS snapshot_ts
_SNAPSHOT_PARAM = re.compile(
    r"cast\(\s*('[^']*'|\d+)\s+AS\s+\w+\s*\)\s+AS\s+(snapshot_dt|as_of_load_ts|snapshot_ts)\b",
    re.IGNORECASE,
)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _split_once(body: str, marker: str, label: str) -> tuple[str, str]:
    # Fail closed on each checkpoint split: a missing or duplicated marker
    # means the file's shape changed in a way this script no longer
    # understands, and silently guessing which split to use (or skipping the
    # split) could emit SQL that either won't run or silently validates the
    # wrong thing.
    count = body.count(marker)
    if count != 1:
        sys.exit(
            f"ERROR: found {count} occurrences of the {label} marker "
            f"({marker!r}) in data/borrower_history.txt, expected exactly 1. "
            "Either the file's checkpoint split changed shape (update the "
            "marker constant in this script) or the marker is "
            "missing/duplicated -- refusing to guess how to split the file."
        )
    before, after = body.split(marker, 1)
    return before, after


def _require_placeholder(part: str, placeholder: str, label: str) -> None:
    if placeholder not in part:
        sys.exit(
            f"ERROR: {placeholder!r} not found anywhere in {label} of "
            "data/borrower_history.txt -- it should reference the checkpoint "
            "table by this placeholder. Refusing to generate SQL that would "
            "still reference the old inline CTE."
        )


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {Path(__file__).name} <validation_schema>")
    schema = sys.argv[1]
    if not _VALID_SCHEMA.match(schema):
        sys.exit(
            f"ERROR: '{schema}' is not a valid unquoted SQL identifier "
            "(letters/digits/underscore, not starting with a digit). "
            "Refusing to interpolate it into a CREATE TABLE/VIEW statement."
        )

    if not SRC.exists():
        sys.exit(f"ERROR: {SRC} not found")

    sha = _git("log", "-1", "--format=%H", "--", "data/borrower_history.txt") or "unknown"
    dirty = bool(_git("status", "--porcelain", "--", "data/borrower_history.txt"))
    dirty_note = " -- WORKING TREE HAS UNCOMMITTED CHANGES TO THIS FILE, SHA ABOVE IS STALE" if dirty else ""

    body = SRC.read_text()

    part_a0, rest0 = _split_once(body, CHECKPOINT_MARKER_0, "checkpoint 0")
    part_a1, rest = _split_once(rest0, CHECKPOINT_MARKER_1, "checkpoint 1")
    part_a2, part_a3 = _split_once(rest, CHECKPOINT_MARKER_2, "checkpoint 2")

    checkpoint_table_0 = f"{schema}.{CHECKPOINT_TABLE_NAME_0}"
    checkpoint_table_1 = f"{schema}.{CHECKPOINT_TABLE_NAME_1}"
    checkpoint_table_2 = f"{schema}.{CHECKPOINT_TABLE_NAME_2}"

    _require_placeholder(part_a1, CHECKPOINT_PLACEHOLDER_0, "Part A1 (between checkpoint 0 and checkpoint 1)")
    _require_placeholder(part_a2, CHECKPOINT_PLACEHOLDER_1, "Part A2 (between checkpoint 1 and checkpoint 2)")
    _require_placeholder(part_a3, CHECKPOINT_PLACEHOLDER_2, "Part A3 (after checkpoint 2)")
    part_a1 = part_a1.replace(CHECKPOINT_PLACEHOLDER_0, checkpoint_table_0)
    part_a2 = part_a2.replace(CHECKPOINT_PLACEHOLDER_1, checkpoint_table_1)
    part_a3 = part_a3.replace(CHECKPOINT_PLACEHOLDER_2, checkpoint_table_2)

    # Extract the literal snapshot_dt/as_of_load_ts/snapshot_ts baked into
    # THIS file, so whoever runs the validation queries copies the exact
    # values instead of retyping them from memory or a stale note -- the
    # generated vw_bh_output and the hand-run validation queries silently
    # drifting onto different cutoffs would invalidate every reconciliation
    # result without necessarily looking wrong.
    #
    # Fail closed: if borrower_history.txt's snapshots CTE has changed shape
    # such that a parameter is missing, or the file's copies of it disagree
    # (e.g. one got hand-edited and the others didn't), silently picking one
    # match via a dict comprehension would let a wrong or stale value flow
    # into a runnable CREATE TABLE/VIEW statement with no warning that
    # anything went wrong. Refuse to emit SQL at all in that case instead.
    #
    # NOTE: the snapshots CTE is legitimately redefined twice in this file --
    # once in checkpoint 0's own WITH clause, and again at the start of
    # checkpoint 1's remainder, since checkpoint 1 is a separate SQL
    # statement that can't see checkpoint 0's CTEs. So each param is expected
    # to match at least once and every match must agree, not "exactly one
    # match total".
    matches = _SNAPSHOT_PARAM.findall(body)
    counts: dict[str, list[str]] = {"snapshot_dt": [], "as_of_load_ts": [], "snapshot_ts": []}
    for value, name in matches:
        counts[name].append(value)

    errors = [
        f"  {name}: found {len(values)} matches ({values!r}), expected at least 1 and all equal"
        for name, values in counts.items()
        if len(values) == 0 or len(set(values)) != 1
    ]
    if errors:
        sys.exit(
            "ERROR: could not unambiguously extract snapshot_dt/as_of_load_ts/"
            "snapshot_ts from data/borrower_history.txt's snapshots CTE -- "
            "refusing to generate SQL against a mismatched or guessed cutoff.\n"
            + "\n".join(errors)
            + "\nEither the snapshots CTE has changed shape (update "
            "_SNAPSHOT_PARAM in this script) or its redefinitions have drifted "
            "out of sync with each other."
        )
    params = {name: values[0] for name, values in counts.items()}

    print("-- Auto-generated by scripts/build_vw_bh_output.py -- DO NOT EDIT BY HAND.")
    print(f"-- Built from data/borrower_history.txt @ commit {sha}{dirty_note}")
    print("-- Record this commit SHA in GATE 6 of borrower_history_validation_queries.sql.")
    print("--")
    print("-- Use these EXACT values for :snapshot_dt / :as_of_load_ts / :snapshot_ts")
    print("-- when running borrower_history_validation_queries.sql -- if they don't match")
    print("-- what you substitute there, GATE 0's views and this production view are")
    print("-- validating two different cutoffs without any error being raised.")
    for name in ("snapshot_dt", "as_of_load_ts", "snapshot_ts"):
        print(f"--   {name}: {params[name]}")
    print("--")
    print("-- Runs as seven statements: checkpoint 0 is dropped and rebuilt first,")
    print("-- then checkpoint 1 (built from checkpoint 0), then checkpoint 2 (built")
    print("-- from checkpoint 1), then vw_bh_output (built from checkpoint 2).")
    print(f"DROP TABLE IF EXISTS {checkpoint_table_0};")
    print(f"CREATE TABLE {checkpoint_table_0} AS")
    print(part_a0.rstrip())
    print(";")
    print()
    print(f"DROP TABLE IF EXISTS {checkpoint_table_1};")
    print(f"CREATE TABLE {checkpoint_table_1} AS")
    print(part_a1.rstrip())
    print(";")
    print()
    print(f"DROP TABLE IF EXISTS {checkpoint_table_2};")
    print(f"CREATE TABLE {checkpoint_table_2} AS")
    print(part_a2.rstrip())
    print(";")
    print()
    print(f"CREATE OR REPLACE VIEW {schema}.vw_bh_output AS")
    print(part_a3.rstrip())
    print(";")


if __name__ == "__main__":
    main()
