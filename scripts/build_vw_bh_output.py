"""
Assembles the two SQL statements used by
data/borrower_history_validation_queries.sql's GATE 1 final-output checks,
from the checked-in data/borrower_history.txt -- so validation always runs
against the exact production query, not a manually pasted (and potentially
stale or hand-edited) copy.

borrower_history.txt is split by its ##BORROWER_HISTORY_CHECKPOINT## marker
into two statements instead of one, because the unsplit query exceeds the
warehouse's query-plan stage-count ceiling (100 max; ~280 observed). Part A
(everything before the marker) is materialized as a physical checkpoint
table; Part B (everything after) is the final CREATE OR REPLACE VIEW, built
from that checkpoint table instead of re-deriving it inline. See the comment
at the marker in borrower_history.txt for the full rationale.

Stamps the git commit SHA of data/borrower_history.txt into the output as a
comment, and warns if the working tree has uncommitted changes to it -- both
so the SHA can be recorded in GATE 6 as evidence of exactly what was
validated.

Usage:
    python scripts/build_vw_bh_output.py <validation_schema> > vw_bh_output.sql
    # then run vw_bh_output.sql in Athena/Trino -- it contains three
    # statements in order: DROP TABLE IF EXISTS + CREATE TABLE ... AS SELECT
    # for the checkpoint, then CREATE OR REPLACE VIEW for vw_bh_output.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "borrower_history.txt"

CHECKPOINT_MARKER = "-- ##BORROWER_HISTORY_CHECKPOINT##"
CHECKPOINT_TABLE_NAME = "tbl_bh_loan_level"
CHECKPOINT_PLACEHOLDER = "{{CHECKPOINT_TABLE}}"

# Athena/Trino unquoted identifier: letters/digits/underscore, not starting
# with a digit. schema is interpolated directly into `CREATE TABLE
# {schema}.tbl_bh_loan_level` / `CREATE OR REPLACE VIEW {schema}.vw_bh_output`
# below -- reject anything else rather than emit malformed or unintended SQL
# from a typo'd or pasted-wrong argument.
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

    # Fail closed on the checkpoint split, same reasoning as the snapshot-
    # parameter check below: a missing or duplicated marker means the file's
    # shape changed in a way this script no longer understands, and silently
    # guessing which split to use (or skipping the split) could emit SQL
    # that either won't run or silently validates the wrong thing.
    marker_count = body.count(CHECKPOINT_MARKER)
    if marker_count != 1:
        sys.exit(
            f"ERROR: found {marker_count} occurrences of the checkpoint marker "
            f"({CHECKPOINT_MARKER!r}) in data/borrower_history.txt, expected "
            "exactly 1. Either the file's checkpoint split changed shape "
            "(update CHECKPOINT_MARKER in this script) or the marker is "
            "missing/duplicated -- refusing to guess how to split the file."
        )
    part_a, part_b = body.split(CHECKPOINT_MARKER, 1)

    placeholder_count = part_b.count(CHECKPOINT_PLACEHOLDER)
    if placeholder_count == 0:
        sys.exit(
            f"ERROR: {CHECKPOINT_PLACEHOLDER!r} not found anywhere after the "
            "checkpoint marker in data/borrower_history.txt -- Part B should "
            "reference the checkpoint table by this placeholder. Refusing to "
            "generate SQL that would still reference the old inline CTE."
        )
    checkpoint_table = f"{schema}.{CHECKPOINT_TABLE_NAME}"
    part_b = part_b.replace(CHECKPOINT_PLACEHOLDER, checkpoint_table)

    # Extract the literal snapshot_dt/as_of_load_ts/snapshot_ts baked into
    # THIS file, so whoever runs the validation queries copies the exact
    # values instead of retyping them from memory or a stale note -- the
    # generated vw_bh_output and the hand-run validation queries silently
    # drifting onto different cutoffs would invalidate every reconciliation
    # result without necessarily looking wrong.
    #
    # Fail closed: if borrower_history.txt's snapshots CTE has changed shape
    # such that a parameter is missing, or duplicated (e.g. a second CTE
    # elsewhere in the file coincidentally matches the pattern), silently
    # picking one match via a dict comprehension would let a wrong or stale
    # value flow into a runnable CREATE TABLE/VIEW statement with no warning
    # that anything went wrong. Refuse to emit SQL at all in that case instead.
    matches = _SNAPSHOT_PARAM.findall(body)
    counts: dict[str, list[str]] = {"snapshot_dt": [], "as_of_load_ts": [], "snapshot_ts": []}
    for value, name in matches:
        counts[name].append(value)

    errors = [
        f"  {name}: found {len(values)} matches ({values!r}), expected exactly 1"
        for name, values in counts.items()
        if len(values) != 1
    ]
    if errors:
        sys.exit(
            "ERROR: could not unambiguously extract snapshot_dt/as_of_load_ts/"
            "snapshot_ts from data/borrower_history.txt's snapshots CTE -- "
            "refusing to generate SQL against a mismatched or guessed cutoff.\n"
            + "\n".join(errors)
            + "\nEither the snapshots CTE has changed shape (update "
            "_SNAPSHOT_PARAM in this script) or a duplicate/ambiguous match "
            "exists elsewhere in the file."
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
    print(f"-- Runs as three statements: the checkpoint table ({checkpoint_table})")
    print("-- is dropped and rebuilt first, then vw_bh_output is (re)created from it.")
    print(f"DROP TABLE IF EXISTS {checkpoint_table};")
    print(f"CREATE TABLE {checkpoint_table} AS")
    print(part_a.rstrip())
    print(";")
    print()
    print(f"CREATE OR REPLACE VIEW {schema}.vw_bh_output AS")
    print(part_b.rstrip())
    print(";")


if __name__ == "__main__":
    main()
