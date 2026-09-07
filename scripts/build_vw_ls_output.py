"""
Assembles the single CREATE VIEW statement used by
data/loan_summary_query_validation_queries.sql's GATE 1 final-output
checks, from the checked-in data/loan_summary_query.txt -- so validation
always runs against the exact production query, not a manually pasted
(and potentially stale or hand-edited) copy. Mirrors
scripts/build_vw_bh_output.py's rationale for borrower_history.txt, but
much simpler: loan_summary_query.txt has no checkpoint markers and runs as
one statement (no stage-count ceiling concerns observed at design time --
see that file's own EXECUTION comment), so there is nothing to split.

Stamps the git commit SHA of data/loan_summary_query.txt into the output
as a comment, and warns if the working tree has uncommitted changes to
it -- both so the SHA can be recorded in GATE 6 as evidence of exactly
what was validated.

Usage:
    python scripts/build_vw_ls_output.py <validation_schema> > vw_ls_output.sql
    # then run vw_ls_output.sql in Athena/Trino -- it contains one
    # statement: CREATE OR REPLACE VIEW <schema>.vw_ls_output AS
    # <loan_summary_query.txt's query body, verbatim>.

Testing against a partial/in-progress reload
---------------------------------------------
The checked-in snapshot_dt/as_of_load_ts are PRODUCTION values -- they may
be later than what's actually been reloaded into the warehouse (e.g. data
loaded only through 2026-04-14 while the file's snapshot_dt is 20260731).
Running the production literals against partial data does not error --
it silently produces truncated/empty recent-activity windows for
everything after the load cutoff, which looks like "no activity" rather
than "not loaded yet."

Rather than hand-edit the checked-in file to test against partial data
(reintroducing the exact drift risk this script exists to avoid), pass
--snapshot-dt / --as-of-load-ts to substitute test-time values ONLY in the
generated output -- data/loan_summary_query.txt itself is never modified.
The header comment records both the production literals and the override
actually used, so GATE 6 can capture which cutoff was really validated.

    python scripts/build_vw_ls_output.py <validation_schema> \\
        --snapshot-dt 20260414 \\
        --as-of-load-ts "2026-04-15 00:00:00.000" \\
        > vw_ls_output_test.sql

Use the SAME overridden values for :snapshot_dt / :as_of_load_ts when
running loan_summary_query_validation_queries.sql's GATE 0 views for this
test -- otherwise GATE 0 and vw_ls_output validate different cutoffs.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "loan_summary_query.txt"

QUERY_START_MARKER = "SELECT *"

# Athena/Trino unquoted identifier: letters/digits/underscore, not starting
# with a digit. schema is interpolated directly into
# `CREATE OR REPLACE VIEW {schema}.vw_ls_output` below -- reject anything
# else rather than emit malformed or unintended SQL from a typo'd or
# pasted-wrong argument.
_VALID_SCHEMA = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Matches the two literals in loan_summary_query.txt's `snapshots` CTE, e.g.
#   cast(20260731 as bigint) as snapshot_dt,
#   cast('2026-08-20 00:00:00.000' as timestamp) as as_of_load_ts
_SNAPSHOT_PARAM = re.compile(
    r"cast\(\s*('[^']*'|\d+)\s+as\s+\w+\s*\)\s+as\s+(snapshot_dt|as_of_load_ts)\b",
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
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the vw_ls_output CREATE VIEW statement from the "
            "checked-in data/loan_summary_query.txt."
        ),
    )
    parser.add_argument("validation_schema")
    parser.add_argument(
        "--snapshot-dt",
        metavar="YYYYMMDD",
        help=(
            "Override snapshot_dt (plain integer, e.g. 20260414) for THIS "
            "generated output only -- data/loan_summary_query.txt is never "
            "modified. Use when testing against a partial/in-progress "
            "reload that doesn't yet reach the file's checked-in "
            "production snapshot_dt."
        ),
    )
    parser.add_argument(
        "--as-of-load-ts",
        metavar="'YYYY-MM-DD HH:MM:SS.fff'",
        help=(
            "Override as_of_load_ts (unquoted, e.g. '2026-04-15 "
            "00:00:00.000') for THIS generated output only -- "
            "data/loan_summary_query.txt is never modified."
        ),
    )
    args = parser.parse_args()

    schema = args.validation_schema
    if not _VALID_SCHEMA.match(schema):
        sys.exit(
            f"ERROR: '{schema}' is not a valid unquoted SQL identifier "
            "(letters/digits/underscore, not starting with a digit). "
            "Refusing to interpolate it into a CREATE VIEW statement."
        )
    if args.snapshot_dt is not None and not re.fullmatch(r"\d+", args.snapshot_dt):
        sys.exit(
            f"ERROR: --snapshot-dt must be a plain integer like 20260414, "
            f"got {args.snapshot_dt!r}."
        )

    if not SRC.exists():
        sys.exit(f"ERROR: {SRC} not found")

    sha = _git("log", "-1", "--format=%H", "--", "data/loan_summary_query.txt") or "unknown"
    dirty = bool(_git("status", "--porcelain", "--", "data/loan_summary_query.txt"))
    dirty_note = " -- WORKING TREE HAS UNCOMMITTED CHANGES TO THIS FILE, SHA ABOVE IS STALE" if dirty else ""

    body = SRC.read_text()

    # Fail closed: a missing or duplicated query-start marker means the
    # file's shape changed in a way this script no longer understands --
    # silently guessing where the header ends and the query begins could
    # emit SQL that either won't run or silently validates the wrong thing.
    idx = body.find(f"\n{QUERY_START_MARKER}\n")
    if idx == -1 or body.find(f"\n{QUERY_START_MARKER}\n", idx + 1) != -1:
        sys.exit(
            f"ERROR: could not find exactly one {QUERY_START_MARKER!r} line "
            "in data/loan_summary_query.txt marking where the header comment "
            "ends and the query body begins. Either the file's shape changed "
            "(update QUERY_START_MARKER in this script) or the marker is "
            "missing/duplicated -- refusing to guess where to split."
        )
    query_body = body[idx + 1:]

    # Extract the literal snapshot_dt/as_of_load_ts baked into THIS file, so
    # whoever runs the validation queries copies the exact values instead of
    # retyping them from memory or a stale note -- the generated
    # vw_ls_output and the hand-run GATE 0 views silently drifting onto
    # different cutoffs would invalidate every check without necessarily
    # looking wrong. Matched within query_body (not the header comment)
    # since that's the text actually spliced/emitted below.
    matches_by_name: dict[str, list[re.Match]] = {"snapshot_dt": [], "as_of_load_ts": []}
    for m in _SNAPSHOT_PARAM.finditer(query_body):
        matches_by_name[m.group(2).lower()].append(m)

    errors = [
        f"  {name}: found {len(ms)} matches, expected exactly 1"
        for name, ms in matches_by_name.items()
        if len(ms) != 1
    ]
    if errors:
        sys.exit(
            "ERROR: could not unambiguously extract snapshot_dt/as_of_load_ts "
            "from data/loan_summary_query.txt's snapshots CTE -- refusing to "
            "generate SQL against a mismatched or guessed cutoff.\n"
            + "\n".join(errors)
            + "\nThe snapshots CTE has changed shape -- update _SNAPSHOT_PARAM "
            "in this script."
        )
    production_params = {name: ms[0].group(1) for name, ms in matches_by_name.items()}

    # Splice overrides into the exact matched literal spans (not a blind
    # string replace) so nothing else in the query body that happens to
    # contain the same digits/text is touched. Apply in descending start
    # order so earlier spans' offsets stay valid after later substitutions.
    override_values = {
        "snapshot_dt": args.snapshot_dt,
        "as_of_load_ts": f"'{args.as_of_load_ts}'" if args.as_of_load_ts is not None else None,
    }
    effective_params = dict(production_params)
    spans = []
    for name, new_text in override_values.items():
        if new_text is None:
            continue
        m = matches_by_name[name][0]
        spans.append((m.start(1), m.end(1), new_text))
        effective_params[name] = new_text
    for start, end, new_text in sorted(spans, key=lambda s: s[0], reverse=True):
        query_body = query_body[:start] + new_text + query_body[end:]

    print("-- Auto-generated by scripts/build_vw_ls_output.py -- DO NOT EDIT BY HAND.")
    print(f"-- Built from data/loan_summary_query.txt @ commit {sha}{dirty_note}")
    print("-- Record this commit SHA in GATE 6 of loan_summary_query_validation_queries.sql.")
    print("--")
    if spans:
        print("-- OVERRIDE IN EFFECT -- this view does NOT use loan_summary_query.txt's")
        print("-- checked-in PRODUCTION cutoff (e.g. for testing against a partial/")
        print("-- in-progress reload). data/loan_summary_query.txt itself is unmodified:")
        for name in ("snapshot_dt", "as_of_load_ts"):
            if name in [n for n, v in override_values.items() if v is not None]:
                print(f"--   {name}: production {production_params[name]} -> used {effective_params[name]}")
        print("--")
    print("-- Use these EXACT values for :snapshot_dt / :as_of_load_ts when")
    print("-- running loan_summary_query_validation_queries.sql -- if they don't")
    print("-- match what you substitute there, GATE 0's views and this production")
    print("-- view are validating two different cutoffs without any error raised.")
    for name in ("snapshot_dt", "as_of_load_ts"):
        print(f"--   {name}: {effective_params[name]}")
    print("--")
    print(f"CREATE OR REPLACE VIEW {schema}.vw_ls_output AS")
    print(query_body.rstrip().rstrip(";"))
    print(";")


if __name__ == "__main__":
    main()
