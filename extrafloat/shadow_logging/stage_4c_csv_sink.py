"""
extrafloat/shadow_logging/stage_4c_csv_sink.py
=================================================
The Stage 4C Decision Log sink adapter -- CSV-files-in-a-folder, per the
reconnaissance answer (Stage 4C-S1): production currently runs entirely
on CSV files exported from the data warehouse, with no live database
connection anywhere in this codebase (confirmed by a repo-wide search --
no DB client library, no connection string, no existing write-to-
warehouse pattern anywhere). The user's own words: "Let's have a folder
where files are saved. Similar to the existing structure. We will later
on have these files pushed to a database" -- so this sink is explicitly
an INTERIM mechanism, not the final home for this data.

"Similar to the existing structure" / "same convention as other outputs"
means the same mkdir(parents=True, exist_ok=True) + to_csv(index=False)
mechanism run_credit_risk_pipeline.py's own --output flag already uses
for the engine's main output (run_credit_risk_pipeline.py:771-773).

ONE DELIBERATE DIFFERENCE from that existing convention, justified
explicitly: the engine's main --output file is a single current-state
snapshot, overwritten every run by design (it only ever needs to reflect
"the latest assigned_limit"). The Decision Log is an append-only
HISTORICAL record -- overwriting it every run would destroy the ability
to validate anything prospectively. So this sink writes ONE FILE PER
run_id into the configured directory, never overwriting a different
run's file, rather than writing into one shared file that every run
would clobber.

ATOMICITY: every write goes to a temp file in the SAME directory first,
then an atomic os.replace() onto the final filename. On a local or
network filesystem (confirmed: this is NOT object storage) os.replace()
is atomic -- a reader (including the future DB-push process this sink
is an interim step for) can never observe a partially-written file; it
either sees the complete prior file, or the complete new one, never
something in between.

IDEMPOTENCY on retry: if the final file for a given run_id already
exists, write_decision_log_rows() treats that as a sign the run already
completed successfully and returns without re-writing -- never raises,
never silently corrupts an existing file by overwriting it with
(possibly different) retry data. This is possible ONLY because of the
atomicity guarantee above: a final file existing at all means a prior
attempt's to_csv + os.replace() both completed, so there is no partial-
write case to worry about distinguishing.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DECISION_LOG_FILENAME_PREFIX = "stage_4c_decision_log"
RUN_MANIFEST_FILENAME_PREFIX = "stage_4c_run_manifest"


def _sanitize_run_id_for_filename(run_id) -> str:
    """run_id is typically an ISO-8601-with-'Z' timestamp string
    (":" and other characters are invalid or awkward in filenames on
    some filesystems) -- replace anything outside [A-Za-z0-9_.-] with
    "_", never silently truncate or hash it away (the sanitized name
    must still be visually traceable back to the real run_id)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(run_id))


class CsvDecisionLogSink:
    """Append-only, one-file-per-run-id CSV sink, writing into a single
    configured output directory. Implements the duck-typed sink protocol
    extrafloat.shadow_logging.stage_4c_decision_logger expects:
    write_decision_log_rows(df) and write_manifest(manifest_dict)."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def _decision_log_path(self, run_id) -> Path:
        return self.output_dir / f"{DECISION_LOG_FILENAME_PREFIX}_{_sanitize_run_id_for_filename(run_id)}.csv"

    def _manifest_path(self, run_id) -> Path:
        return self.output_dir / f"{RUN_MANIFEST_FILENAME_PREFIX}_{_sanitize_run_id_for_filename(run_id)}.json"

    def _write_atomic_text(self, final_path: Path, write_fn) -> None:
        """write_fn(tmp_path) must write the complete file content to
        tmp_path. Writes to a temp file in the SAME directory (so the
        final os.replace() is a same-filesystem rename, not a cross-
        filesystem copy, which would not be atomic), then atomically
        replaces final_path. Cleans up the temp file on any failure --
        a failed write must never leave a stray temp file behind."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = final_path.with_name(final_path.name + f".tmp{os.getpid()}")
        try:
            write_fn(tmp_path)
            os.replace(tmp_path, final_path)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    logger.exception("CsvDecisionLogSink: failed to clean up temp file %s after a failed write", tmp_path)
            raise

    def write_decision_log_rows(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("CsvDecisionLogSink.write_decision_log_rows: refusing to write an empty DataFrame.")
        run_ids = df["run_id"].unique()
        if len(run_ids) != 1:
            raise ValueError(
                f"CsvDecisionLogSink.write_decision_log_rows: expected exactly one distinct run_id per "
                f"write (this sink writes one file per run), got {len(run_ids)}: {list(run_ids)[:5]}"
            )
        run_id = run_ids[0]
        final_path = self._decision_log_path(run_id)

        if final_path.exists():
            # Idempotent retry: a complete file for this run_id can only
            # exist if a prior attempt's write_fn + os.replace() both
            # already completed successfully (see module docstring) --
            # treat this as "already done," not an error, and never
            # overwrite it with what might be different retry data.
            logger.info(
                "CsvDecisionLogSink: Decision Log file already exists for run_id=%s at %s -- "
                "treating this write as an idempotent no-op (not re-writing).",
                run_id, final_path,
            )
            return

        self._write_atomic_text(final_path, lambda tmp_path: df.to_csv(tmp_path, index=False))
        logger.info("CsvDecisionLogSink: wrote %d Decision Log row(s) for run_id=%s to %s", len(df), run_id, final_path)

    def write_manifest(self, manifest: dict) -> None:
        run_id = manifest.get("run_id")
        if run_id is None:
            raise ValueError("CsvDecisionLogSink.write_manifest: manifest is missing run_id.")
        final_path = self._manifest_path(run_id)
        # Manifests are small metadata, not bulk data, and always written
        # unconditionally (never skipped as "already done") -- a retry's
        # manifest write must always reflect the outcome of THIS attempt,
        # so a later attempt's SUCCESS can correct an earlier FAILED
        # manifest for the same run_id. The write itself is still atomic
        # (temp + replace), so a reader never observes a half-written
        # manifest file.
        self._write_atomic_text(
            final_path, lambda tmp_path: tmp_path.write_text(json.dumps(manifest, default=str, indent=2))
        )

    def read_manifest(self, run_id) -> dict | None:
        """Convenience for retry logic / tests: returns the manifest
        dict for run_id if one has been written, else None. Not part of
        the sink protocol stage_4c_decision_logger requires -- callers
        that want to inspect prior-run state before deciding whether to
        retry can use this directly."""
        path = self._manifest_path(run_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def has_decision_log(self, run_id) -> bool:
        """Convenience for retry logic / tests: true iff a complete
        Decision Log file already exists for run_id."""
        return self._decision_log_path(run_id).exists()
