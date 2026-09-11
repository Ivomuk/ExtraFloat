-- ============================================================================
-- ever_anomaly_open_export.sql
-- ============================================================================
-- ever_anomaly_open (loan_ever_anomaly_open CTE, borrower_history.txt
-- checkpoint 1) is computed but never surfaced past tbl_bh_loan_final --
-- it stops short of vw_bh_output by design, to keep the final 50-column
-- contract unchanged (see borrower_history.txt's header comment / KNOWN
-- LIMITATION note). tbl_bh_loan_final is still a real, queryable table
-- though (it's the checkpoint-1 CREATE TABLE build_vw_bh_output.py
-- generates), so the flag is one query away without touching the view.
--
-- loan_final is loan-grain (one row per disbursement_fid); this collapses
-- to phonenumber-grain (bool_or -- true if ANY of the borrower's loans was
-- ever ANOMALY_OPEN) to match wl_bl_eval_matched_agents.csv's per-agent
-- grain for the join in
-- scripts/check_defaulter_visibility_at_snapshot.py.
--
-- Replace `your_schema` below with whatever schema
-- scripts/build_vw_bh_output.py was actually run against.
-- ============================================================================

SELECT
    phonenumber,
    bool_or(COALESCE(ever_anomaly_open, false)) AS ever_anomaly_open
FROM your_schema.tbl_bh_loan_final
GROUP BY phonenumber;

-- Export the result as CSV to data/ever_anomaly_open.csv (or any path),
-- then pass it to the diagnostic:
--   python scripts\check_defaulter_visibility_at_snapshot.py ^
--       --matched-file wl_bl_eval_matched_agents.csv ^
--       --ever-anomaly-open-file data\ever_anomaly_open.csv
