-- ============================================================================
-- recheck_stale_loan_final.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. loan_core.disbursement_ts uses the exact same
-- computation (MIN(CASE WHEN txn_type='disbursement' THEN event_ts END)
-- FROM tbl_bh_classified) as 24h_single_table_check.sql's disb_event_ts,
-- which just confirmed elapsed_seconds > 86400 (genuinely past 24h) for
-- these 5 loans' recovery events, using ONLY tbl_bh_classified with no
-- cross-table join. Production's checkpoints CTE should therefore compute
-- the same >24h result and exclude these events from recovery_24h -- so
-- before concluding there's a real bug in the FILTER logic, rule out the
-- simpler explanation: tbl_bh_loan_final wasn't rebuilt in the same run as
-- tbl_bh_classified (there have been at least two different vw_bh_output.sql
-- runs this session -- before and after the source-table cleanup) and is
-- simply stale. This re-checks recovery_24h fresh, right now.
-- ============================================================================

SELECT requestid, phonenumber, disbursement_ts, disbursed_amount,
principal_cure_ts, hours_to_principal_cure, recovery_24h,
on_time_24h_flag, default_24h_flag
FROM analytics.tbl_bh_loan_final
WHERE requestid IN (
38485872800, 39328881491, 37673344680, 38485874433, 37952520161
);
-- RESULT:
-- INTERPRETATION: if recovery_24h still shows the full disbursed_amount
-- here, tbl_bh_loan_final is self-consistent with itself (not stale
-- relative to a prior run) and there IS a genuine discrepancy between what
-- the SQL text says should happen and what production actually computed --
-- worth escalating as a real Trino behavior question rather than chasing
-- further hypotheses blind. If it now shows 0 (matching hand/single-table
-- calculation), tbl_bh_loan_final WAS stale when the original
-- 24h_complement_diagnostic.sql sample was pulled -- re-run the full
-- vw_bh_output.sql generation fresh and redo GATE 1b to get a real
-- bad_24h_complement_identity count before concluding anything further.
