-- ============================================================================
-- anomaly_open_fix_verification.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. anomaly_open_b2b_correlation_test.sql
-- quantified an 8.7-point exact-match gap (70.3% vs 61.6%) for loans EVER
-- ANOMALY_OPEN, by re-deriving attribution from the raw tables and
-- comparing it to lifetime_repaid_ugx -- that diagnostic never touched
-- borrower_history.txt itself, so it can't prove the fix actually closes
-- the gap in the real pipeline, only that the underlying mismatch exists
-- in the data. The fix in borrower_history.txt's loan_final overrides
-- total_recovered/total_cure_cashflow with ls_lifetime_repaid_ugx directly
-- for any loan where ever_anomaly_open is true -- so THIS checks the fix
-- against production's own materialized checkpoint 1 output
-- ({{CHECKPOINT_TABLE_1}} / tbl_bh_loan_final), which now carries
-- total_recovered (corrected), ls_lifetime_repaid_ugx, and ever_anomaly_open
-- all as columns on the same row after the fix.
--
-- Requires the current checkpoint 1 table to already exist -- it's the
-- second of the four statements scripts/build_vw_bh_output.py generates,
-- so if vw_bh_output has been rebuilt for GATE 1 (as it has), this table
-- already exists in the warehouse; no extra build step needed.
-- ============================================================================

SELECT
ever_anomaly_open,
COUNT(*) AS n_loans,
COUNT_IF(ls_lifetime_repaid_ugx IS NULL) AS n_no_loan_state_row,
ROUND(100.0 * COUNT_IF(ls_lifetime_repaid_ugx IS NULL) / NULLIF(COUNT(*), 0), 2) AS pct_no_loan_state_row,
COUNT_IF(ABS(total_recovered - ls_lifetime_repaid_ugx) <= 1) AS n_exact_match,
ROUND(100.0 * COUNT_IF(ABS(total_recovered - ls_lifetime_repaid_ugx) <= 1)
/ NULLIF(COUNT(*), 0), 2) AS pct_exact_match_overall,
COUNT_IF(ls_lifetime_repaid_ugx IS NOT NULL
AND ABS(total_recovered - ls_lifetime_repaid_ugx) <= 1) AS n_exact_match_when_ls_available,
ROUND(100.0 * COUNT_IF(ls_lifetime_repaid_ugx IS NOT NULL
AND ABS(total_recovered - ls_lifetime_repaid_ugx) <= 1)
/ NULLIF(COUNT_IF(ls_lifetime_repaid_ugx IS NOT NULL), 0), 2) AS pct_exact_match_when_ls_available
FROM your_schema.tbl_bh_loan_final
GROUP BY ever_anomaly_open
ORDER BY ever_anomaly_open;
-- RESULT:
-- INTERPRETATION: this is the direct before/after proof, not a re-derived
-- proxy. For ever_anomaly_open = true:
--   - pct_exact_match_when_ls_available should be ~100% (not just improved
--     -- the fix is a direct substitution of total_recovered with
--     ls_lifetime_repaid_ugx, so once a loan_state_daily row exists, exact
--     match is close to tautological by construction; a value meaningfully
--     below 100% here would mean the CASE/JOIN in loan_final has a bug, not
--     that the fix is merely imperfect -- investigate the join/COALESCE
--     logic rather than accepting this as a residual limitation).
--   - pct_no_loan_state_row is the size of the ONE remaining, already-
--     documented gap: loans with no loan_state_daily row at all (the
--     same-day-second-disbursement limitation noted in loan_final's ls_*
--     comment) still fall back to the old, potentially misattributed
--     total_recovered. pct_exact_match_overall for ever_anomaly_open = true
--     should sit close to (1 - pct_no_loan_state_row) if that residual
--     population's fallback value is essentially never coincidentally
--     exact -- confirms both HOW MUCH of the original 8.7-point gap closed
--     and exactly WHERE the remainder lives, rather than a vague "should be
--     small."
-- For ever_anomaly_open = false: nothing in loan_final changed for these
-- loans (the CASE's ELSE branch is byte-identical to the pre-fix formula),
-- so pct_exact_match_overall here is a continuity check, not a fix target
-- -- expect it to land close to anomaly_open_b2b_correlation_test.sql's
-- 70.3% (small differences are expected: that test re-derives attribution
-- standalone from raw tables, this reads production's own total_recovered,
-- which is built through {{CHECKPOINT_TABLE_0}}'s classified/loan_core
-- rather than being recomputed independently -- a large divergence here
-- would flag a methodology mismatch worth chasing, not the fix itself).
