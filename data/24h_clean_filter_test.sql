-- ============================================================================
-- 24h_clean_filter_test.sql -- ad-hoc diagnostic, not part of the committed
-- validation suite. Final, fully clean reproduction combining what the two
-- prior diagnostics each got half of:
--   - 24h_single_table_check.sql: computed disb_event_ts via a window
--     function WITHIN tbl_bh_classified alone (no cross-table join, so no
--     masking-mismatch risk) -- confirmed elapsed_seconds > 86400 (genuinely
--     past 24h) for these events. But it never applied the FILTER aggregate.
--   - 24h_filter_reproduction.sql: applied the FILTER aggregate
--     (COALESCE(MAX(cumulative_cure) FILTER (WHERE hours_since_disbursement
--     <= 24),0)) but got disbursement_ts via a correlated subquery JOINING
--     tbl_bh_classified to tbl_bh_loan_final on phonenumber -- a join that,
--     given msisdns are masked, could in principle have matched the wrong
--     row and contaminated the result.
-- This uses the single-table (no-join) disb_event_ts AND applies the
-- filtered aggregate, closing the gap between the two.
-- ============================================================================

WITH events AS (
SELECT
requestid,
phonenumber,
event_ts,
CASE WHEN txn_type IN ('recovery','reimbursement') THEN amount ELSE 0 END AS cure_amount,
MIN(CASE WHEN txn_type = 'disbursement' THEN event_ts END) OVER (
PARTITION BY requestid, phonenumber
) AS disb_event_ts
FROM analytics.tbl_bh_classified
WHERE requestid IN (
38485872800, 39328881491, 37673344680, 38485874433, 37952520161
)
),
cure_progress_clean AS (
SELECT
requestid,
phonenumber,
event_ts,
cure_amount,
SUM(cure_amount) OVER (
PARTITION BY requestid, phonenumber ORDER BY event_ts
) AS cumulative_cure,
date_diff('second', disb_event_ts, event_ts) AS elapsed_seconds,
date_diff('second', disb_event_ts, event_ts) / 3600.0 AS hours_since_disbursement
FROM events
)
SELECT
requestid,
phonenumber,
COALESCE(MAX(cumulative_cure) FILTER (WHERE hours_since_disbursement <= 24), 0) AS recovery_24h_clean,
MAX(elapsed_seconds) AS max_elapsed_seconds_in_group,
COUNT(CASE WHEN hours_since_disbursement <= 24 THEN 1 END) AS n_rows_within_24h_clean,
COUNT(*) AS n_rows_total
FROM cure_progress_clean
GROUP BY requestid, phonenumber;
-- RESULT:
-- INTERPRETATION: this is fully clean -- no cross-table join anywhere,
-- disb_event_ts computed identically to loan_core.disbursement_ts's own
-- MIN(CASE WHEN txn_type='disbursement'...) FROM the same source table. If
-- recovery_24h_clean comes back 0 here (matching hand-calculation, unlike
-- the earlier join-based reproduction), the masking-mismatch theory was
-- correct and the earlier "bug" was an artifact of that join. If it STILL
-- comes back as the full disbursed_amount despite max_elapsed_seconds_
-- in_group clearly exceeding 86400, that's conclusive: the FILTER clause is
-- not excluding the row it should, a genuine, confirmed defect independent
-- of any join or staleness explanation.
