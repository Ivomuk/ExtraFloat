-- ============================================================================
-- 24h_single_table_check.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. 24h_filter_reproduction.sql's join back to
-- tbl_bh_loan_final (to get disbursement_ts via a correlated subquery on
-- phonenumber) returned phonenumber values that look one digit shorter
-- than what appeared in the raw trace (25676 vs 256766, etc.) -- possibly
-- just a display/paste artifact, but possibly a real type mismatch
-- silently matching the wrong row and shifting the disbursement_ts used in
-- the hours calculation. This eliminates that risk entirely by computing
-- everything from tbl_bh_classified alone -- no join to any other table --
-- using the disbursement row's OWN event_ts as the anchor, and shows the
-- exact hours value at full precision (no rounding) so there's no
-- ambiguity left about what Trino actually computes.
-- ============================================================================

WITH events AS (
SELECT
requestid,
phonenumber,
event_ts,
txn_type,
amount,
MIN(CASE WHEN txn_type = 'disbursement' THEN event_ts END) OVER (
PARTITION BY requestid, phonenumber
) AS disb_event_ts
FROM analytics.tbl_bh_classified
WHERE requestid IN (
38485872800, 39328881491, 37673344680, 38485874433, 37952520161
)
)
SELECT
requestid,
phonenumber,
txn_type,
event_ts,
disb_event_ts,
date_diff('millisecond', disb_event_ts, event_ts) AS elapsed_milliseconds,
date_diff('second', disb_event_ts, event_ts) AS elapsed_seconds,
ROUND(date_diff('second', disb_event_ts, event_ts) / 3600.0, 6) AS hours_since_disbursement_exact
FROM events
ORDER BY requestid, event_ts;
-- RESULT:
-- INTERPRETATION: elapsed_seconds/elapsed_milliseconds is ground truth, no
-- rounding possible. If it shows ~86440-86550 seconds (24h + 40sec to
-- 2.5min, matching the earlier hand calculation) yet hours_since_
-- disbursement_exact still rounds to <= 24.0 somehow, that points at a
-- date_diff/division behavior worth escalating directly. If elapsed_seconds
-- instead comes back <= 86400 (i.e. the recovery event is NOT actually past
-- 24h once computed this way, contradicting the earlier by-hand timestamp
-- subtraction), the earlier phonenumber-join theory was right and this
-- single-table version gives the true, uncontaminated numbers.
