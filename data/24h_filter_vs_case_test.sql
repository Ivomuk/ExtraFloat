-- ============================================================================
-- 24h_filter_vs_case_test.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. 24h_clean_filter_test.sql conclusively showed
-- COALESCE(MAX(cumulative_cure) FILTER (WHERE hours_since_disbursement <=
-- 24),0) including a row whose own hours_since_disbursement is unambiguously
-- > 24 (elapsed_seconds 86,493-86,553, all > 86,400), with zero joins and
-- zero staleness risk in the test. This compares that exact FILTER
-- expression against a logically-equivalent CASE-based formulation on the
-- SAME data in the SAME query, to isolate whether FILTER specifically is
-- the problem (a real engine-behavior finding, and a one-line fix: replace
-- FILTER with CASE in checkpoints) or something else entirely.
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
date_diff('second', disb_event_ts, event_ts) / 3600.0 AS hours_since_disbursement
FROM events
)
SELECT
requestid,
phonenumber,
COALESCE(MAX(cumulative_cure) FILTER (WHERE hours_since_disbursement <= 24), 0) AS recovery_24h_via_filter,
COALESCE(MAX(CASE WHEN hours_since_disbursement <= 24 THEN cumulative_cure END), 0) AS recovery_24h_via_case
FROM cure_progress_clean
GROUP BY requestid, phonenumber;
-- RESULT:
-- INTERPRETATION: these two columns are logically equivalent SQL and MUST
-- produce the same value for every row. If recovery_24h_via_filter shows
-- the full disbursed_amount while recovery_24h_via_case shows 0, FILTER is
-- confirmed as the specific broken construct -- the fix is a one-line
-- swap in borrower_history.txt's checkpoints CTE: replace `MAX(x) FILTER
-- (WHERE cond)` with `MAX(CASE WHEN cond THEN x END)` for recovery_6h/12h/
-- 24h/26h. If both columns show the same (wrong) value, the problem is
-- upstream of the aggregate step and this needs further investigation.
