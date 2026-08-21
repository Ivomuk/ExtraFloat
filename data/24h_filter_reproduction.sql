-- ============================================================================
-- 24h_filter_reproduction.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. By hand, from 24h_raw_event_trace.sql's raw
-- events: each of these 5 loans has exactly one disbursement event
-- (cumulative_cure=0) and one recovery event landing 24h + 40sec to 2.5min
-- AFTER disbursement_ts (confirmed by exact timestamp subtraction, not a
-- rounding artifact). checkpoints' recovery_24h = COALESCE(MAX(cumulative_
-- cure) FILTER (WHERE hours_since_disbursement <= 24), 0) should therefore
-- exclude the recovery row (since it's >24h) and evaluate to 0 for all 5 --
-- but tbl_bh_loan_final.recovery_24h actually shows the full
-- disbursed_amount for these same loans. This directly reproduces
-- checkpoints' exact FILTER-aggregate expression from the raw event data,
-- so we can see whether Trino's FILTER behaves as expected here (recovery_
-- 24h_reproduced should be 0) or not (matching the unexpected stored
-- value) -- isolating an engine-behavior question from anything else.
-- ============================================================================

WITH raw_events AS (
SELECT
c.requestid,
c.phonenumber,
c.event_ts,
CASE WHEN c.txn_type IN ('recovery','reimbursement') THEN c.amount ELSE 0 END AS cure_amount,
(SELECT MIN(disbursement_ts) FROM analytics.tbl_bh_loan_final lf WHERE lf.requestid = c.requestid AND lf.phonenumber = c.phonenumber) AS disbursement_ts
FROM analytics.tbl_bh_classified c
WHERE requestid IN (
38485872800, 39328881491, 37673344680, 38485874433, 37952520161
)
),
cure_progress_repro AS (
SELECT
requestid,
phonenumber,
event_ts,
cure_amount,
SUM(cure_amount) OVER (
PARTITION BY requestid, phonenumber ORDER BY event_ts
) AS cumulative_cure,
date_diff('second', disbursement_ts, event_ts) / 3600.0 AS hours_since_disbursement
FROM raw_events
)
SELECT
requestid,
phonenumber,
COALESCE(MAX(cumulative_cure) FILTER (WHERE hours_since_disbursement <= 24), 0) AS recovery_24h_reproduced,
MAX(cumulative_cure) AS total_cure_cashflow_reproduced,
MAX(hours_since_disbursement) AS max_hours_since_disbursement_in_group,
MIN(CASE WHEN hours_since_disbursement <= 24 THEN cumulative_cure END) AS min_cumulative_cure_within_24h,
MAX(CASE WHEN hours_since_disbursement <= 24 THEN cumulative_cure END) AS max_cumulative_cure_within_24h,
COUNT(CASE WHEN hours_since_disbursement <= 24 THEN 1 END) AS n_rows_within_24h,
COUNT(*) AS n_rows_total
FROM cure_progress_repro
GROUP BY requestid, phonenumber;
-- RESULT:
-- INTERPRETATION: if recovery_24h_reproduced comes back 0 for all 5 (as
-- hand arithmetic predicts), the bug is NOT in the FILTER logic itself --
-- something else differs between this reproduction and the actual
-- production checkpoints CTE (e.g. tbl_bh_loan_final is stale from a
-- different run, or the real checkpoints/cure_progress_base text differs
-- from what's in the current borrower_history.txt). If recovery_24h_
-- reproduced ALSO comes back as the full disbursed_amount (matching the
-- unexpected production value), that's a genuine, confirmed Trino FILTER-
-- clause behavior surprise worth escalating, not a query-authoring bug.
