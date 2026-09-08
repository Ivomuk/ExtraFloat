-- ============================================================================
-- anomaly_open_b2b_correlation_test.sql -- ad-hoc diagnostic, not part of
-- the committed validation suite. Mirrors repayment_uid_cross_customer_
-- correlation_test.sql's structure exactly, but for a different exposure:
-- anomaly_open_aging_only_test.sql confirmed a second loan is required to
-- ever trigger ANOMALY_OPEN, and anomaly_open_raw_timeline_examples.sql
-- showed the mechanism is a phonenumber-window attribution problem, not a
-- balance transfer -- repayments made after the second loan's
-- disbursement get counted toward the WRONG loan by borrower_history.txt's
-- timing heuristic, even though the source system correctly keeps
-- crediting the original loan. This quantifies how much of that actually
-- shows up as B2b mismatch: cross-tabulates "was this loan EVER
-- ANOMALY_OPEN at some point in its own history" (ground truth, from the
-- full daily loan_state_daily timeline, not just the latest snapshot --
-- most such loans later resolve to CLOSED and are NOT excluded by the
-- production reconciliation's current-status anomaly filter) against
-- exact-match rate, for the SAME surviving-loan population
-- b2b_reconciliation_*.sql already scores.
--
-- Self-contained (raw tables), bounded to date_key <= 20260609.
-- ============================================================================

WITH disb_dedup AS (
SELECT disbursement_fid, phonenumber, disbursement_ts, disbursed_amount
FROM (
SELECT d.*,
ROW_NUMBER() OVER (PARTITION BY disbursement_fid ORDER BY inserted_ts DESC) rn
FROM (
SELECT
disbursement_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(disbursement_ts AS timestamp) AS disbursement_ts,
cast(disbursement_amount_ugx AS double) AS disbursed_amount,
inserted_ts
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND try_cast(disbursement_ts AS timestamp) IS NOT NULL
AND disbursement_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date_key <= 20260609
) d
)
WHERE rn = 1
),
disb_windows AS (
SELECT
disbursement_fid, phonenumber, disbursement_ts, disbursed_amount,
LEAD(disbursement_ts) OVER (PARTITION BY phonenumber ORDER BY disbursement_ts, disbursement_fid) AS next_disbursement_ts
FROM disb_dedup
),
repay_dedup AS (
SELECT phonenumber, repayment_ts, repayment_amount
FROM (
SELECT r.*,
ROW_NUMBER() OVER (PARTITION BY repayment_fid ORDER BY inserted_ts DESC) rn
FROM (
SELECT
repayment_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(repayment_ts AS timestamp) AS repayment_ts,
cast(repayment_amount_ugx AS double) AS repayment_amount,
inserted_ts
FROM analytics.momo_loan_book_tracker_repayments_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND try_cast(repayment_ts AS timestamp) IS NOT NULL
AND repayment_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date_key <= 20260609
) r
)
WHERE rn = 1
),
loan_state_latest AS (
SELECT disbursement_fid, loan_status, is_anomaly_open, lifetime_repaid_ugx
FROM (
SELECT lsld.*,
ROW_NUMBER() OVER (PARTITION BY disbursement_fid ORDER BY date_key DESC) rn2
FROM (
SELECT *
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (PARTITION BY disbursement_fid, date_key ORDER BY inserted_ts DESC) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE ova = 'XTRAFLOAT-AGENT'
AND disbursement_fid IS NOT NULL
AND date_key <= 20260609
)
WHERE rn = 1
) lsld
)
WHERE rn2 = 1
),
loan_state_ever_anomaly AS (
SELECT disbursement_fid,
MAX(CASE WHEN loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true THEN 1 ELSE 0 END) AS is_exposed
FROM (
SELECT lsd.disbursement_fid, lsd.loan_status, lsd.is_anomaly_open
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (PARTITION BY disbursement_fid, date_key ORDER BY inserted_ts DESC) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE ova = 'XTRAFLOAT-AGENT'
AND disbursement_fid IS NOT NULL
AND date_key <= 20260609
) lsd
WHERE rn = 1
) lsd
GROUP BY disbursement_fid
),
loan_state_current_anomalies AS (
SELECT disbursement_fid FROM loan_state_latest
WHERE loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true
),
surviving_loans AS (
-- SAME population the production b2b reconciliation scores today: current
-- status not ANOMALY_OPEN (most ever-anomalous loans have already
-- resolved to CLOSED by now and are NOT excluded here).
SELECT disbursement_fid, phonenumber, disbursement_ts, disbursed_amount
FROM disb_dedup
WHERE disbursement_fid NOT IN (SELECT disbursement_fid FROM loan_state_current_anomalies)
),
surviving_windows AS (
SELECT w.disbursement_fid, w.phonenumber, w.disbursement_ts, w.next_disbursement_ts, s.disbursed_amount
FROM disb_windows w
JOIN surviving_loans s ON s.disbursement_fid = w.disbursement_fid
),
per_loan_attributed AS (
SELECT
w.disbursement_fid,
w.disbursed_amount,
COALESCE(SUM(ABS(r.repayment_amount)), 0) AS attributed_repaid_abs
FROM surviving_windows w
LEFT JOIN repay_dedup r
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
GROUP BY w.disbursement_fid, w.disbursed_amount
),
joined AS (
SELECT
pla.disbursement_fid,
pla.attributed_repaid_abs,
lsl.lifetime_repaid_ugx,
COALESCE(ea.is_exposed, 0) AS is_exposed
FROM per_loan_attributed pla
JOIN loan_state_latest lsl ON lsl.disbursement_fid = pla.disbursement_fid
LEFT JOIN loan_state_ever_anomaly ea ON ea.disbursement_fid = pla.disbursement_fid
)
SELECT
is_exposed,
COUNT(*) AS n_matched,
COUNT_IF(ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1) AS n_exact_match,
ROUND(100.0 * COUNT_IF(ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1)
/ NULLIF(COUNT(*), 0), 2) AS pct_exact_match
FROM joined
GROUP BY is_exposed
ORDER BY is_exposed;
-- RESULT:
-- INTERPRETATION: two rows -- is_exposed=0 (this loan was never
-- ANOMALY_OPEN at any point) and is_exposed=1 (it was, at some point,
-- even though it has since resolved and is counted as "surviving" today).
-- If pct_exact_match is meaningfully lower for is_exposed=1, that is
-- direct, quantified evidence the attribution-window mismatch mechanism
-- (repayments landing after a second loan's disbursement getting counted
-- toward the wrong loan) is a real, material driver of B2b's gap --
-- distinct from and additive to the already-confirmed repayment_uid
-- cross-customer effect (52.7% vs 69.9% exact match). Also check n_matched
-- for is_exposed=1 against the ~90,730 total ANOMALY_OPEN count from
-- loan_concurrency_by_month.sql -- most should show up here as "surviving"
-- (since they resolve to CLOSED), confirming this exposure is NOT already
-- being filtered out by the production reconciliation's current-status-
-- only anomaly exclusion.
