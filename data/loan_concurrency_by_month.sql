-- ============================================================================
-- loan_concurrency_by_month.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. repayment_uid_cross_customer_correlation_
-- test.sql confirmed cross-customer repayment_uid exposure is a real
-- driver of mismatch (52.7% vs 69.9% exact match) but repayment_uid_
-- cross_customer_prevalence_by_month.sql showed April isn't where that
-- exposure concentrates (4.3%, mid-range across Jan-May) -- so it can't
-- explain why April-disbursed loans reconcile meaningfully worse than
-- Jan-Mar (59% vs 72.5% exact match, per b2b_reconciliation_april.sql).
-- This tests a different hypothesis: loan CONCURRENCY. The reconciliation
-- logic attributes repayments to a loan via a phonenumber-scoped window
-- bounded by that customer's NEXT disbursement (LEAD(disbursement_ts));
-- if a customer takes their next loan very soon after the one being
-- reconciled, that window is squeezed tight, and repayments genuinely
-- belonging to the earlier loan can spill past the window boundary (or
-- vice versa) -- a mechanism completely independent of repayment_uid
-- batch-tagging. Scoped to date_key <= 20260609 to match what's loaded.
--
-- Two measures, by disbursement month:
--   1. ANOMALY_OPEN rate -- the loan_state table's own ground-truth signal
--      for "agent took a new loan before settling the current one" (the
--      most extreme form of concurrency, already excluded from the
--      reconciliation population as a known confound).
--   2. Among SURVIVING (non-anomaly) loans, what fraction have their next
--      disbursement within 24h/48h -- a proxy for "the attribution window
--      was already tight even for loans that didn't trip the ANOMALY_OPEN
--      flag," which is exactly what would stress the reconciliation
--      heuristic without being caught by the anomaly exclusion at all.
-- ============================================================================

WITH disb_dedup AS (
SELECT disbursement_fid, phonenumber, disbursement_ts
FROM (
SELECT d.*,
ROW_NUMBER() OVER (PARTITION BY disbursement_fid ORDER BY inserted_ts DESC) rn
FROM (
SELECT
disbursement_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(disbursement_ts AS timestamp) AS disbursement_ts,
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
disbursement_fid, phonenumber, disbursement_ts,
LEAD(disbursement_ts) OVER (PARTITION BY phonenumber ORDER BY disbursement_ts, disbursement_fid) AS next_disbursement_ts
FROM disb_dedup
),
loan_state_snapshot AS (
SELECT disbursement_fid, loan_status, is_anomaly_open
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
loan_state_anomalies AS (
SELECT disbursement_fid FROM loan_state_snapshot
WHERE loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true
),
per_loan AS (
SELECT
w.disbursement_fid,
date_trunc('month', w.disbursement_ts) AS disb_month,
CASE WHEN a.disbursement_fid IS NOT NULL THEN 1 ELSE 0 END AS is_anomaly,
date_diff('hour', w.disbursement_ts, w.next_disbursement_ts) AS hours_to_next_disbursement
FROM disb_windows w
LEFT JOIN loan_state_anomalies a ON a.disbursement_fid = w.disbursement_fid
)
SELECT
disb_month,
COUNT(*) AS total_loans,
SUM(is_anomaly) AS n_anomaly_open,
ROUND(100.0 * SUM(is_anomaly) / NULLIF(COUNT(*), 0), 2) AS pct_anomaly_open,
COUNT_IF(is_anomaly = 0) AS n_surviving,
COUNT_IF(is_anomaly = 0 AND hours_to_next_disbursement IS NOT NULL AND hours_to_next_disbursement < 24) AS n_surviving_next_within_24h,
ROUND(100.0 * COUNT_IF(is_anomaly = 0 AND hours_to_next_disbursement IS NOT NULL AND hours_to_next_disbursement < 24)
/ NULLIF(COUNT_IF(is_anomaly = 0), 0), 2) AS pct_surviving_next_within_24h,
COUNT_IF(is_anomaly = 0 AND hours_to_next_disbursement IS NOT NULL AND hours_to_next_disbursement < 48) AS n_surviving_next_within_48h,
ROUND(100.0 * COUNT_IF(is_anomaly = 0 AND hours_to_next_disbursement IS NOT NULL AND hours_to_next_disbursement < 48)
/ NULLIF(COUNT_IF(is_anomaly = 0), 0), 2) AS pct_surviving_next_within_48h,
APPROX_PERCENTILE(CASE WHEN is_anomaly = 0 THEN hours_to_next_disbursement END, 0.5) AS median_hours_to_next_disbursement_surviving
FROM per_loan
GROUP BY disb_month
ORDER BY disb_month;
-- RESULT:
-- INTERPRETATION: compare April's row against the surrounding months on
-- BOTH measures. If pct_anomaly_open and/or pct_surviving_next_within_24h/
-- 48h are meaningfully HIGHER for April than Jan-Mar/May, that's evidence
-- loan concurrency (rapid reborrowing squeezing the attribution window) is
-- a real, April-concentrated driver of the reconciliation gap -- distinct
-- from and additive to the cross-customer repayment_uid exposure effect
-- already confirmed. If April looks similar to surrounding months here
-- too, concurrency isn't the answer either, and the April-specific cause
-- remains unidentified -- worth widening the search (e.g. a specific
-- large batch/system event on a particular day within April) rather than
-- another population-level demographic cut.
