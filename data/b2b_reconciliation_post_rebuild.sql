-- ============================================================================
-- b2b_reconciliation_post_rebuild.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. repayment_uid_rebuild_verification.sql
-- confirmed the rebuilt momo_loan_book_tracker_repayments_daily (reloaded
-- through March 2026) no longer shows the mass batch-tagging pattern
-- (duplicate rate 40.6% -> 4.1%, cross-customer sharing 43.8% -> 2.9%, max
-- customers per repayment_uid 68,489 -> 4). That confirms the underlying
-- DATA got cleaner, but not yet whether it changed the actual B2b
-- reconciliation OUTCOME. This re-runs the real exact-match/materiality
-- test against the rebuilt table for Jan-Mar 2026 (the same window whose
-- pre-rebuild baseline was 71.8% pooled exact match, 91.1%/95.1% within
-- 1%/5% of principal, per b2b_diff_distribution_by_period.sql) so the two
-- can be compared directly. Self-contained against the raw tables (mirrors
-- vw_bh_disb_dedup/disb_windows/repay_dedup/loan_state_snapshot/
-- surviving_loans inline) so it can run immediately without first
-- rebuilding the GATE 0 views against the new load. Bounded to
-- date_key <= 20260331 to match what's actually been reloaded.
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
AND date_key <= 20260331
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
AND date_key <= 20260331
) r
)
WHERE rn = 1
),
loan_state_snapshot AS (
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
AND date_key <= 20260331
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
surviving_loans AS (
SELECT disbursement_fid, phonenumber, disbursement_ts, disbursed_amount
FROM disb_dedup
WHERE disbursement_fid NOT IN (SELECT disbursement_fid FROM loan_state_anomalies)
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
pla.disbursed_amount,
pla.attributed_repaid_abs,
lsl.lifetime_repaid_ugx,
(pla.attributed_repaid_abs - lsl.lifetime_repaid_ugx) AS diff_ugx
FROM per_loan_attributed pla
JOIN loan_state_snapshot lsl ON lsl.disbursement_fid = pla.disbursement_fid
)
SELECT
COUNT(*) AS n_matched,
COUNT_IF(ABS(diff_ugx) <= 1) AS n_exact_match,
ROUND(100.0 * COUNT_IF(ABS(diff_ugx) <= 1) / NULLIF(COUNT(*), 0), 2) AS pct_exact_match,
COUNT_IF(ABS(diff_ugx) > 1 AND ABS(diff_ugx) <= 0.01 * disbursed_amount) AS n_within_1pct_principal,
COUNT_IF(ABS(diff_ugx) > 1 AND ABS(diff_ugx) <= 0.05 * disbursed_amount) AS n_within_5pct_principal,
ROUND(100.0 * (COUNT_IF(ABS(diff_ugx) <= 1) + COUNT_IF(ABS(diff_ugx) > 1 AND ABS(diff_ugx) <= 0.01 * disbursed_amount))
/ NULLIF(COUNT(*), 0), 2) AS pct_within_1pct_principal_cumulative,
ROUND(100.0 * (COUNT_IF(ABS(diff_ugx) <= 1) + COUNT_IF(ABS(diff_ugx) > 1 AND ABS(diff_ugx) <= 0.05 * disbursed_amount))
/ NULLIF(COUNT(*), 0), 2) AS pct_within_5pct_principal_cumulative,
SUM(ABS(diff_ugx)) AS total_abs_diff_ugx,
SUM(lifetime_repaid_ugx) AS total_lifetime_repaid_ugx,
ROUND(100.0 * SUM(ABS(diff_ugx)) / NULLIF(SUM(lifetime_repaid_ugx), 0), 4) AS pct_volume_error
FROM joined;
-- RESULT:
-- INTERPRETATION: compare directly against the PRE-REBUILD Jan-Mar
-- baseline (b2b_diff_distribution_by_period.sql, same population):
--   pct_exact_match                       was 71.8% pooled
--   pct_within_1pct_principal_cumulative  was 91.1%
--   pct_within_5pct_principal_cumulative  was 95.1%
--   pct_volume_error                      was 4.8174%
-- If these have improved, that's direct proof the rebuild helped the
-- actual reconciliation outcome, not just the repayment_uid quality
-- metrics. If they're flat/similar, the batch-tagging issue wasn't a
-- meaningful driver of Jan-Mar's already-small gap (consistent with
-- Jan-Mar's gap being mostly small/immaterial noise, as already
-- documented) even though the underlying data got measurably cleaner.
