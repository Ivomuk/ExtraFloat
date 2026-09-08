-- ============================================================================
-- b2b_reconciliation_april.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. b2b_reconciliation_post_rebuild.sql already
-- confirmed the rebuild didn't change the Jan-Mar reconciliation outcome
-- much -- but Jan-Mar was never the incident period. This isolates loans
-- actually DISBURSED IN APRIL 2026 specifically (the month the original
-- B2b mismatch was reported in) and reconciles them against the REBUILT
-- repayments table, using the full repayment/loan-state history available
-- through the current load (date_key <= 20260609) rather than bounding
-- everything to April -- an April loan's repayments can legitimately land
-- in May/June, and truncating those would manufacture false mismatches
-- that have nothing to do with the batch-tagging issue.
--
-- Population: surviving_loans is filtered to disbursement_ts falling
-- within April 2026 specifically (not "disbursed on or before some
-- cutoff", unlike b2b_reconciliation_post_rebuild.sql) -- this is a
-- disbursement-month cohort, matching b2b_monthly_breakdown.sql's own
-- convention for isolating a single month's severity. disb_dedup/
-- disb_windows themselves still cover the FULL loaded population (not
-- April-only), since a given phone number's window boundaries (LEAD
-- disbursement_ts) depend on its complete disbursement timeline, not just
-- the April subset.
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
surviving_loans AS (
SELECT disbursement_fid, phonenumber, disbursement_ts, disbursed_amount
FROM disb_dedup
WHERE disbursement_fid NOT IN (SELECT disbursement_fid FROM loan_state_anomalies)
AND date(disbursement_ts) >= DATE '2026-04-01'
AND date(disbursement_ts) <= DATE '2026-04-30'
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
-- INTERPRETATION: this is the actual incident-period test. Compare
-- against two references:
--   (a) The ORIGINAL pre-rebuild figures that motivated this entire
--       investigation (the ~68.6% exact match figure discussed for the
--       window logic generally, and the raw 256774 cross-customer
--       repayment_uid example shared with the table owner).
--   (b) b2b_reconciliation_post_rebuild.sql's Jan-Mar post-rebuild result
--       (72.5% exact match, 91.1%/94.7% within 1%/5% of principal) --
--       if April's numbers land close to that, the rebuild fixed April
--       specifically, not just an unrelated adjacent period. If April is
--       still meaningfully worse than Jan-Mar, that's evidence the
--       incident had a real, April-specific cause beyond the batch-
--       tagging issue the rebuild addressed.
-- Also worth cross-checking against repayment_uid_rebuild_verification.sql
-- run over the same window -- if THAT shows April's repayment_uid data is
-- now clean but this reconciliation still shows a gap, the batch-tagging
-- fix and the reconciliation outcome have decoupled, meaning something
-- else drove the original April incident.
