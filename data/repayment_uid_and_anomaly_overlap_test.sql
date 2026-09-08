-- ============================================================================
-- repayment_uid_and_anomaly_overlap_test.sql -- ad-hoc diagnostic, not part
-- of the committed validation suite. Two separate exposure mechanisms have
-- now been independently confirmed as real, non-trivial drivers of B2b
-- mismatch:
--   repayment_uid cross-customer exposure  -- 6.3% of surviving loans,
--                                              69.9% vs 52.7% exact match
--                                              (17.2-point gap)
--   ANOMALY_OPEN (ever, not just current)  -- 17% of surviving loans,
--                                              70.3% vs 61.6% exact match
--                                              (8.7-point gap)
-- Before treating their contributions as simply additive, this checks
-- whether the two exposed populations actually overlap much -- a loan
-- caught by both mechanisms at once would otherwise have its mismatch
-- double-counted when reasoning about combined impact. Cross-tabulates
-- BOTH exposure flags together (2x2) against exact-match rate for the
-- same surviving-loan population the other two tests scored.
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
SELECT repayment_fid, repayment_uid, phonenumber, repayment_ts, repayment_amount
FROM (
SELECT r.*,
ROW_NUMBER() OVER (PARTITION BY repayment_fid ORDER BY inserted_ts DESC) rn
FROM (
SELECT
repayment_fid,
repayment_uid,
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
uid_customer_stats AS (
SELECT
COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR)) AS uid_key,
COUNT(DISTINCT phonenumber) AS distinct_customers
FROM repay_dedup
GROUP BY COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR))
),
cross_customer_repayments AS (
SELECT r.phonenumber, r.repayment_amount, r.repayment_ts
FROM repay_dedup r
JOIN uid_customer_stats u
ON u.uid_key = COALESCE(CAST(r.repayment_uid AS VARCHAR), CAST(r.repayment_fid AS VARCHAR))
WHERE u.distinct_customers > 1
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
MAX(CASE WHEN loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true THEN 1 ELSE 0 END) AS is_exposed_anomaly
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (PARTITION BY disbursement_fid, date_key ORDER BY inserted_ts DESC) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE ova = 'XTRAFLOAT-AGENT'
AND disbursement_fid IS NOT NULL
AND date_key <= 20260609
) lsd
WHERE rn = 1
GROUP BY disbursement_fid
),
loan_state_current_anomalies AS (
SELECT disbursement_fid FROM loan_state_latest
WHERE loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true
),
surviving_loans AS (
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
COALESCE(SUM(ABS(a.repayment_amount)), 0) AS attributed_repaid_abs,
MAX(CASE WHEN r.phonenumber IS NOT NULL THEN 1 ELSE 0 END) AS is_exposed_uid
FROM surviving_windows w
LEFT JOIN repay_dedup a
ON a.phonenumber = w.phonenumber
AND a.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR a.repayment_ts < w.next_disbursement_ts)
LEFT JOIN cross_customer_repayments r
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
GROUP BY w.disbursement_fid, w.disbursed_amount
),
joined AS (
SELECT
pla.disbursement_fid,
pla.attributed_repaid_abs,
pla.is_exposed_uid,
COALESCE(ea.is_exposed_anomaly, 0) AS is_exposed_anomaly,
lsl.lifetime_repaid_ugx
FROM per_loan_attributed pla
JOIN loan_state_latest lsl ON lsl.disbursement_fid = pla.disbursement_fid
LEFT JOIN loan_state_ever_anomaly ea ON ea.disbursement_fid = pla.disbursement_fid
)
SELECT
is_exposed_uid,
is_exposed_anomaly,
COUNT(*) AS n_matched,
COUNT_IF(ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1) AS n_exact_match,
ROUND(100.0 * COUNT_IF(ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1)
/ NULLIF(COUNT(*), 0), 2) AS pct_exact_match
FROM joined
GROUP BY is_exposed_uid, is_exposed_anomaly
ORDER BY is_exposed_uid, is_exposed_anomaly;
-- RESULT:
-- INTERPRETATION: four rows -- (0,0) exposed to neither, (1,0) uid-only,
-- (0,1) anomaly-only, (1,1) both. Compare n_matched for (1,1) against the
-- naive product-of-independent-rates expectation (781,798/4,601,171 x
-- 781,798/4,601,171 x 4,601,171, i.e. what you'd expect if the two
-- exposures were statistically independent) -- if (1,1)'s actual count is
-- much larger than that, the two mechanisms co-occur more than chance
-- (plausibly the same root cause: agents with erratic repayment behavior
-- are prone to both), and their combined contribution to the overall gap
-- is LESS than simply adding the two individually-measured effects (since
-- some of the same loans are being counted in both). If pct_exact_match
-- for (1,1) is not meaningfully worse than the worse of (1,0)/(0,1) alone,
-- that also argues against treating the two effects as additive. Use
-- whichever of the four cells' pct_exact_match values are lowest to decide
-- which mechanism (or their combination) deserves the primary mention when
-- summarizing B2b's gap.
