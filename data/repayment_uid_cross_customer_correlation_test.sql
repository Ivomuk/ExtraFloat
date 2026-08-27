-- ============================================================================
-- repayment_uid_cross_customer_correlation_test.sql -- ad-hoc diagnostic,
-- not part of the committed validation suite.
-- repayment_uid_cross_customer_prevalence.sql's Step 2 found 25.2% of
-- surviving loans (1,111,056 of 4,403,072) have at least one repayment in
-- their attribution window traceable to a repayment_uid shared with a
-- DIFFERENT customer -- a number on the same order as B2b's overall ~31%
-- mismatch rate. But that's an EXPOSURE number, not a HARM number: a loan
-- can have a cross-customer-tagged repayment in its window and still
-- reconcile perfectly, if that repayment genuinely belongs to it and
-- merely shares a batch tag with many others. This settles whether
-- exposure actually correlates with mismatch: cross-tabulates "exposed to
-- a cross-customer uid" against "exact match" for the same matched
-- population B2b already scores.
-- ============================================================================

WITH uid_customer_stats AS (
SELECT
COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR)) AS uid_key,
COUNT(DISTINCT phonenumber) AS distinct_customers
FROM :validation_schema.vw_bh_repay_dedup
GROUP BY COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR))
),
cross_customer_repayments AS (
SELECT r.phonenumber, r.repayment_amount, r.repayment_ts
FROM :validation_schema.vw_bh_repay_dedup r
JOIN uid_customer_stats u
ON u.uid_key = COALESCE(CAST(r.repayment_uid AS VARCHAR), CAST(r.repayment_fid AS VARCHAR))
WHERE u.distinct_customers > 1
),
surviving_windows AS (
SELECT
w.disbursement_fid,
w.phonenumber,
w.disbursement_ts,
w.next_disbursement_ts,
s.disbursed_amount
FROM :validation_schema.vw_bh_disb_windows w
JOIN :validation_schema.vw_bh_surviving_loans s
ON s.disbursement_fid = w.disbursement_fid
),
per_loan_attributed AS (
SELECT
w.disbursement_fid,
w.disbursed_amount,
COALESCE(SUM(ABS(a.repayment_amount)), 0) AS attributed_repaid_abs,
MAX(CASE WHEN r.phonenumber IS NOT NULL THEN 1 ELSE 0 END) AS is_exposed
FROM surviving_windows w
LEFT JOIN :validation_schema.vw_bh_repay_dedup a
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
pla.is_exposed,
lsl.lifetime_repaid_ugx
FROM per_loan_attributed pla
JOIN :validation_schema.vw_bh_loan_state_snapshot lsl
ON lsl.disbursement_fid = pla.disbursement_fid
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
-- INTERPRETATION: two rows -- is_exposed=0 (no cross-customer-tagged
-- repayment in this loan's window) and is_exposed=1 (at least one). If
-- pct_exact_match is dramatically lower for is_exposed=1 than is_exposed=0
-- (e.g. 40% vs 80%), cross-customer uid exposure is a real, substantial
-- driver of B2b's gap -- bigger than any of the four previously-confirmed
-- mechanisms -- and worth escalating to the table owner as the primary
-- finding, not a side note. If the two rates are similar, the 25.2%
-- exposure figure is mostly incidental overlap with otherwise-fine loans,
-- and the existing conclusion (materiality-based gate, no single dominant
-- cause) stands as documented.
