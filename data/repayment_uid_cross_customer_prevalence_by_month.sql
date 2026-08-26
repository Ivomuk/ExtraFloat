-- ============================================================================
-- repayment_uid_cross_customer_prevalence_by_month.sql -- ad-hoc
-- diagnostic, not part of the committed validation suite.
-- repayment_uid_cross_customer_prevalence.sql quantifies the 256772/256774
-- cross-customer repayment_uid pattern as a single blended number. Given
-- b2b_monthly_breakdown.sql already found B2b's overall gap is
-- concentrated in a dateable April 2026 incident (with a May recovery
-- tail), this checks whether the cross-customer uid pattern specifically
-- is ALSO concentrated there (reinforcing it as part of the same bounded
-- incident) or spread evenly across months (a persistent background
-- behavior unrelated to April).
-- ============================================================================

-- Step 1: prevalence of cross-customer repayment_uid rows, by the month
-- each repayment itself landed in.
WITH uid_customer_stats AS (
SELECT
COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR)) AS uid_key,
COUNT(DISTINCT phonenumber) AS distinct_customers
FROM :validation_schema.vw_bh_repay_dedup
GROUP BY COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR))
),
flagged AS (
SELECT
r.repayment_ts,
r.repayment_amount,
CASE WHEN u.distinct_customers > 1 THEN 1 ELSE 0 END AS is_cross_customer
FROM :validation_schema.vw_bh_repay_dedup r
JOIN uid_customer_stats u
ON u.uid_key = COALESCE(CAST(r.repayment_uid AS VARCHAR), CAST(r.repayment_fid AS VARCHAR))
)
SELECT
date_trunc('month', repayment_ts) AS repay_month,
COUNT(*) AS total_repayment_rows,
SUM(is_cross_customer) AS cross_customer_rows,
ROUND(100.0 * SUM(is_cross_customer) / NULLIF(COUNT(*), 0), 4) AS pct_cross_customer_rows,
SUM(CASE WHEN is_cross_customer = 1 THEN repayment_amount ELSE 0 END) AS cross_customer_amount_ugx
FROM flagged
GROUP BY date_trunc('month', repayment_ts)
ORDER BY repay_month;
-- RESULT:
-- INTERPRETATION: if pct_cross_customer_rows spikes in April 2026 (and
-- possibly into early May) the way B2b's overall exact-match rate did,
-- that's more evidence this specific mechanism is part of the same
-- bounded incident. If it's roughly flat every month, cross-customer uid
-- sharing is an ongoing background behavior in the source system,
-- independent of whatever else happened in April.

-- Step 2: severity by each surviving loan's OWN disbursement month --
-- what share of loans disbursed in each month have a repayment in their
-- attribution window traceable to a cross-customer uid cluster.
WITH uid_customer_stats AS (
SELECT
COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR)) AS uid_key,
COUNT(DISTINCT phonenumber) AS distinct_customers
FROM :validation_schema.vw_bh_repay_dedup
GROUP BY COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR))
),
cross_customer_repayments AS (
SELECT r.phonenumber, r.repayment_ts
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
w.next_disbursement_ts
FROM :validation_schema.vw_bh_disb_windows w
JOIN :validation_schema.vw_bh_surviving_loans s
ON s.disbursement_fid = w.disbursement_fid
),
loan_flagged AS (
SELECT
w.disbursement_fid,
w.disbursement_ts,
MAX(CASE WHEN r.phonenumber IS NOT NULL THEN 1 ELSE 0 END) AS is_affected
FROM surviving_windows w
LEFT JOIN cross_customer_repayments r
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
GROUP BY w.disbursement_fid, w.disbursement_ts
)
SELECT
date_trunc('month', disbursement_ts) AS disb_month,
COUNT(*) AS total_surviving_loans,
SUM(is_affected) AS n_loans_affected,
ROUND(100.0 * SUM(is_affected) / NULLIF(COUNT(*), 0), 4) AS pct_loans_affected
FROM loan_flagged
GROUP BY date_trunc('month', disbursement_ts)
ORDER BY disb_month;
-- RESULT:
-- INTERPRETATION: same comparison as Step 1, but on the metric that
-- actually matters for B2b -- share of loans exposed, not share of raw
-- repayment rows. Compare this month-by-month shape against
-- b2b_monthly_breakdown.sql's pct_exact_match/pct_volume_error curve: if
-- they move together (both worst in April), cross-customer uid sharing is
-- a meaningful contributor to the April incident specifically, not just a
-- coincidentally-timed curiosity.
