-- ============================================================================
-- repayment_uid_cross_customer_prevalence.sql -- ad-hoc diagnostic, not
-- part of the committed validation suite. The 256774 worked example
-- confirmed repayment_uid 1609381785765 is shared across TWO unrelated
-- customers (256772 and 256774), spanning three days and 18 rows --
-- direct evidence repayment_uid is a batch/settlement-run identifier, not
-- a per-transaction or per-customer key. This quantifies how widespread
-- that specific pattern is: (1) how many repayment_uid values span
-- multiple distinct phone numbers system-wide, and (2) how many of our
-- SURVIVING loans actually have a repayment landing in their attribution
-- window that came from one of those cross-customer clusters -- the
-- direct measure of how many loans are actually at risk from this
-- mechanism, not just how unusual the raw table looks in isolation.
-- ============================================================================

-- Step 1: prevalence of repayment_uid values shared across multiple
-- distinct customers (phonenumbers), system-wide.
WITH uid_customer_stats AS (
SELECT
COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR)) AS uid_key,
COUNT(DISTINCT phonenumber) AS distinct_customers,
COUNT(*) AS rows_in_cluster,
SUM(repayment_amount) AS total_amount_ugx
FROM :validation_schema.vw_bh_repay_dedup
GROUP BY COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR))
)
SELECT
COUNT(*) AS total_uid_clusters,
SUM(CASE WHEN distinct_customers > 1 THEN 1 ELSE 0 END) AS cross_customer_uid_clusters,
SUM(CASE WHEN distinct_customers > 1 THEN rows_in_cluster ELSE 0 END) AS rows_in_cross_customer_clusters,
ROUND(100.0 * SUM(CASE WHEN distinct_customers > 1 THEN rows_in_cluster ELSE 0 END)
/ (SELECT COUNT(*) FROM :validation_schema.vw_bh_repay_dedup), 2) AS pct_rows_in_cross_customer_clusters,
SUM(CASE WHEN distinct_customers > 1 THEN total_amount_ugx ELSE 0 END) AS total_amount_in_cross_customer_clusters_ugx,
MAX(distinct_customers) AS max_customers_sharing_one_uid
FROM uid_customer_stats;
-- RESULT:
-- INTERPRETATION: cross_customer_uid_clusters / total_uid_clusters and
-- pct_rows_in_cross_customer_clusters show how common the 256772/256774
-- pattern actually is, not just that it happened once.
-- max_customers_sharing_one_uid shows the largest observed "batch" --
-- if this is much bigger than 2, some settlement runs touch many
-- customers at once.

-- Step 2: how many SURVIVING loans have a repayment landing in their
-- attribution window that came from a cross-customer uid cluster -- the
-- direct measure of how many loans are actually exposed to this
-- mechanism's contamination, versus loans whose only repayments came from
-- clean, single-customer uid clusters.
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
affected_loans AS (
SELECT DISTINCT w.disbursement_fid
FROM surviving_windows w
JOIN cross_customer_repayments r
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
)
SELECT
(SELECT COUNT(*) FROM :validation_schema.vw_bh_surviving_loans) AS total_surviving_loans,
COUNT(*) AS n_loans_with_cross_customer_repayment_in_window,
ROUND(100.0 * COUNT(*) / NULLIF((SELECT COUNT(*) FROM :validation_schema.vw_bh_surviving_loans), 0), 4) AS pct_surviving_loans_affected
FROM affected_loans;
-- RESULT:
-- INTERPRETATION: pct_surviving_loans_affected is the real-world severity
-- number -- what share of loans in the actual production population have
-- at least one repayment in their window traceable to a repayment_uid
-- shared with a DIFFERENT customer. If this is small (a fraction of a
-- percent), the cross-customer batching is real and worth raising to the
-- table owner, but not a significant driver of B2b's overall gap on its
-- own -- consistent with how the two customer examples traced by hand
-- turned out to be dramatic-looking but numerically rare. If it's large,
-- it deserves more weight in the B2b conclusion than currently documented.
