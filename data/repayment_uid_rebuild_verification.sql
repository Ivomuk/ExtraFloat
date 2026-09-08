-- ============================================================================
-- repayment_uid_rebuild_verification.sql -- ad-hoc diagnostic, not part of
-- the committed validation suite. analytics.momo_loan_book_tracker_
-- repayments_daily has been rebuilt; the warehouse now has data through
-- 2026-06-09, which for the first time includes April 2026 -- the actual
-- incident period this whole investigation started from. This re-checks
-- the two headline repayment_uid problems found against the OLD table --
-- (1) raw duplicate-posting rate (repayment_uid_dedup_test.sql: 40.6% of
-- all rows, driven by a small number of huge multi-customer clusters) and
-- (2) cross-customer repayment_uid sharing (repayment_uid_cross_customer_
-- prevalence.sql: 43.8% of rows, one uid shared by up to 68,489 distinct
-- customers) -- against the REBUILT table, scoped to date_key <= 20260609
-- to match what's actually been reloaded so far. Queries the raw table
-- directly (mirrors vw_bh_repay_dedup's own fid-level dedup logic inline)
-- rather than depending on the validation schema's GATE 0 views, so this
-- can be run immediately without first rebuilding those views against the
-- new load.
-- ============================================================================

WITH repay_fid_deduped AS (
SELECT
repayment_fid,
repayment_uid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(repayment_ts AS timestamp) AS repayment_ts,
cast(repayment_amount_ugx AS double) AS repayment_amount
FROM (
SELECT r.*,
ROW_NUMBER() OVER (
PARTITION BY repayment_fid
ORDER BY inserted_ts DESC
) rn
FROM analytics.momo_loan_book_tracker_repayments_daily r
WHERE ova = 'XTRAFLOAT-AGENT'
AND repayment_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND try_cast(repayment_ts AS timestamp) IS NOT NULL
AND date_key <= 20260609
)
WHERE rn = 1
),

-- Step 1: raw repayment_uid duplication rate (exact key, same approach as
-- repayment_uid_dedup_test.sql). Uses approx_distinct() for the same
-- performance reason as before.
dup_check AS (
SELECT
COUNT(*) AS total_repayment_rows,
approx_distinct(COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR))) AS distinct_repayment_uids_approx,
COUNT_IF(repayment_uid IS NULL) AS n_null_repayment_uid
FROM repay_fid_deduped
),

-- Step 2 setup: cross-customer repayment_uid sharing (same approach as
-- repayment_uid_cross_customer_prevalence.sql).
uid_customer_stats AS (
SELECT
COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR)) AS uid_key,
COUNT(DISTINCT phonenumber) AS distinct_customers,
COUNT(*) AS rows_in_cluster
FROM repay_fid_deduped
GROUP BY COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR))
)

SELECT
d.total_repayment_rows,
d.distinct_repayment_uids_approx,
d.total_repayment_rows - d.distinct_repayment_uids_approx AS duplicate_rows_by_uid_approx,
ROUND(100.0 * (d.total_repayment_rows - d.distinct_repayment_uids_approx) / NULLIF(d.total_repayment_rows, 0), 2) AS pct_rows_are_uid_duplicates_approx,
d.n_null_repayment_uid,
(SELECT COUNT(*) FROM uid_customer_stats) AS total_uid_clusters,
(SELECT COUNT(*) FROM uid_customer_stats WHERE distinct_customers > 1) AS cross_customer_uid_clusters,
(SELECT SUM(rows_in_cluster) FROM uid_customer_stats WHERE distinct_customers > 1) AS rows_in_cross_customer_clusters,
ROUND(100.0 * (SELECT COALESCE(SUM(rows_in_cluster), 0) FROM uid_customer_stats WHERE distinct_customers > 1)
/ NULLIF(d.total_repayment_rows, 0), 2) AS pct_rows_in_cross_customer_clusters,
(SELECT MAX(distinct_customers) FROM uid_customer_stats) AS max_customers_sharing_one_uid
FROM dup_check d;
-- RESULT:
-- INTERPRETATION: compare every column directly against the pre-rebuild
-- numbers:
--   pct_rows_are_uid_duplicates_approx   was 40.6%
--   pct_rows_in_cross_customer_clusters  was 43.8%
--   max_customers_sharing_one_uid        was 68,489
-- If these have dropped to near-zero (single-digit percent, max customers
-- in the single/low-double digits), the rebuild fixed the batch-tagging
-- issue at the source and the repayment_uid field is now trustworthy.
-- If they're still elevated, the rebuild didn't address this specific
-- problem (or reloaded the same underlying batch-tagging behavior), and
-- B2b's reconciliation approach should stay as documented (materiality-
-- based gate, no repayment_uid-based dedup) rather than assuming a fix.
-- This now covers Jan-Jun 2026 (date_key <= 20260609), which INCLUDES
-- April 2026 -- the actual incident period this investigation started
-- from. Unlike the earlier Jan-Mar-only run, a clean result here is
-- direct evidence the fix covers the incident window itself, not just an
-- adjacent period. If elevated, isolate April specifically (date_key
-- BETWEEN 20260401 AND 20260430) to see whether it's disproportionately
-- affected relative to the rest of the window.
