-- ============================================================================
-- duplicate_repayment_burst_test.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. clean_loan_raw_trace.sql found the actual
-- root cause of B2b's persistent reconciliation gap: the source system
-- posts the SAME real repayment multiple times under DIFFERENT
-- repayment_fid values, for the same phonenumber and amount, seconds apart.
-- Confirmed examples: loan 39604855038 / 256772 has 10 rows of exactly
-- 750,000 UGX all within 2026-04-01 17:00:22-17:00:36 (14 seconds); loan
-- 39634283120 / 256773 has 7 rows of exactly 750,000 UGX all within
-- 2026-04-03 14:57:52-53 (1-2 seconds). Both times, attributed_repaid_abs
-- exactly equals (duplicate row count) x (repayment amount) -- our window
-- heuristic sums every duplicate as independent, while the tracker's own
-- lifetime_repaid_ugx/gross_repaid_ugx fields evidently collapse the burst
-- via their own idempotency logic. vw_bh_repay_dedup's existing dedup only
-- collapses rows sharing the SAME repayment_fid, so these pass through
-- untouched.
--
-- This (1) quantifies how common these same-phonenumber/same-amount bursts
-- are across the whole repayment population, and (2) tests whether
-- collapsing each burst (rows within BURST_SECONDS of the previous row in
-- the same phonenumber+amount partition) to a single row closes the B2b
-- reconciliation gap (baseline 68.6% exact-match).
-- ============================================================================

-- Step 1: prevalence of same-phonenumber/same-amount bursts.
WITH gapped AS (
SELECT
phonenumber, repayment_amount, repayment_ts, repayment_fid,
date_diff('second', LAG(repayment_ts) OVER (
PARTITION BY phonenumber, repayment_amount ORDER BY repayment_ts, repayment_fid
), repayment_ts) AS gap_seconds
FROM :validation_schema.vw_bh_repay_dedup
),
grouped AS (
SELECT *,
SUM(CASE WHEN gap_seconds IS NULL OR gap_seconds > 5 THEN 1 ELSE 0 END) OVER (
PARTITION BY phonenumber, repayment_amount ORDER BY repayment_ts, repayment_fid
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
) AS burst_group
FROM gapped
),
burst_collapsed AS (
SELECT
phonenumber, repayment_amount, burst_group,
MIN(repayment_ts) AS repayment_ts,
MIN(repayment_fid) AS repayment_fid,
COUNT(*) AS rows_in_burst
FROM grouped
GROUP BY phonenumber, repayment_amount, burst_group
)
SELECT
(SELECT COUNT(*) FROM :validation_schema.vw_bh_repay_dedup) AS total_repayment_rows,
COUNT(*) AS total_after_burst_collapse,
SUM(CASE WHEN rows_in_burst > 1 THEN 1 ELSE 0 END) AS burst_groups_with_duplicates,
SUM(CASE WHEN rows_in_burst > 1 THEN rows_in_burst - 1 ELSE 0 END) AS duplicate_rows_removed,
ROUND(100.0 * SUM(CASE WHEN rows_in_burst > 1 THEN rows_in_burst - 1 ELSE 0 END)
/ (SELECT COUNT(*) FROM :validation_schema.vw_bh_repay_dedup), 2) AS pct_rows_are_duplicates,
SUM(CASE WHEN rows_in_burst > 1 THEN (rows_in_burst - 1) * repayment_amount ELSE 0 END) AS duplicate_ugx_removed
FROM burst_collapsed;
-- RESULT:
-- INTERPRETATION: pct_rows_are_duplicates tells us how big this problem is
-- at scale. If it's a meaningful share (not just our two hand-picked
-- examples), duplicate_ugx_removed shows the total inflation this causes.

-- Step 2: reconciliation rate over the FULL surviving population after
-- collapsing bursts, same exact-match methodology as every prior B2b test.
WITH gapped AS (
SELECT
phonenumber, repayment_amount, repayment_ts, repayment_fid,
date_diff('second', LAG(repayment_ts) OVER (
PARTITION BY phonenumber, repayment_amount ORDER BY repayment_ts, repayment_fid
), repayment_ts) AS gap_seconds
FROM :validation_schema.vw_bh_repay_dedup
),
grouped AS (
SELECT *,
SUM(CASE WHEN gap_seconds IS NULL OR gap_seconds > 5 THEN 1 ELSE 0 END) OVER (
PARTITION BY phonenumber, repayment_amount ORDER BY repayment_ts, repayment_fid
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
) AS burst_group
FROM gapped
),
burst_collapsed AS (
SELECT
phonenumber, repayment_amount,
MIN(repayment_ts) AS repayment_ts
FROM grouped
GROUP BY phonenumber, repayment_amount, burst_group
),
attributed AS (
SELECT w.disbursement_fid, r.repayment_amount
FROM burst_collapsed r
JOIN :validation_schema.vw_bh_disb_windows w
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
WHERE w.disbursement_fid IN (SELECT disbursement_fid FROM :validation_schema.vw_bh_surviving_loans)
),
per_loan_attributed AS (
SELECT
w.disbursement_fid,
w.disbursed_amount,
COALESCE(SUM(ABS(a.repayment_amount)), 0) AS attributed_repaid_abs
FROM :validation_schema.vw_bh_surviving_loans w
LEFT JOIN attributed a ON a.disbursement_fid = w.disbursement_fid
GROUP BY w.disbursement_fid, w.disbursed_amount
),
joined AS (
SELECT
pla.disbursement_fid,
pla.attributed_repaid_abs,
lsl.lifetime_repaid_ugx,
CASE WHEN pla.disbursement_fid IS NULL THEN 1 ELSE 0 END AS state_only,
CASE WHEN lsl.disbursement_fid IS NULL THEN 1 ELSE 0 END AS disbursement_only
FROM per_loan_attributed pla
FULL OUTER JOIN :validation_schema.vw_bh_loan_state_snapshot lsl
ON lsl.disbursement_fid = pla.disbursement_fid
)
SELECT
COUNT(*) AS n_total,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0 THEN 1 ELSE 0 END) AS n_matched,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_exact_match,
ROUND(100.0 * SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END)
/ NULLIF(SUM(CASE WHEN state_only = 0 AND disbursement_only = 0 THEN 1 ELSE 0 END), 0), 2) AS pct_exact_match_after_burst_collapse
FROM joined;
-- RESULT:
-- INTERPRETATION: compare pct_exact_match_after_burst_collapse against the
-- 68.6% baseline. If this closes most of the gap (unlike the ~2%
-- improvements from loan_uid-aware windowing, gross-vs-net, and
-- ever-anomalous exclusion), burst-duplicate collapsing is the dominant
-- fix -- and should be added to vw_bh_repay_dedup's dedup logic (and
-- borrower_history.txt's real repay_raw/repay_attributed CTEs) as an
-- additional GROUP BY (phonenumber, repayment_amount, burst_group) step
-- before attribution, not just repayment_fid-level dedup.
