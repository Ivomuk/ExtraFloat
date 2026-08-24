-- ============================================================================
-- repayment_uid_dedup_test.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. repayment_full_schema_check.sql found the
-- real mechanism behind 256772's 10 "duplicate" repayment rows: they all
-- share the SAME repayment_uid (1609381785765) but 10 DIFFERENT
-- repayment_fid values, with every other column identical -- the same
-- fid-vs-uid split already found for loans (loan_uid vs disbursement_fid).
-- vw_bh_repay_dedup has only ever deduped by repayment_fid, so these
-- true duplicates pass straight through. duplicate_repayment_burst_test.sql's
-- fuzzy "same phonenumber+amount within 5 seconds" heuristic only caught
-- 0.5% of rows because it's an approximation of this; deduping by the exact
-- repayment_uid key should catch every duplicate regardless of how far
-- apart the retries are posted (hours, even days), not just sub-minute
-- bursts.
--
-- This (1) quantifies how many repayment_fid rows collapse under the real
-- repayment_uid key across the whole population, and (2) re-tests B2b's
-- reconciliation rate using a repayment_uid-deduped repayment view instead
-- of the fuzzy burst heuristic.
-- ============================================================================

-- Step 1: prevalence of repayment_uid duplication (exact key, not the fuzzy
-- amount/time-window proxy). Uses approx_distinct() (HyperLogLog, ~2.3%
-- standard error by default) instead of COUNT(DISTINCT ...) -- an exact
-- distinct count over ~6.4M rows is expensive, and this is a diagnostic
-- signal (is duplication meaningfully bigger than the fuzzy heuristic's
-- 0.5%?), not a value that needs to be exact to the row. Computed once in
-- the agg CTE and reused, instead of repeating the DISTINCT aggregation
-- three times in one SELECT. No passthrough CTE for vw_bh_repay_dedup --
-- querying it directly in agg since there's no filtering/transformation to
-- justify the extra layer. COUNT_IF and NULLIF(total_repayment_rows, 0)
-- guard the percentage against a divide-by-zero if the view ever returns
-- no rows, matching the NULLIF guards used elsewhere in this file's B2b
-- diagnostics.
WITH agg AS (
SELECT
COUNT(*) AS total_repayment_rows,
approx_distinct(COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR))) AS distinct_repayment_uids_approx,
COUNT_IF(repayment_uid IS NULL) AS n_null_repayment_uid
FROM :validation_schema.vw_bh_repay_dedup
)
SELECT
total_repayment_rows,
distinct_repayment_uids_approx,
total_repayment_rows - distinct_repayment_uids_approx AS duplicate_rows_by_uid_approx,
ROUND(100.0 * (total_repayment_rows - distinct_repayment_uids_approx) / NULLIF(total_repayment_rows, 0), 2) AS pct_rows_are_uid_duplicates_approx,
n_null_repayment_uid
FROM agg;
-- RESULT:
-- INTERPRETATION: compare pct_rows_are_uid_duplicates_approx against
-- duplicate_repayment_burst_test.sql's 0.5% -- if this is meaningfully
-- larger, the exact repayment_uid key is catching real duplicates the fuzzy
-- time-window heuristic missed (e.g. retries more than 5 seconds apart).
-- The ~2.3% standard error on the approx_distinct count is negligible next
-- to a difference of that size; if the two numbers come back close instead,
-- rerun with exact COUNT(DISTINCT) to confirm before concluding they're
-- the same. n_null_repayment_uid tells us how often the fallback to
-- repayment_fid (treating a NULL-uid row as its own group) is actually
-- exercised.

-- Step 2: reconciliation rate over the full surviving population using a
-- repayment_uid-deduped repayment set (one row per repayment_uid, earliest
-- repayment_ts / lowest repayment_fid kept), same exact-match methodology
-- as every prior B2b test.
WITH uid_deduped AS (
SELECT phonenumber, repayment_amount, repayment_ts
FROM (
SELECT r.*,
ROW_NUMBER() OVER (
PARTITION BY COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR))
ORDER BY repayment_ts, repayment_fid
) rn
FROM :validation_schema.vw_bh_repay_dedup r
)
WHERE rn = 1
),
attributed AS (
SELECT w.disbursement_fid, r.repayment_amount
FROM uid_deduped r
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
/ NULLIF(SUM(CASE WHEN state_only = 0 AND disbursement_only = 0 THEN 1 ELSE 0 END), 0), 2) AS pct_exact_match_after_uid_dedup
FROM joined;
-- RESULT:
-- INTERPRETATION: compare pct_exact_match_after_uid_dedup against the 68.6%
-- baseline (and the 68.7% from the fuzzy burst-window fix). If this is a
-- much bigger jump, repayment_uid-based dedup (not repayment_fid-based) is
-- the real, precise fix -- add a second ROW_NUMBER() PARTITION BY
-- COALESCE(repayment_uid, repayment_fid) pass to vw_bh_repay_dedup's
-- definition here and to borrower_history.txt's real repay_raw/repay_dedup
-- CTE, replacing duplicate_repayment_burst_test.sql's fuzzy heuristic
-- entirely rather than stacking both.
