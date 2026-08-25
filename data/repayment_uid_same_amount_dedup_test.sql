-- ============================================================================
-- repayment_uid_same_amount_dedup_test.sql -- ad-hoc diagnostic, not part
-- of the committed validation suite. repayment_uid_dedup_test.sql's
-- population-wide result REFUTED the hypothesis that repayment_uid alone is
-- a safe "collapse to one real payment" key: deduping by repayment_uid
-- collapsed 40.6% of all repayment rows, but B2b's exact-match rate
-- actually DROPPED (68.6% -> 67.4%), not improved. That means most
-- repayment_uid clusters must contain rows with DIFFERENT amounts --
-- legitimately separate real repayments sharing a repayment_uid for some
-- other reason (a batch/session/settlement grouping, not a
-- one-real-payment identifier) -- and collapsing those throws away real
-- repaid money.
--
-- The two examples that led to this hypothesis (256772's 10 rows, 256773's
-- 7 rows) had one thing in common beyond sharing a repayment_uid: every row
-- in each cluster ALSO shared the exact same repayment_amount. This tests
-- the narrower, more defensible hypothesis: only collapse rows that share
-- BOTH repayment_uid AND repayment_amount, leaving mixed-amount uid
-- clusters (which are apparently the majority, and are apparently real
-- distinct repayments) untouched.
-- ============================================================================

-- Step 1: how repayment_uid clusters break down -- uniform-amount (true
-- duplicate candidates, like our two traced examples) vs mixed-amount
-- (apparently real distinct repayments sharing a uid for another reason).
WITH keyed AS (
SELECT
COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR)) AS uid_key,
repayment_amount
FROM :validation_schema.vw_bh_repay_dedup
),
cluster_stats AS (
SELECT
uid_key,
COUNT(*) AS rows_in_cluster,
COUNT(DISTINCT repayment_amount) AS distinct_amounts_in_cluster
FROM keyed
GROUP BY uid_key
)
SELECT
COUNT(*) AS total_uid_clusters,
SUM(CASE WHEN rows_in_cluster > 1 THEN 1 ELSE 0 END) AS multi_row_clusters,
SUM(CASE WHEN rows_in_cluster > 1 AND distinct_amounts_in_cluster = 1 THEN 1 ELSE 0 END) AS multi_row_uniform_amount_clusters,
SUM(CASE WHEN rows_in_cluster > 1 AND distinct_amounts_in_cluster > 1 THEN 1 ELSE 0 END) AS multi_row_mixed_amount_clusters,
SUM(CASE WHEN rows_in_cluster > 1 AND distinct_amounts_in_cluster = 1 THEN rows_in_cluster - 1 ELSE 0 END) AS true_duplicate_rows_removed,
SUM(CASE WHEN rows_in_cluster > 1 THEN rows_in_cluster ELSE 0 END) AS rows_in_any_multi_row_cluster
FROM cluster_stats;
-- RESULT:
-- INTERPRETATION: multi_row_uniform_amount_clusters (like 256772/256773)
-- are the safe-to-collapse case; multi_row_mixed_amount_clusters are the
-- ones repayment_uid_dedup_test.sql wrongly collapsed, losing real repaid
-- money. If mixed-amount clusters dominate rows_in_any_multi_row_cluster,
-- that confirms most of the original 40.6% "duplication" was actually
-- legitimate distinct repayments, not retries.

-- Step 2: reconciliation rate using the refined (repayment_uid,
-- repayment_amount) composite dedup key -- only collapses rows that are
-- BOTH same uid AND same amount, leaving mixed-amount uid clusters intact.
WITH uid_amount_deduped AS (
SELECT phonenumber, repayment_amount, repayment_ts
FROM (
SELECT
phonenumber,
repayment_amount,
repayment_ts,
repayment_fid,
ROW_NUMBER() OVER (
PARTITION BY COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR)), repayment_amount
ORDER BY repayment_ts, repayment_fid
) AS rn
FROM :validation_schema.vw_bh_repay_dedup
)
WHERE rn = 1
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
COALESCE(SUM(ABS(r.repayment_amount)), 0) AS attributed_repaid_abs
FROM surviving_windows w
LEFT JOIN uid_amount_deduped r
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
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
COUNT_IF(state_only = 0 AND disbursement_only = 0) AS n_matched,
COUNT_IF(state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1) AS n_exact_match,
ROUND(100.0 * COUNT_IF(state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1)
/ NULLIF(COUNT_IF(state_only = 0 AND disbursement_only = 0), 0), 2) AS pct_exact_match_after_uid_amount_dedup
FROM joined;
-- RESULT:
-- INTERPRETATION: compare pct_exact_match_after_uid_amount_dedup against
-- the 68.6% baseline AND the 67.4% from collapsing on repayment_uid alone.
-- If this lands ABOVE 68.6%, the composite key is the real fix -- narrower
-- than repayment_uid alone, but actually safe. If it's flat or still below
-- baseline, even "same uid + same amount" isn't reliably "one real
-- payment" (e.g. two genuinely separate same-amount repayments could
-- coincidentally share a uid too), and repayment_uid shouldn't be used for
-- dedup at all -- fall back to treating the 68.6% baseline as the honest
-- number and rely on the materiality-based gate already documented in
-- borrower_history_validation_queries.sql.
