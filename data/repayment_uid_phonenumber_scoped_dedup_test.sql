-- ============================================================================
-- repayment_uid_phonenumber_scoped_dedup_test.sql -- ad-hoc diagnostic, not
-- part of the committed validation suite. repayment_uid_dedup_test.sql's
-- dedup key (COALESCE(repayment_uid, repayment_fid), partitioned GLOBALLY)
-- came back net-harmful (66.9% exact match vs a 68.6-68.7% baseline) --
-- consistent with the earlier finding that repayment_uid is a batch/
-- settlement-run identifier, not a per-transaction key, since a single uid
-- has been directly confirmed to span multiple UNRELATED customers (see
-- the WORKED EXAMPLE in borrower_history_validation_queries.sql: one uid,
-- 256772 and 256774, three days apart).
--
-- Critically, that global partition doesn't just merge duplicate rows when
-- a uid spans customers -- ROW_NUMBER() OVER (PARTITION BY
-- COALESCE(repayment_uid, repayment_fid) ...) has NO phonenumber in the
-- partition key, so when N customers share one uid, only ONE of their N
-- rows survives AT ALL (whichever wins the ORDER BY tie-break) -- every
-- other customer's repayment is silently deleted from attribution, not
-- just deduplicated. That is a much more severe defect than "collapsing
-- true retries a little too aggressively," and is the most likely
-- explanation for the net-harmful result.
--
-- This tests whether SCOPING the same dedup key to phonenumber recovers
-- the benefit (catching a customer's own repeated retries under one uid,
-- which duplicate_repayment_burst_test.sql's fuzzy heuristic under-caught)
-- while eliminating the cross-customer deletion entirely by construction --
-- a row can only ever be "deduplicated away" by another row from the SAME
-- customer, never a different one.
--
-- Self-contained against the GATE 0 views (:validation_schema.vw_bh_*),
-- current as of :snapshot_dt/:as_of_load_ts.
-- ============================================================================

-- Step 1: decompose how much of the global dedup key's collapsing is
-- within-customer (recoverable safely) vs cross-customer (the dangerous
-- part, eliminated by phonenumber-scoping). If distinct_phone_scoped_approx
-- is close to distinct_global_approx, most duplication was already
-- within-customer and phonenumber-scoping changes little; if it's
-- meaningfully higher (more groups = less collapsing), a real chunk of the
-- global key's "duplicate rows" were actually distinct customers' rows
-- being incorrectly merged.
WITH agg AS (
SELECT
COUNT(*) AS total_repayment_rows,
approx_distinct(COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR))) AS distinct_global_approx,
approx_distinct(phonenumber || '|' || COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR))) AS distinct_phone_scoped_approx
FROM :validation_schema.vw_bh_repay_dedup
)
SELECT
total_repayment_rows,
distinct_global_approx,
total_repayment_rows - distinct_global_approx AS rows_collapsed_by_global_key,
distinct_phone_scoped_approx,
total_repayment_rows - distinct_phone_scoped_approx AS rows_collapsed_by_phone_scoped_key,
distinct_phone_scoped_approx - distinct_global_approx AS extra_groups_from_phone_scoping,
ROUND(100.0 * (distinct_phone_scoped_approx - distinct_global_approx)
/ NULLIF(total_repayment_rows - distinct_global_approx, 0), 2) AS pct_of_global_collapse_that_was_cross_customer
FROM agg;
-- RESULT:
-- INTERPRETATION: pct_of_global_collapse_that_was_cross_customer estimates
-- what fraction of the rows the global key "deduplicated away" actually
-- belonged to a DIFFERENT customer than the row that survived -- i.e. were
-- deleted from attribution entirely, not merged with a true duplicate of
-- themselves. A large value here (e.g. >20-30%) would mean the earlier
-- net-harmful result was driven substantially by silent cross-customer data
-- loss, not by over-aggressive-but-directionally-correct deduplication.

-- Step 2: re-test B2b's reconciliation rate using the phonenumber-scoped
-- dedup key. Identical methodology to repayment_uid_dedup_test.sql's Step
-- 2 -- only the PARTITION BY in uid_deduped changes (adds phonenumber).
WITH uid_deduped AS (
SELECT phonenumber, repayment_amount, repayment_ts
FROM (
SELECT
phonenumber,
repayment_amount,
repayment_ts,
repayment_fid,
ROW_NUMBER() OVER (
PARTITION BY phonenumber, COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR))
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
LEFT JOIN uid_deduped r
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
/ NULLIF(COUNT_IF(state_only = 0 AND disbursement_only = 0), 0), 2) AS pct_exact_match_after_phone_scoped_uid_dedup
FROM joined;
-- RESULT:
-- INTERPRETATION: compare pct_exact_match_after_phone_scoped_uid_dedup
-- against THREE reference points: (a) repayment_uid_dedup_test.sql's 66.9%
-- (global key, same current population) -- phone-scoping should do at
-- least as well as this, since it never deletes another customer's row;
-- (b) the current no-dedup baseline for this exact population (rerun GATE
-- 0's B2b block against the current :snapshot_dt if not already done this
-- session -- do not compare against the stale pre-fix 68.6%/68.7%); (c)
-- duplicate_repayment_burst_test.sql's fuzzy-heuristic result. If
-- phone-scoped dedup meaningfully BEATS the no-dedup baseline (not just
-- beats the global-key result), that is the real, safe fix -- add
-- phonenumber to the PARTITION BY in vw_bh_repay_dedup's dedup pass here
-- and in borrower_history.txt's real repay_raw/repay_dedup CTE. If it only
-- ties or barely beats no-dedup, the true retries this catches are too rare
-- to be worth the added complexity, and the dedup hypothesis should be
-- closed out as rejected (not just "still open") for this data.
