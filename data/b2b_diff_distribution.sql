-- ============================================================================
-- b2b_diff_distribution.sql -- ad-hoc diagnostic, not part of the committed
-- validation suite. Four independent hypotheses traced from hand-picked
-- "biggest mismatch" examples (loan_uid-aware windowing, gross-vs-net,
-- ever-anomalous exclusion, same-phonenumber/same-amount burst duplicates)
-- each turned out to be a REAL, exactly-confirmed mechanism for the
-- specific examples traced, but each only moved the population-wide exact
-- match rate by ~0.1-2.4%. That's a sign the top-N-by-|diff| sampling
-- approach keeps surfacing dramatic but rare outliers, not the dominant
-- cause of the ~31% gap.
--
-- Instead of guessing another narrative, this characterizes the actual
-- DISTRIBUTION of mismatch sizes across every surviving loan with a
-- matched state snapshot: is the gap dominated by a huge count of small/
-- immaterial differences (rounding, fees, a few thousand UGX -- essentially
-- noise for credit-risk purposes), or by a huge count of large,
-- loan-sized misattributions (materially dangerous)? Also reports the
-- aggregate error as a fraction of total repaid volume, which is often the
-- more decision-relevant number than a binary exact-match count.
-- ============================================================================

WITH attributed AS (
SELECT w.disbursement_fid, r.repayment_amount
FROM :validation_schema.vw_bh_repay_dedup r
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
pla.disbursed_amount,
pla.attributed_repaid_abs,
lsl.lifetime_repaid_ugx,
(pla.attributed_repaid_abs - lsl.lifetime_repaid_ugx) AS diff_ugx
FROM per_loan_attributed pla
JOIN :validation_schema.vw_bh_loan_state_snapshot lsl
ON lsl.disbursement_fid = pla.disbursement_fid
)
SELECT
COUNT(*) AS n_matched,
SUM(CASE WHEN ABS(diff_ugx) <= 1 THEN 1 ELSE 0 END) AS n_exact,
SUM(CASE WHEN ABS(diff_ugx) > 1 AND ABS(diff_ugx) <= 1000 THEN 1 ELSE 0 END) AS n_diff_le_1k,
SUM(CASE WHEN ABS(diff_ugx) > 1000 AND ABS(diff_ugx) <= 10000 THEN 1 ELSE 0 END) AS n_diff_1k_10k,
SUM(CASE WHEN ABS(diff_ugx) > 10000 AND ABS(diff_ugx) <= 100000 THEN 1 ELSE 0 END) AS n_diff_10k_100k,
SUM(CASE WHEN ABS(diff_ugx) > 100000 AND ABS(diff_ugx) <= 500000 THEN 1 ELSE 0 END) AS n_diff_100k_500k,
SUM(CASE WHEN ABS(diff_ugx) > 500000 THEN 1 ELSE 0 END) AS n_diff_gt_500k,
-- materiality relative to each loan's own principal, not just absolute UGX
SUM(CASE WHEN ABS(diff_ugx) > 1 AND ABS(diff_ugx) <= 0.01 * disbursed_amount THEN 1 ELSE 0 END) AS n_mismatch_within_1pct_of_principal,
SUM(CASE WHEN ABS(diff_ugx) > 1 AND ABS(diff_ugx) <= 0.05 * disbursed_amount THEN 1 ELSE 0 END) AS n_mismatch_within_5pct_of_principal,
-- percentiles of absolute mismatch size, mismatched loans only
approx_percentile(ABS(diff_ugx), 0.5) FILTER (WHERE ABS(diff_ugx) > 1) AS median_abs_diff_ugx,
approx_percentile(ABS(diff_ugx), 0.9) FILTER (WHERE ABS(diff_ugx) > 1) AS p90_abs_diff_ugx,
approx_percentile(ABS(diff_ugx), 0.99) FILTER (WHERE ABS(diff_ugx) > 1) AS p99_abs_diff_ugx,
-- aggregate error as a fraction of total repaid volume -- the
-- decision-relevant number for whether this matters for credit risk
SUM(ABS(diff_ugx)) AS total_abs_diff_ugx,
SUM(lifetime_repaid_ugx) AS total_lifetime_repaid_ugx,
ROUND(100.0 * SUM(ABS(diff_ugx)) / NULLIF(SUM(lifetime_repaid_ugx), 0), 4) AS pct_total_volume_error
FROM joined;
-- RESULT:
-- INTERPRETATION: if n_diff_le_1k + n_diff_1k_10k dominates n_matched (i.e.
-- most "mismatches" are a few thousand UGX -- rounding/fee noise) and
-- pct_total_volume_error is small (well under 1%), the 68.6% exact-match
-- GATE is simply too strict for what's actually a materially sound
-- reconciliation -- the fix is to loosen GATE B2b's threshold (e.g. match
-- within 1% of principal, not within 1 UGX), not to keep chasing individual
-- root causes. If instead n_diff_gt_500k or n_diff_100k_500k is large and
-- pct_total_volume_error is high, the gap is materially real and the
-- dominant cause is still undiscovered -- worth then looking at whether
-- disbursement_only/state_only (no-match, not just mismatched-amount) rows
-- are the bigger population, since this query only looks at loans that
-- matched a snapshot at all.
