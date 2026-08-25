-- ============================================================================
-- b2b_diff_distribution_by_period.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. b2b_monthly_breakdown.sql found the blended
-- 68.6%/6.28% B2b figures are dragged down by a dateable April 2026
-- incident (with a May recovery tail) -- Jan-Mar sit in a healthier
-- 70-74% exact-match / 4.5-5.1% volume-error band. But "exact match" is a
-- strict to-the-UGX identity test, and even 70-74% is far from 100% --
-- b2b_diff_distribution.sql's materiality breakdown (87.9%/92.3% within
-- 1%/5% of principal) was computed on the BLENDED population, so it's not
-- yet known whether Jan-Mar's ~27-30% non-exact loans are mostly small/
-- immaterial too, or whether even the "clean" months hide a meaningful
-- share of large misattributions once you stop averaging them against
-- April's much bigger ones. This reruns the same materiality distribution
-- separately for Jan-Mar (baseline), April (incident), and May+ (recovery
-- tail) instead of assuming the pattern from the aggregate volume-error
-- number alone.
-- ============================================================================

WITH surviving_windows AS (
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
w.disbursement_ts,
w.disbursed_amount,
COALESCE(SUM(ABS(r.repayment_amount)), 0) AS attributed_repaid_abs
FROM surviving_windows w
LEFT JOIN :validation_schema.vw_bh_repay_dedup r
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
GROUP BY w.disbursement_fid, w.disbursement_ts, w.disbursed_amount
),
joined AS (
SELECT
pla.disbursement_ts,
pla.disbursed_amount,
(pla.attributed_repaid_abs - lsl.lifetime_repaid_ugx) AS diff_ugx,
lsl.lifetime_repaid_ugx
FROM per_loan_attributed pla
JOIN :validation_schema.vw_bh_loan_state_snapshot lsl
ON lsl.disbursement_fid = pla.disbursement_fid
)
SELECT
CASE
WHEN disbursement_ts < TIMESTAMP '2026-04-01 00:00:00.000' THEN 'Jan-Mar (baseline)'
WHEN disbursement_ts < TIMESTAMP '2026-05-01 00:00:00.000' THEN 'Apr (incident)'
ELSE 'May+ (recovery tail)'
END AS period,
CASE
WHEN disbursement_ts < TIMESTAMP '2026-04-01 00:00:00.000' THEN 1
WHEN disbursement_ts < TIMESTAMP '2026-05-01 00:00:00.000' THEN 2
ELSE 3
END AS period_order,
COUNT(*) AS n_matched,
SUM(CASE WHEN ABS(diff_ugx) <= 1 THEN 1 ELSE 0 END) AS n_exact,
SUM(CASE WHEN ABS(diff_ugx) > 1 AND ABS(diff_ugx) <= 1000 THEN 1 ELSE 0 END) AS n_diff_le_1k,
SUM(CASE WHEN ABS(diff_ugx) > 1000 AND ABS(diff_ugx) <= 10000 THEN 1 ELSE 0 END) AS n_diff_1k_10k,
SUM(CASE WHEN ABS(diff_ugx) > 10000 AND ABS(diff_ugx) <= 100000 THEN 1 ELSE 0 END) AS n_diff_10k_100k,
SUM(CASE WHEN ABS(diff_ugx) > 100000 AND ABS(diff_ugx) <= 500000 THEN 1 ELSE 0 END) AS n_diff_100k_500k,
SUM(CASE WHEN ABS(diff_ugx) > 500000 THEN 1 ELSE 0 END) AS n_diff_gt_500k,
SUM(CASE WHEN ABS(diff_ugx) > 1 AND ABS(diff_ugx) <= 0.01 * disbursed_amount THEN 1 ELSE 0 END) AS n_mismatch_within_1pct_of_principal,
SUM(CASE WHEN ABS(diff_ugx) > 1 AND ABS(diff_ugx) <= 0.05 * disbursed_amount THEN 1 ELSE 0 END) AS n_mismatch_within_5pct_of_principal,
approx_percentile(ABS(diff_ugx), 0.5) FILTER (WHERE ABS(diff_ugx) > 1) AS median_abs_diff_ugx,
approx_percentile(ABS(diff_ugx), 0.9) FILTER (WHERE ABS(diff_ugx) > 1) AS p90_abs_diff_ugx,
approx_percentile(ABS(diff_ugx), 0.99) FILTER (WHERE ABS(diff_ugx) > 1) AS p99_abs_diff_ugx,
SUM(ABS(diff_ugx)) AS total_abs_diff_ugx,
SUM(lifetime_repaid_ugx) AS total_lifetime_repaid_ugx,
ROUND(100.0 * SUM(ABS(diff_ugx)) / NULLIF(SUM(lifetime_repaid_ugx), 0), 4) AS pct_total_volume_error
FROM joined
GROUP BY 1, 2
ORDER BY 2;
-- RESULT:
-- INTERPRETATION: compare the n_mismatch_within_1pct/5pct_of_principal
-- shares and the n_diff_gt_500k / n_diff_100k_500k counts across the three
-- rows. If Jan-Mar's within-1pct/5pct shares are similarly high to the
-- blended population's 87.9%/92.3% (from b2b_diff_distribution.sql) AND
-- April's are meaningfully LOWER (a bigger share of April's mismatches are
-- large, not just more numerous), that confirms April is qualitatively
-- different -- a real misattribution incident -- not just "the same kind
-- of noise, more of it." If Jan-Mar's own within-1pct/5pct shares are
-- notably below the blended figures, that means even the baseline months
-- carry a real chunk of large mismatches, and the earlier "mostly
-- immaterial noise" conclusion needs qualifying rather than treated as
-- true everywhere.
