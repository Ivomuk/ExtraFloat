-- ============================================================================
-- attribution_materiality_verification.sql -- ad-hoc diagnostic, not part
-- of the committed validation suite. anomaly_open_fix_verification.sql
-- confirmed the ANOMALY_OPEN fix at the strictest possible bar (exact
-- match to within 1 UGX): 72.9% pooled, up from ~68.8% pre-fix. But every
-- prior materiality analysis in this project (b2b_diff_distribution_by_
-- period.sql, pre-fix) found strict exact-match understates real accuracy
-- substantially -- even the "clean" Jan-Mar period only hit ~71.8% exact
-- match despite being 91.1%/95.1% accurate within 1%/5% of loan principal.
-- This re-runs that same materiality methodology (from GATE 0's B2b block
-- in borrower_history_validation_queries.sql), but against production's
-- own CORRECTED total_recovered on tbl_bh_loan_final (post ANOMALY_OPEN
-- fix), not a re-derived proxy -- the number to actually quote if asked
-- "how well is attribution working."
--
-- Requires the current tbl_bh_loan_final (checkpoint 1) to already exist,
-- built from the current data/borrower_history.txt via
-- scripts/build_vw_bh_output.py on :snapshot_dt=20260609/
-- :as_of_load_ts=2026-09-09 (same table anomaly_open_fix_verification.sql
-- used). Substitute your_schema below.
-- ============================================================================

-- Step 1: materiality tiers by ever_anomaly_open, for a direct
-- apples-to-apples comparison against anomaly_open_fix_verification.sql's
-- strict-match result (100% / 70.3% when ls available) -- confirms the
-- materiality view is consistent with, not contradicting, that result.
WITH agg AS (
SELECT
ever_anomaly_open,
COUNT(*) AS n_loans,
COUNT_IF(ls_lifetime_repaid_ugx IS NULL) AS n_no_loan_state_row,
COUNT_IF(ls_lifetime_repaid_ugx IS NOT NULL) AS n_matched,
COUNT_IF(ls_lifetime_repaid_ugx IS NOT NULL
AND ABS(total_recovered - ls_lifetime_repaid_ugx) <= 1) AS n_exact_match,
COUNT_IF(ls_lifetime_repaid_ugx IS NOT NULL
AND ABS(total_recovered - ls_lifetime_repaid_ugx) > 1
AND ABS(total_recovered - ls_lifetime_repaid_ugx) <= 0.01 * disbursed_amount) AS n_within_1pct_only,
COUNT_IF(ls_lifetime_repaid_ugx IS NOT NULL
AND ABS(total_recovered - ls_lifetime_repaid_ugx) > 0.01 * disbursed_amount
AND ABS(total_recovered - ls_lifetime_repaid_ugx) <= 0.05 * disbursed_amount) AS n_within_5pct_only,
approx_percentile(
CASE WHEN ls_lifetime_repaid_ugx IS NOT NULL
THEN ABS(total_recovered - ls_lifetime_repaid_ugx) END, 0.95) AS p95_abs_diff_ugx,
SUM(CASE WHEN ls_lifetime_repaid_ugx IS NOT NULL
THEN ABS(total_recovered - ls_lifetime_repaid_ugx) ELSE 0 END) AS total_abs_diff_ugx
FROM your_schema.tbl_bh_loan_final
GROUP BY ever_anomaly_open
)
SELECT
ever_anomaly_open,
n_loans,
n_matched,
n_exact_match,
ROUND(100.0 * n_exact_match / NULLIF(n_matched, 0), 2) AS pct_exact_match,
n_within_1pct_only,
ROUND(100.0 * (n_exact_match + n_within_1pct_only) / NULLIF(n_matched, 0), 2) AS pct_within_1pct_cumulative,
n_within_5pct_only,
ROUND(100.0 * (n_exact_match + n_within_1pct_only + n_within_5pct_only) / NULLIF(n_matched, 0), 2) AS pct_within_5pct_cumulative,
p95_abs_diff_ugx,
total_abs_diff_ugx
FROM agg
ORDER BY ever_anomaly_open;
-- RESULT (Step 1):

-- Step 2: pooled across the whole population -- the single headline number
-- to quote when asked "how well is attribution working."
WITH agg AS (
SELECT
COUNT(*) AS n_loans,
COUNT_IF(ls_lifetime_repaid_ugx IS NULL) AS n_no_loan_state_row,
COUNT_IF(ls_lifetime_repaid_ugx IS NOT NULL) AS n_matched,
COUNT_IF(ls_lifetime_repaid_ugx IS NOT NULL
AND ABS(total_recovered - ls_lifetime_repaid_ugx) <= 1) AS n_exact_match,
COUNT_IF(ls_lifetime_repaid_ugx IS NOT NULL
AND ABS(total_recovered - ls_lifetime_repaid_ugx) > 1
AND ABS(total_recovered - ls_lifetime_repaid_ugx) <= 0.01 * disbursed_amount) AS n_within_1pct_only,
COUNT_IF(ls_lifetime_repaid_ugx IS NOT NULL
AND ABS(total_recovered - ls_lifetime_repaid_ugx) > 0.01 * disbursed_amount
AND ABS(total_recovered - ls_lifetime_repaid_ugx) <= 0.05 * disbursed_amount) AS n_within_5pct_only,
approx_percentile(
CASE WHEN ls_lifetime_repaid_ugx IS NOT NULL
THEN ABS(total_recovered - ls_lifetime_repaid_ugx) END, 0.95) AS p95_abs_diff_ugx,
SUM(CASE WHEN ls_lifetime_repaid_ugx IS NOT NULL
THEN ABS(total_recovered - ls_lifetime_repaid_ugx) ELSE 0 END) AS total_abs_diff_ugx,
SUM(disbursed_amount) AS total_disbursed_ugx
FROM your_schema.tbl_bh_loan_final
)
SELECT
'POOLED_ALL_LOANS' AS population,
n_loans,
n_matched,
n_exact_match,
ROUND(100.0 * n_exact_match / NULLIF(n_matched, 0), 2) AS pct_exact_match,
n_within_1pct_only,
ROUND(100.0 * (n_exact_match + n_within_1pct_only) / NULLIF(n_matched, 0), 2) AS pct_within_1pct_cumulative,
n_within_5pct_only,
ROUND(100.0 * (n_exact_match + n_within_1pct_only + n_within_5pct_only) / NULLIF(n_matched, 0), 2) AS pct_within_5pct_cumulative,
p95_abs_diff_ugx,
total_abs_diff_ugx,
ROUND(100.0 * total_abs_diff_ugx / NULLIF(total_disbursed_ugx, 0), 4) AS pct_volume_error
FROM agg;
-- RESULT (Step 2):
-- INTERPRETATION: pct_within_1pct_cumulative and pct_within_5pct_cumulative
-- are the numbers to quote for "how well is attribution working" in a
-- practical sense -- expect them to sit well above the 72.9% strict
-- exact-match figure, consistent with the pre-fix Jan-Mar precedent
-- (91.1%/95.1%). pct_volume_error (total UGX misattributed as a share of
-- total UGX disbursed) is the complementary financial-materiality framing
-- -- a small pct_volume_error alongside a modest pct_exact_match means the
-- mismatches that remain are mostly small-dollar/rounding-scale, not large
-- systematic misses. If pct_within_5pct_cumulative is NOT meaningfully
-- higher than pct_exact_match, that would mean the remaining gap is unlike
-- the pre-fix pattern (concentrated in small differences) and is instead
-- large, material mismatches -- worth re-examining rather than assuming
-- the fix generalizes the same way materiality did before it.
