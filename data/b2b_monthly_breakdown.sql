-- ============================================================================
-- b2b_monthly_breakdown.sql -- ad-hoc diagnostic, not part of the committed
-- validation suite. B2b's 68.6% exact-match rate is a single number across
-- the entire snapshot window. Every concrete mismatch mechanism traced by
-- hand so far (the 7-way loan_uid merge, the repayment_uid duplicate
-- bursts for 256772/256773) clustered around the SAME few days in early
-- April -- raising the question of whether the mismatch rate is roughly
-- uniform over time, or concentrated in specific months (a bounded
-- incident inflating the aggregate number) vs. spread evenly (an ongoing
-- structural gap). Breaks B2b's reconciliation down by each loan's
-- disbursement month, using the SAME baseline attribution as the
-- production-gating B2b query (repayment_fid-level dedup only -- not any
-- of the repayment_uid dedup variants still being tested), so this is a
-- clean read on the current, real behavior.
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
w.disbursement_ts,
w.disbursed_amount,
COALESCE(SUM(ABS(a.repayment_amount)), 0) AS attributed_repaid_abs
FROM :validation_schema.vw_bh_surviving_loans w
LEFT JOIN attributed a ON a.disbursement_fid = w.disbursement_fid
GROUP BY w.disbursement_fid, w.disbursement_ts, w.disbursed_amount
),
joined AS (
SELECT
pla.disbursement_ts,
pla.attributed_repaid_abs,
lsl.lifetime_repaid_ugx
FROM per_loan_attributed pla
JOIN :validation_schema.vw_bh_loan_state_snapshot lsl
ON lsl.disbursement_fid = pla.disbursement_fid
)
SELECT
date_trunc('month', disbursement_ts) AS disb_month,
COUNT(*) AS n_matched,
COUNT_IF(ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1) AS n_exact_match,
ROUND(100.0 * COUNT_IF(ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1)
/ NULLIF(COUNT(*), 0), 2) AS pct_exact_match,
SUM(ABS(attributed_repaid_abs - lifetime_repaid_ugx)) AS total_abs_diff_ugx,
SUM(lifetime_repaid_ugx) AS total_lifetime_repaid_ugx,
ROUND(100.0 * SUM(ABS(attributed_repaid_abs - lifetime_repaid_ugx))
/ NULLIF(SUM(lifetime_repaid_ugx), 0), 4) AS pct_volume_error
FROM joined
GROUP BY date_trunc('month', disbursement_ts)
ORDER BY disb_month;
-- RESULT:
-- INTERPRETATION: if pct_exact_match is roughly flat (within a few points)
-- across every month, the ~31% gap is an ongoing structural property of
-- the time-window heuristic, and the materiality-based gate already
-- documented in borrower_history_validation_queries.sql is the right
-- long-term framing. If instead a handful of months (e.g. the one
-- containing the traced April batch event) show a MUCH lower
-- pct_exact_match / much higher pct_volume_error than the rest, that
-- month is likely a bounded incident (a source-system migration, a
-- reconciliation backfill, a one-time batch reprocessing run) -- worth
-- naming to the table owner specifically, since excluding or separately
-- explaining that one period could bring the overall rate much closer to
-- target without any code change. n_matched per month also roughly shows
-- loan volume growth/seasonality, useful context for reading the other
-- two columns.
