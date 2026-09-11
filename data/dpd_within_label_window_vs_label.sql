-- ============================================================================
-- dpd_within_label_window_vs_label.sql
-- ============================================================================
-- Corrects a measurement-alignment problem in
-- dpd_bucket_distribution_vs_label.sql's Part 2: that query compared
-- bad_state_3dpd_30d against dpd_bucket/days_past_due at each loan's
-- ABSOLUTE LATEST observed snapshot -- which can be months after
-- disbursement (e.g. one raw sample loan sat at days_past_due=121),
-- completely outside bad_state_3dpd_30d's own 30-day-post-disbursement
-- label window. That's not apples-to-apples: a loan can resolve fine
-- WITHIN its 30-day window (bad_state=0, "good") yet still show a huge
-- days_past_due at some unrelated later point, which is exactly why that
-- query found avg_days_past_due nearly identical (117.6 vs 118.4) between
-- bad_state=0 and bad_state=1 within the "3+" bucket -- an artifact of
-- misalignment, not evidence duration doesn't discriminate.
--
-- This measures MAX(days_past_due) strictly WITHIN each loan's own
-- (loan_date, label_horizon_30d_end] window -- the same window
-- bad_state_3dpd_30d itself is defined over (see
-- loan_state_query_updated_materialized.txt lines 555-558, and
-- tmp_loan_label_assessment's own state_date > loan_date AND
-- state_date <= label_horizon_30d_end gating, e.g. lines 647-648).
--
-- Buckets chosen to test finer granularity than the raw dpd_bucket column
-- (confirmed via dpd_bucket_distribution_vs_label.sql's Part 1 to only
-- have 0/1/2/3+/NONE -- "3+" alone can't distinguish mildly late from
-- severely late), and to give a first read on where a genuinely
-- discriminating threshold might sit for a future prior-loan COUNT
-- feature (e.g. "count of prior loans whose max_dpd_within_30d exceeded
-- <threshold>").
-- ============================================================================

WITH xtrafloat_penalty_dated AS (
    SELECT
        disbursement_fid,
        days_past_due,
        CAST(DATE_PARSE(CAST(date_key AS VARCHAR), '%Y%m%d') AS DATE) AS state_date
    FROM analytics.momo_loan_book_tracker_xtrafloat_loan_state_daily
    WHERE ova = 'XTRAFLOAT-AGENT'
      AND date_key >= 20260101
),

max_dpd_within_30d AS (
    SELECT
        l.disbursement_fid,
        MAX(p.days_past_due) AS max_dpd_within_30d
    FROM hive.analytics.tmp_loan_label_assessment l
    LEFT JOIN xtrafloat_penalty_dated p
      ON p.disbursement_fid = l.disbursement_fid
     AND p.state_date > l.loan_date
     AND p.state_date <= l.label_horizon_30d_end
    WHERE l.label_eligible_30d = 1
    GROUP BY l.disbursement_fid
)

SELECT
    l.bad_state_3dpd_30d,
    CASE
        WHEN m.max_dpd_within_30d IS NULL THEN 'NEVER_PAST_DUE'
        WHEN m.max_dpd_within_30d <= 2 THEN '1-2'
        WHEN m.max_dpd_within_30d <= 6 THEN '3-6'
        WHEN m.max_dpd_within_30d <= 13 THEN '7-13'
        WHEN m.max_dpd_within_30d <= 29 THEN '14-29'
        ELSE '30+'
    END AS max_dpd_bucket_within_30d,
    COUNT(*) AS n
FROM hive.analytics.tmp_loan_label_assessment l
LEFT JOIN max_dpd_within_30d m
    ON m.disbursement_fid = l.disbursement_fid
WHERE l.label_eligible_30d = 1
GROUP BY 1, 2
ORDER BY 1, 2;
