-- ============================================================================
-- dpd_bucket_split_by_anomaly_open.sql
-- ============================================================================
-- dpd_within_label_window_vs_label.sql found a non-monotonic bad_state_3dpd_30d
-- rate across max_dpd_within_30d buckets: 9.85% (never past due), 23.48%
-- (1-2 days -- unexpectedly HIGH), 11.32%/10.86%/11.01% (3-29 days), 20.08%
-- (30+ days). Hypothesis: the "1-2 days" bucket's elevated rate isn't about
-- repayment lateness at all -- it's ANOMALY_OPEN loans (a second loan
-- disbursed while the first is unpaid, which can happen within a day or
-- two) getting swept into bad_state_3dpd_30d via its OR condition
-- (days_aging > 3 OR ANOMALY_OPEN -- see lines 555-558), independent of
-- how many days the loan was actually past due.
--
-- rollover_observed_30d (loan_state_query_updated_materialized.txt lines
-- 686-695) is exactly this signal -- ANOMALY_OPEN observed within the same
-- 30-day label window -- already computed in tmp_loan_label_assessment, no
-- need to recompute it. This splits each max_dpd_within_30d bucket by it.
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
    CASE
        WHEN m.max_dpd_within_30d IS NULL THEN 'NEVER_PAST_DUE'
        WHEN m.max_dpd_within_30d <= 2 THEN '1-2'
        WHEN m.max_dpd_within_30d <= 6 THEN '3-6'
        WHEN m.max_dpd_within_30d <= 13 THEN '7-13'
        WHEN m.max_dpd_within_30d <= 29 THEN '14-29'
        ELSE '30+'
    END AS max_dpd_bucket_within_30d,
    l.rollover_observed_30d,
    l.bad_state_3dpd_30d,
    COUNT(*) AS n
FROM hive.analytics.tmp_loan_label_assessment l
LEFT JOIN max_dpd_within_30d m
    ON m.disbursement_fid = l.disbursement_fid
WHERE l.label_eligible_30d = 1
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;
