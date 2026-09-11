-- ============================================================================
-- dpd_bucket_distribution_vs_label.sql
-- ============================================================================
-- Both penalty-flag cuts tested so far bracket the problem from opposite
-- sides without solving it:
--   - ever_penalty_2_due ("ever assessed"): 92.3% of ALL loans -- saturates,
--     can't discriminate (see label_vs_penalty_comparison.sql).
--   - penalty_2_unpaid_at_latest ("still unpaid at closure"): 0.2% of all
--     loans -- directionally correct (2.4x lift in bad_state_3dpd_30d=1
--     rate: 0.39% vs 0.16%) but too rare to matter for 99.6% of bad loans
--     (see penalty_unpaid_vs_penalty_assessed_rate.sql).
--
-- Both are binary cuts forcing a continuous phenomenon (how many days a
-- loan actually took to resolve) into a single yes/no threshold. This
-- checks whether days_past_due / dpd_bucket -- both present on
-- analytics.momo_loan_book_tracker_xtrafloat_loan_state_daily, confirmed
-- via raw sample rows earlier this session (NULL/NONE once a loan is
-- FULLY_CLOSED, a real day-count + "3+" bucket while still owing) --
-- carries more usable signal by preserving magnitude instead of a single
-- cut point.
--
-- PART 1: what dpd_bucket values actually exist (only "NONE" and "3+" seen
-- in the small sample so far -- need the full distribution to know if
-- finer buckets exist, e.g. matching the business's own 1-7d/7-30d/30+d
-- risk-tier language from earlier this session's open_loan_aging_risk_
-- tiers.sql work).
--
-- PART 2: dpd_bucket (at the loan's LATEST observed snapshot -- same dedup
-- pattern as penalty_unpaid_vs_penalty_assessed_rate.sql) cross-tabbed
-- against bad_state_3dpd_30d, restricted to the model's actual training
-- population (label_eligible_30d = 1).
-- ============================================================================

-- PART 1: full value distribution (sanity check before Part 2)
SELECT
    dpd_bucket,
    COUNT(*) AS n
FROM analytics.momo_loan_book_tracker_xtrafloat_loan_state_daily
WHERE ova = 'XTRAFLOAT-AGENT'
  AND date_key >= 20260101
GROUP BY dpd_bucket
ORDER BY n DESC;

-- PART 2: dpd_bucket at latest observed snapshot vs. bad_state_3dpd_30d
WITH latest_dpd_ranked AS (
    SELECT
        disbursement_fid,
        days_past_due,
        dpd_bucket,
        ROW_NUMBER() OVER (
            PARTITION BY disbursement_fid
            ORDER BY date_key DESC, inserted_ts DESC
        ) AS latest_rn
    FROM analytics.momo_loan_book_tracker_xtrafloat_loan_state_daily
    WHERE ova = 'XTRAFLOAT-AGENT'
      AND date_key >= 20260101
),

latest_dpd AS (
    SELECT disbursement_fid, days_past_due, dpd_bucket
    FROM latest_dpd_ranked
    WHERE latest_rn = 1
)

SELECT
    l.bad_state_3dpd_30d,
    COALESCE(d.dpd_bucket, 'NO_MATCH') AS dpd_bucket_at_latest,
    COUNT(*) AS n,
    AVG(d.days_past_due) AS avg_days_past_due
FROM hive.analytics.tmp_loan_label_assessment l
LEFT JOIN latest_dpd d
    ON d.disbursement_fid = l.disbursement_fid
WHERE l.label_eligible_30d = 1
GROUP BY 1, 2
ORDER BY 1, n DESC;
