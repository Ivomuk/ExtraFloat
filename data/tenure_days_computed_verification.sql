-- ============================================================================
-- tenure_days_computed_verification.sql -- ad-hoc diagnostic, not part of
-- the committed pipeline. Verifies a proposed replacement for the source
-- column ls.tenure_days, confirmed this session to be 0% populated across
-- every loan_status in both hive.analytics.tmp_loan_state (~39.6M rows) and
-- the raw source table analytics.momo_loan_book_tracker_loan_state_daily --
-- i.e. it is a dead column, not something the same-day-closure JOIN fix in
-- prior_loan_state_candidates could ever have populated.
--
-- Proposed replacement (for closed loans -- the only population the 4
-- affected features actually read tenure_days for):
--     DATE_DIFF('day', current_loan_start_date, closure_date)
-- computed directly on hive.analytics.tmp_loan_state, using two columns
-- already confirmed reliable elsewhere this session (closure_date: 100%
-- populated on genuinely terminal rows; current_loan_start_date: used
-- throughout prior_loan_state_candidates's JOIN conditions already).
--
-- This script:
--   1. Computes that expression for every closed loan and reports its
--      distribution (checking in particular for negative values, which
--      would indicate a data-integrity problem, not just a formula bug).
--   2. Cross-checks it, where possible, against an independent source:
--      analytics.momo_loan_book_tracker_xtrafloat_loan_state_daily's
--      days_since_disbursement, at that loan's final observed daily row
--      (same ROW_NUMBER() OVER (PARTITION BY disbursement_fid ORDER BY
--      date_key DESC, inserted_ts DESC) pattern as this file's own
--      xtrafloat_loan_final_state_ranked CTE). This cross-check is only
--      possible for closed loans that resolve to a disbursement_fid via
--      hive.analytics.tmp_target_loans's bridge, which only covers the
--      2026-01-01 to 2026-07-14 training/validation cohort window -- so
--      expect partial coverage here, not 100%.
--
-- Run this and inspect both sections before folding the DATE_DIFF
-- expression into loan_state_query_updated_materialized.txt /
-- loan_state_query_updated.txt's normalized_loan_state CTE.
-- ============================================================================

WITH closed_loans_computed_tenure AS (
    SELECT
        ls.customer_msisdn,
        ls.loan_uid,
        ls.loan_seq,
        ls.current_loan_start_date,
        ls.closure_date,
        ls.loan_status,

        DATE_DIFF(
            'day',
            ls.current_loan_start_date,
            ls.closure_date
        ) AS computed_tenure_days

    FROM hive.analytics.tmp_loan_state ls
    WHERE ls.closure_date IS NOT NULL
      AND ls.loan_status IN ('SETTLED', 'CLOSED', 'OVERPAID')
),

-- ----------------------------------------------------------------------
-- Section 1: distribution / sanity check of the computed value alone.
-- ----------------------------------------------------------------------
section_1_distribution AS (
    SELECT
        COUNT(*) AS n_closed_loans,
        COUNT(computed_tenure_days) AS n_non_null_computed_tenure_days,

        COUNT_IF(computed_tenure_days < 0)
            AS n_negative_tenure_days,

        MIN(computed_tenure_days) AS min_tenure_days,
        approx_percentile(computed_tenure_days, 0.5) AS p50_tenure_days,
        approx_percentile(computed_tenure_days, 0.9) AS p90_tenure_days,
        approx_percentile(computed_tenure_days, 0.99) AS p99_tenure_days,
        MAX(computed_tenure_days) AS max_tenure_days,
        AVG(computed_tenure_days) AS avg_tenure_days

    FROM closed_loans_computed_tenure
),

-- ----------------------------------------------------------------------
-- Section 2: cross-check against the xtrafloat table's
-- days_since_disbursement, for the subset of closed loans that resolve
-- to a disbursement_fid via tmp_target_loans's bridge.
-- ----------------------------------------------------------------------
bridged_closed_loans AS (
    SELECT
        c.*,
        t.disbursement_fid
    FROM closed_loans_computed_tenure c
    JOIN hive.analytics.tmp_target_loans t
      ON t.target_loan_uid = c.loan_uid
     AND t.msisdn = c.customer_msisdn
),

xtrafloat_final_state_ranked AS (
    SELECT
        pen.disbursement_fid,
        pen.days_since_disbursement,
        pen.days_past_due,

        ROW_NUMBER() OVER (
            PARTITION BY pen.disbursement_fid
            ORDER BY pen.date_key DESC, pen.inserted_ts DESC
        ) AS final_state_rn

    FROM analytics.momo_loan_book_tracker_xtrafloat_loan_state_daily pen
    WHERE pen.ova = 'XTRAFLOAT-AGENT'
      AND pen.date_key >= 20260101
),

xtrafloat_final_state AS (
    SELECT disbursement_fid, days_since_disbursement, days_past_due
    FROM xtrafloat_final_state_ranked
    WHERE final_state_rn = 1
),

section_2_cross_check AS (
    SELECT
        b.computed_tenure_days,
        x.days_since_disbursement AS xtrafloat_days_since_disbursement,
        x.days_past_due AS xtrafloat_days_past_due,

        b.computed_tenure_days - x.days_since_disbursement
            AS tenure_diff

    FROM bridged_closed_loans b
    JOIN xtrafloat_final_state x
      ON x.disbursement_fid = b.disbursement_fid
)

SELECT 'section_1_distribution' AS section, * FROM section_1_distribution;

-- Run separately (Trino/Athena clients generally only return the last
-- statement's result set per execution -- run each SELECT below on its
-- own if your client doesn't show every statement's output):

-- Coverage of the bridge itself.
-- SELECT
--     (SELECT COUNT(*) FROM closed_loans_computed_tenure) AS n_closed_loans,
--     (SELECT COUNT(*) FROM bridged_closed_loans) AS n_bridged_to_disbursement_fid,
--     (SELECT COUNT(*) FROM section_2_cross_check) AS n_with_xtrafloat_value;

-- Cross-check agreement, where both values are available.
-- SELECT
--     COUNT(*) AS n_compared,
--     COUNT_IF(tenure_diff = 0) AS n_exact_match,
--     COUNT_IF(ABS(tenure_diff) <= 1) AS n_within_1_day,
--     AVG(ABS(tenure_diff)) AS avg_abs_diff,
--     approx_percentile(ABS(tenure_diff), 0.9) AS p90_abs_diff,
--     MAX(ABS(tenure_diff)) AS max_abs_diff
-- FROM section_2_cross_check;
