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
-- already confirmed reliable elsewhere this session.
--
-- THREE INDEPENDENT QUERIES BELOW -- run each one separately. A CTE only
-- lives for the single statement that defines it (it is not a table), so
-- each query below repeats its own full WITH clause rather than assuming
-- an earlier query's CTEs are still queryable.
--
-- Query 1: distribution / sanity check of the computed value alone.
--   Already run -- confirmed sane: 7,025,955 closed loans, 100% non-null,
--   median 0 days, p90 5 days, p99 39 days, max 226 days, mean 2.22 days.
--   Only 199 negative values (0.0028%), immaterial.
--
-- Query 2: bridge coverage -- how many of those closed loans even resolve
--   to a disbursement_fid via tmp_target_loans (only covers the
--   2026-01-01 to 2026-07-14 training/validation cohort window).
--   Already run -- 5,695,188 / 7,025,955 (81.1%) resolved.
--
-- Query 3: cross-check agreement between the computed value and the
--   xtrafloat table's days_since_disbursement, for whichever loans Query 2
--   found a disbursement_fid for. FIRST VERSION (superseded, kept here as
--   a documented dead end) took xtrafloat's row at MAX(date_key) per
--   disbursement_fid -- that diverged badly (avg_abs_diff ~18.75 days,
--   only 18.8% exact match out of 5,695,188 compared, p90 diff 108 days),
--   because that table doesn't stop emitting rows at closure the way
--   tmp_loan_state does (its penalty flags are already confirmed
--   elsewhere in this file to persist on later daily rows past
--   resolution) -- so "last row" measures "how long this table kept
--   tracking the loan afterward," not tenure. CURRENT VERSION instead
--   takes xtrafloat's row nearest to (on or before) the loan's own
--   closure_date, for a true as-of-closure-day comparison.
-- ============================================================================


-- ----------------------------------------------------------------------
-- Query 1 (already run): distribution / sanity check.
-- ----------------------------------------------------------------------
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
)
SELECT
    COUNT(*) AS n_closed_loans,
    COUNT(computed_tenure_days) AS n_non_null_computed_tenure_days,
    COUNT_IF(computed_tenure_days < 0) AS n_negative_tenure_days,
    MIN(computed_tenure_days) AS min_tenure_days,
    approx_percentile(computed_tenure_days, 0.5) AS p50_tenure_days,
    approx_percentile(computed_tenure_days, 0.9) AS p90_tenure_days,
    approx_percentile(computed_tenure_days, 0.99) AS p99_tenure_days,
    MAX(computed_tenure_days) AS max_tenure_days,
    AVG(computed_tenure_days) AS avg_tenure_days
FROM closed_loans_computed_tenure;


-- ----------------------------------------------------------------------
-- Query 2: bridge coverage. Run this separately.
-- ----------------------------------------------------------------------
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

bridged_closed_loans AS (
    SELECT
        c.*,
        t.disbursement_fid
    FROM closed_loans_computed_tenure c
    JOIN hive.analytics.tmp_target_loans t
      ON t.target_loan_uid = c.loan_uid
     AND t.msisdn = c.customer_msisdn
)

SELECT
    (SELECT COUNT(*) FROM closed_loans_computed_tenure) AS n_closed_loans,
    (SELECT COUNT(*) FROM bridged_closed_loans) AS n_bridged_to_disbursement_fid;


-- ----------------------------------------------------------------------
-- Query 3: cross-check agreement against xtrafloat's
-- days_since_disbursement. Run this separately.
-- ----------------------------------------------------------------------
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

bridged_closed_loans AS (
    SELECT
        c.*,
        t.disbursement_fid
    FROM closed_loans_computed_tenure c
    JOIN hive.analytics.tmp_target_loans t
      ON t.target_loan_uid = c.loan_uid
     AND t.msisdn = c.customer_msisdn
),

-- NOTE: xtrafloat_final_state (MAX(date_key) per disbursement_fid, no
-- closure-day alignment) was tried first and diverged badly from
-- computed_tenure_days (avg_abs_diff ~18.75 days, only 18.8% exact match)
-- -- consistent with this table NOT stopping at closure the way
-- tmp_loan_state does (its penalty flags are already confirmed elsewhere
-- in this file to persist on later daily rows past resolution). So instead
-- of the loan's LAST row, take the row nearest to (on or before) that
-- loan's own closure_date -- a true apples-to-apples "what did each source
-- say tenure was, as of the closure day" comparison.
xtrafloat_dated AS (
    SELECT
        pen.disbursement_fid,
        pen.days_since_disbursement,
        pen.days_past_due,
        CAST(DATE_PARSE(CAST(pen.date_key AS VARCHAR), '%Y%m%d') AS DATE)
            AS state_date

    FROM analytics.momo_loan_book_tracker_xtrafloat_loan_state_daily pen
    WHERE pen.ova = 'XTRAFLOAT-AGENT'
      AND pen.date_key >= 20260101
),

xtrafloat_at_closure_ranked AS (
    SELECT
        b.loan_uid,
        b.computed_tenure_days,
        x.days_since_disbursement,
        x.days_past_due,

        ROW_NUMBER() OVER (
            PARTITION BY b.loan_uid
            ORDER BY x.state_date DESC
        ) AS closure_proximity_rn

    FROM bridged_closed_loans b
    JOIN xtrafloat_dated x
      ON x.disbursement_fid = b.disbursement_fid
     AND x.state_date <= b.closure_date
),

section_2_cross_check AS (
    SELECT
        computed_tenure_days,
        days_since_disbursement AS xtrafloat_days_since_disbursement,
        days_past_due AS xtrafloat_days_past_due,

        computed_tenure_days - days_since_disbursement
            AS tenure_diff

    FROM xtrafloat_at_closure_ranked
    WHERE closure_proximity_rn = 1
)

SELECT
    COUNT(*) AS n_compared,
    COUNT_IF(tenure_diff = 0) AS n_exact_match,
    COUNT_IF(ABS(tenure_diff) <= 1) AS n_within_1_day,
    AVG(ABS(tenure_diff)) AS avg_abs_diff,
    approx_percentile(ABS(tenure_diff), 0.9) AS p90_abs_diff,
    MAX(ABS(tenure_diff)) AS max_abs_diff
FROM section_2_cross_check;
