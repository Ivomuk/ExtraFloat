-- ============================================================================
-- prior_loan_history_candidates_vs_blacklist_export_corrected.sql
-- ============================================================================
-- Corrected sibling of prior_loan_history_candidates_vs_blacklist_export.sql,
-- built to quantify how much of that file's validated AUCs (prior_avg_
-- collection_ratio AUC=0.786, most_recent_prior_dpd's monotonic 2.25%->29.24%
-- climb, etc.) were inflated by look-ahead leakage.
--
-- THE LEAK (confirmed against the live SQL, not assumed): the original file's
-- xtrafloat_loan_final_state takes each prior loan's ABSOLUTE final/lifetime
-- state (ROW_NUMBER by date_key DESC, no upper bound), and max_dpd_within_30d
-- gates on the prior loan's OWN label_eligible_30d -- a data-COVERAGE check
-- (was 30 days of state data ever recorded for that loan, as of whenever this
-- query runs), not a time-ELAPSED check (had 30 real days passed since the
-- prior loan's disbursement, as of the TARGET loan's own decision point).
-- Since prior loans commonly stay open 100+ days while newer loans are
-- disbursed in between (confirmed this session via check_concurrent_loans.sql),
-- both mechanisms can silently read information recorded weeks/months AFTER
-- the target loan's own disbursement date.
--
-- THE FIX, applied here: every prior-loan signal is now computed PER
-- (target loan, prior loan) PAIR, bounded to information dated on or before
-- the TARGET's own loan_date -- mirroring the same point-in-time discipline
-- prior_loan_state_candidates already uses correctly elsewhere in this
-- codebase. Output schema is IDENTICAL to the original file (same column
-- names, same population definition) so
-- scripts/check_prior_loan_history_candidates_vs_blacklist.py runs on this
-- export UNMODIFIED -- run both files' exports through that script and
-- compare the two AUC tables directly.
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

xtrafloat_state_dated AS (
    SELECT
        disbursement_fid,
        loan_seq,
        principal_outstanding_ugx,
        total_outstanding_ugx,
        actual_collected_ugx,
        total_due_ugx,
        is_principal_settled,
        inserted_ts,
        CAST(DATE_PARSE(CAST(date_key AS VARCHAR), '%Y%m%d') AS DATE) AS state_date
    FROM analytics.momo_loan_book_tracker_xtrafloat_loan_state_daily
    WHERE ova = 'XTRAFLOAT-AGENT'
      AND date_key >= 20260101
),

-- One row per agent: their most recent label-eligible loan (same population
-- as the original file), now also carrying its own loan_date so every
-- downstream join can bound prior-loan state to "as of this date."
latest_eligible_loan_ranked AS (
    SELECT
        l.msisdn,
        l.disbursement_fid,
        l.loan_date,

        ROW_NUMBER() OVER (
            PARTITION BY l.msisdn
            ORDER BY l.loan_date DESC
        ) AS latest_rn

    FROM hive.analytics.tmp_loan_label_assessment l
    WHERE l.label_eligible_30d = 1
),

latest_eligible_loan AS (
    SELECT msisdn, disbursement_fid, loan_date
    FROM latest_eligible_loan_ranked
    WHERE latest_rn = 1
),

-- Target loan's own disbursement_ts, joined once here rather than via a
-- correlated subquery below.
latest_eligible_loan_dated AS (
    SELECT
        e.msisdn,
        e.disbursement_fid,
        e.loan_date,
        d.disbursement_ts
    FROM latest_eligible_loan e
    JOIN hive.analytics.tmp_disbursements d
      ON d.disbursement_fid = e.disbursement_fid
),

-- All (target, prior) pairs for the population above -- same join shape as
-- the original file's prior_loan_history_candidates, just materialized
-- explicitly here so it can be joined against the daily xtrafloat table
-- with the target's own loan_date as the bound.
target_prior_pairs AS (
    SELECT
        current_loan.disbursement_fid AS target_disbursement_fid,
        current_loan.loan_date AS target_loan_date,
        prior_loan.disbursement_fid AS prior_disbursement_fid,
        prior_loan.disbursement_ts AS prior_disbursement_ts
    FROM latest_eligible_loan_dated current_loan
    JOIN hive.analytics.tmp_disbursements prior_loan
      ON prior_loan.msisdn = current_loan.msisdn
     AND prior_loan.disbursement_ts < current_loan.disbursement_ts
),

-- CORRECTED final state: per (target, prior) pair, bounded to the prior
-- loan's state as recorded ON OR BEFORE the TARGET's own loan_date -- not
-- the prior loan's unbounded lifetime-final state.
corrected_final_state_ranked AS (
    SELECT
        p.target_disbursement_fid,
        p.prior_disbursement_fid,
        s.loan_seq,
        s.principal_outstanding_ugx,
        s.total_outstanding_ugx,
        s.actual_collected_ugx,
        s.total_due_ugx,
        s.is_principal_settled,

        ROW_NUMBER() OVER (
            PARTITION BY p.target_disbursement_fid, p.prior_disbursement_fid
            ORDER BY s.state_date DESC, s.inserted_ts DESC
        ) AS state_rn

    FROM target_prior_pairs p
    JOIN xtrafloat_state_dated s
      ON s.disbursement_fid = p.prior_disbursement_fid
     AND s.state_date <= p.target_loan_date
),

corrected_final_state AS (
    SELECT
        target_disbursement_fid,
        prior_disbursement_fid,
        loan_seq,
        principal_outstanding_ugx,
        total_outstanding_ugx,
        actual_collected_ugx,
        total_due_ugx,
        is_principal_settled
    FROM corrected_final_state_ranked
    WHERE state_rn = 1
),

-- CORRECTED dpd: per (target, prior) pair, only counted once 30 real days
-- have elapsed since the PRIOR loan's own disbursement RELATIVE TO THE
-- TARGET's loan_date -- not gated on present-day data coverage.
-- is_eligible = 1 marks that this (target, prior) pair passed the
-- elapsed-time gate at all (a row exists here only if it did) -- lets
-- most_recent_prior_dpd below distinguish "not yet eligible" (no row here,
-- NULL after the LEFT JOIN) from "eligible but confirmed never late"
-- (row exists, MAX(days_past_due) is NULL because no penalty rows matched)
-- without a correlated subquery.
corrected_max_dpd_within_30d AS (
    SELECT
        p.target_disbursement_fid,
        p.prior_disbursement_fid,
        MAX(pd.days_past_due) AS max_dpd_within_30d,
        1 AS is_eligible
    FROM target_prior_pairs p
    JOIN hive.analytics.tmp_loan_label_assessment l
      ON l.disbursement_fid = p.prior_disbursement_fid
    LEFT JOIN xtrafloat_penalty_dated pd
      ON pd.disbursement_fid = l.disbursement_fid
     AND pd.state_date > l.loan_date
     AND pd.state_date <= l.label_horizon_30d_end
     AND pd.state_date <= p.target_loan_date
    WHERE DATE_DIFF('day', l.loan_date, p.target_loan_date) >= 30
    GROUP BY p.target_disbursement_fid, p.prior_disbursement_fid
),

-- Identifies, per target loan, which PRIOR loan is the most recent one --
-- same recency framing as the original file's prior_loans_ranked.
prior_loans_ranked AS (
    SELECT
        target_disbursement_fid,
        prior_disbursement_fid,

        ROW_NUMBER() OVER (
            PARTITION BY target_disbursement_fid
            ORDER BY prior_disbursement_ts DESC
        ) AS recency_rn

    FROM target_prior_pairs
),

most_recent_prior_dpd AS (
    SELECT
        r.target_disbursement_fid,
        dpd.max_dpd_within_30d AS most_recent_prior_dpd_within_30d,
        COALESCE(dpd.is_eligible, 0) AS most_recent_prior_loan_label_eligible_30d

    FROM prior_loans_ranked r
    LEFT JOIN corrected_max_dpd_within_30d dpd
      ON dpd.target_disbursement_fid = r.target_disbursement_fid
     AND dpd.prior_disbursement_fid = r.prior_disbursement_fid
    WHERE r.recency_rn = 1
),

prior_loan_history_candidates AS (
    SELECT
        p.target_disbursement_fid,

        MAX(fs.loan_seq) AS prior_max_loan_seq,
        COUNT(p.prior_disbursement_fid) AS prior_loan_count,

        MAX(fs.principal_outstanding_ugx)
            AS prior_max_principal_outstanding_ugx,
        MAX(fs.total_outstanding_ugx)
            AS prior_max_total_outstanding_ugx,

        AVG(
            CASE WHEN COALESCE(fs.total_due_ugx, 0) > 0
                 THEN fs.actual_collected_ugx / fs.total_due_ugx
            END
        ) AS prior_avg_collection_ratio,

        COUNT_IF(fs.is_principal_settled = false)
            AS prior_principal_unsettled_count,

        CASE
            WHEN COUNT(p.prior_disbursement_fid) > 0
            THEN CAST(COUNT_IF(dpd.max_dpd_within_30d > 3) AS DOUBLE)
                 / COUNT(p.prior_disbursement_fid)
        END AS prior_dpd_exceed_3_rate

    FROM target_prior_pairs p

    LEFT JOIN corrected_final_state fs
      ON fs.target_disbursement_fid = p.target_disbursement_fid
     AND fs.prior_disbursement_fid = p.prior_disbursement_fid

    LEFT JOIN corrected_max_dpd_within_30d dpd
      ON dpd.target_disbursement_fid = p.target_disbursement_fid
     AND dpd.prior_disbursement_fid = p.prior_disbursement_fid

    GROUP BY p.target_disbursement_fid
),

agent_bad_history AS (
    SELECT
        msisdn,
        MAX(bad_state_3dpd_30d) AS has_bad_loan_new_label
    FROM hive.analytics.tmp_loan_label_assessment
    WHERE label_eligible_30d = 1
    GROUP BY msisdn
)

SELECT
    e.msisdn,
    bh.has_bad_loan_new_label,

    COALESCE(c.prior_max_loan_seq, 0) AS prior_max_loan_seq,
    COALESCE(c.prior_loan_count, 0) AS prior_loan_count,
    c.prior_max_principal_outstanding_ugx,
    c.prior_max_total_outstanding_ugx,
    c.prior_avg_collection_ratio,
    COALESCE(c.prior_principal_unsettled_count, 0)
        AS prior_principal_unsettled_count,
    c.prior_dpd_exceed_3_rate,

    CASE
        WHEN mrd.target_disbursement_fid IS NULL THEN 'NO_PRIOR_LOAN'
        WHEN mrd.most_recent_prior_dpd_within_30d IS NULL
         AND mrd.most_recent_prior_loan_label_eligible_30d = 1
            THEN 'NEVER_PAST_DUE_CONFIRMED'
        WHEN mrd.most_recent_prior_dpd_within_30d IS NULL
            THEN 'NEVER_PAST_DUE_CENSORED'
        WHEN mrd.most_recent_prior_dpd_within_30d <= 2 THEN '1-2'
        WHEN mrd.most_recent_prior_dpd_within_30d <= 6 THEN '3-6'
        WHEN mrd.most_recent_prior_dpd_within_30d <= 13 THEN '7-13'
        WHEN mrd.most_recent_prior_dpd_within_30d <= 29 THEN '14-29'
        ELSE '30+'
    END AS most_recent_prior_dpd_bucket_within_30d

FROM latest_eligible_loan e
JOIN agent_bad_history bh
  ON bh.msisdn = e.msisdn
LEFT JOIN prior_loan_history_candidates c
  ON c.target_disbursement_fid = e.disbursement_fid
LEFT JOIN most_recent_prior_dpd mrd
  ON mrd.target_disbursement_fid = e.disbursement_fid;
