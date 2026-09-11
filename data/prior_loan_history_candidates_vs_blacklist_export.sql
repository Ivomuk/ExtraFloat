-- ============================================================================
-- prior_loan_history_candidates_vs_blacklist_export.sql
-- ============================================================================
-- Validates 5 candidate prior-loan-history signals against the business's
-- own blacklist ground truth BEFORE wiring any of them into training/scoring
-- SQL -- same discipline that caught the prior_late_fee_* feature hurting
-- results (see loan_state_query_updated_materialized.txt's
-- prior_penalty_features comment) and the OR ANOMALY_OPEN label bug: cheap
-- SQL-only validation first, code changes only for what survives it.
--
-- All 5 signals are computed strictly from loans disbursed BEFORE the
-- agent's most recent label-eligible loan -- mirrors prior_penalty_features's
-- join shape exactly (msisdn match, strict disbursement_ts < target's), so
-- none of them read the target loan's own outcome:
--
--   1. prior_max_loan_seq
--        Raw loan_seq (from analytics.momo_loan_book_tracker_xtrafloat_
--        loan_state_daily itself, not the derived loan_seq_at_scoring /
--        observed_prior_loan_count features already in the model) of the
--        agent's most recent PRIOR loan -- tests whether the raw sequence
--        number carries information the existing derived tenure features
--        don't.
--   2. prior_max_principal_outstanding_ugx / prior_max_total_outstanding_ugx
--        Magnitude of real, unrecovered loss on any prior loan (final
--        observed daily state) -- a continuous severity signal, unlike the
--        already-tested penalty-due/owed flags which turned out to be
--        near-universal (ever_penalty_1_due/2_due fire on ~92% of loans)
--        and uninformative.
--   3. prior_avg_collection_ratio
--        actual_collected_ugx / total_due_ugx, averaged across prior loans
--        -- a continuous repayment-shortfall signal.
--   4. prior_principal_unsettled_count
--        Count of prior loans where is_principal_settled = false -- a
--        clean "real default occurred" count, sharper than penalty flags
--        (penalties frequently get paid in full even when principal never
--        does -- see prior_avg_collection_ratio's rationale).
--   5. prior_max_dpd_within_30d (bucketed)
--        Re-derives dpd_within_label_window_vs_label.sql's bucketing
--        against the now-corrected bad_state_3dpd_30d label (that file's
--        results are stale -- they ran against the old, ANOMALY_OPEN-
--        contaminated label).
--
-- One row per agent, taking prior-history values as of their most recent
-- label-eligible loan (the point where the full accumulated prior history
-- is visible, matching what a live scoring feature would see). Cross-
-- reference against data/blacklist_aug_20260804.csv /
-- data/whitelist_aug_20260804.csv the same way
-- rollover_only_bad_vs_blacklist_export.sql already does.
-- ============================================================================

WITH xtrafloat_loan_final_state_ranked AS (
    SELECT
        pen.disbursement_fid,
        pen.loan_seq,
        pen.principal_outstanding_ugx,
        pen.total_outstanding_ugx,
        pen.actual_collected_ugx,
        pen.total_due_ugx,
        pen.is_principal_settled,

        ROW_NUMBER() OVER (
            PARTITION BY pen.disbursement_fid
            ORDER BY pen.date_key DESC, pen.inserted_ts DESC
        ) AS final_state_rn

    FROM analytics.momo_loan_book_tracker_xtrafloat_loan_state_daily pen
    WHERE pen.ova = 'XTRAFLOAT-AGENT'
      AND pen.date_key >= 20260101
),

-- One row per loan: its most recently observed daily state, so
-- principal_outstanding_ugx/total_outstanding_ugx/is_principal_settled
-- reflect where the loan actually ended up, not a mid-life snapshot.
xtrafloat_loan_final_state AS (
    SELECT
        disbursement_fid,
        loan_seq,
        principal_outstanding_ugx,
        total_outstanding_ugx,
        actual_collected_ugx,
        total_due_ugx,
        is_principal_settled
    FROM xtrafloat_loan_final_state_ranked
    WHERE final_state_rn = 1
),

xtrafloat_penalty_dated AS (
    SELECT
        disbursement_fid,
        days_past_due,
        CAST(DATE_PARSE(CAST(date_key AS VARCHAR), '%Y%m%d') AS DATE) AS state_date
    FROM analytics.momo_loan_book_tracker_xtrafloat_loan_state_daily
    WHERE ova = 'XTRAFLOAT-AGENT'
      AND date_key >= 20260101
),

-- Per-loan max days_past_due strictly within that loan's own 30-day label
-- window (same measurement-alignment discipline as
-- dpd_within_label_window_vs_label.sql), re-derived here against the
-- corrected label.
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
),

-- Mirrors prior_penalty_features's join shape exactly: computed strictly
-- from loans disbursed BEFORE the target loan, so this is genuinely
-- non-leaky (the target loan's own outcome is never read here).
prior_loan_history_candidates AS (
    SELECT
        current_loan.disbursement_fid AS target_disbursement_fid,

        MAX(fs.loan_seq) AS prior_max_loan_seq,

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

        MAX(dpd.max_dpd_within_30d) AS prior_max_dpd_within_30d

    FROM hive.analytics.tmp_target_loans current_loan

    LEFT JOIN hive.analytics.tmp_disbursements prior_loan
      ON prior_loan.msisdn = current_loan.msisdn
     AND prior_loan.disbursement_ts < current_loan.disbursement_ts

    LEFT JOIN xtrafloat_loan_final_state fs
      ON fs.disbursement_fid = prior_loan.disbursement_fid

    LEFT JOIN max_dpd_within_30d dpd
      ON dpd.disbursement_fid = prior_loan.disbursement_fid

    GROUP BY current_loan.disbursement_fid
),

-- One row per agent: their most recent label-eligible loan is where the
-- full accumulated prior history above is visible.
latest_eligible_loan_ranked AS (
    SELECT
        l.msisdn,
        l.disbursement_fid,

        ROW_NUMBER() OVER (
            PARTITION BY l.msisdn
            ORDER BY l.loan_date DESC
        ) AS latest_rn

    FROM hive.analytics.tmp_loan_label_assessment l
    WHERE l.label_eligible_30d = 1
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
    c.prior_max_principal_outstanding_ugx,
    c.prior_max_total_outstanding_ugx,
    c.prior_avg_collection_ratio,
    COALESCE(c.prior_principal_unsettled_count, 0)
        AS prior_principal_unsettled_count,

    CASE
        WHEN c.prior_max_dpd_within_30d IS NULL THEN 'NEVER_PAST_DUE'
        WHEN c.prior_max_dpd_within_30d <= 2 THEN '1-2'
        WHEN c.prior_max_dpd_within_30d <= 6 THEN '3-6'
        WHEN c.prior_max_dpd_within_30d <= 13 THEN '7-13'
        WHEN c.prior_max_dpd_within_30d <= 29 THEN '14-29'
        ELSE '30+'
    END AS prior_max_dpd_bucket_within_30d

FROM latest_eligible_loan_ranked e
JOIN agent_bad_history bh
  ON bh.msisdn = e.msisdn
LEFT JOIN prior_loan_history_candidates c
  ON c.target_disbursement_fid = e.disbursement_fid
WHERE e.latest_rn = 1;
