-- ============================================================================
-- prior_loan_history_candidates_vs_blacklist_export.sql
-- ============================================================================
-- Validates candidate prior-loan-history signals against the business's own
-- blacklist ground truth BEFORE wiring any of them into training/scoring
-- SQL -- same discipline that caught the prior_late_fee_* feature hurting
-- results (see loan_state_query_updated_materialized.txt's
-- prior_penalty_features comment) and the OR ANOMALY_OPEN label bug: cheap
-- SQL-only validation first, code changes only for what survives it.
--
-- All signals are computed strictly from loans disbursed BEFORE the agent's
-- most recent label-eligible loan -- mirrors prior_penalty_features's join
-- shape exactly (msisdn match, strict disbursement_ts < target's), so none
-- of them read the target loan's own outcome:
--
--   1. prior_max_loan_seq
--        Raw loan_seq (from analytics.momo_loan_book_tracker_xtrafloat_
--        loan_state_daily itself, not the derived loan_seq_at_scoring /
--        observed_prior_loan_count features already in the model) of the
--        agent's most recent PRIOR loan.
--   2. prior_max_principal_outstanding_ugx / prior_max_total_outstanding_ugx
--        Magnitude of real, unrecovered loss on any prior loan (final
--        observed daily state).
--   3. prior_avg_collection_ratio
--        actual_collected_ugx / total_due_ugx, averaged across prior loans.
--   4. prior_principal_unsettled_count
--        Count of prior loans where is_principal_settled = false.
--   5. Duration (days_past_due), TWO framings -- see below.
--
-- ---------------------------------------------------------------------------
-- DURATION FRAMING -- rewritten after the first run
-- ---------------------------------------------------------------------------
-- The first version aggregated MAX(days_past_due) across EVERY prior loan.
-- That result (blacklist rate: NEVER_PAST_DUE=36%, 1-2=30%, 3-6=29%,
-- 7-13=39%, 14-29=42%, 30+=17% -- the "30+" bucket LOWEST of all, holding
-- 72% of the population) was not a genuine duration signal -- it was
-- confounded by prior_max_loan_seq's own finding (AUC=0.337, i.e. MORE
-- prior loans => SAFER agent): an agent with many prior loans has more
-- chances for at least one to have drifted past 30 days purely from
-- exposure, even if that agent is otherwise a good, prolific repeat
-- borrower. A raw MAX over an uncontrolled loan count conflates "duration
-- of lateness" with "how many loans this agent has taken."
--
-- This version tests two framings that don't have that confound:
--   5a. most_recent_prior_dpd_within_30d (bucketed)
--         The dpd outcome of the single most recent PRIOR loan only --
--         no aggregation across a variable-length history. NULL outcomes
--         are further split into NO_PRIOR_LOAN / NEVER_PAST_DUE_CONFIRMED /
--         NEVER_PAST_DUE_CENSORED (see the NEVER_PAST_DUE SPLIT comment
--         below) -- an earlier run folded all three together and the
--         combined bucket's blacklist rate (32%) sat above the mild-
--         lateness buckets (11-12%), breaking the otherwise-monotonic
--         pattern from 1-2 through 30+.
--   5b. prior_dpd_exceed_3_rate
--         COUNT(prior loans with max_dpd_within_30d > 3) /
--         COUNT(all prior loans) -- a rate, not a raw count, so it isn't
--         mechanically pulled toward 1 by having more prior loans.
-- prior_loan_count is also exported for transparency/sanity-checking
-- against prior_max_loan_seq.
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

-- Identifies, per target loan, which PRIOR loan is the most recent one --
-- needed for the recency-based duration framing (5a) so it isn't
-- aggregated across a variable-length history.
prior_loans_ranked AS (
    SELECT
        current_loan.disbursement_fid AS target_disbursement_fid,
        prior_loan.disbursement_fid AS prior_disbursement_fid,

        ROW_NUMBER() OVER (
            PARTITION BY current_loan.disbursement_fid
            ORDER BY prior_loan.disbursement_ts DESC
        ) AS recency_rn

    FROM hive.analytics.tmp_target_loans current_loan

    JOIN hive.analytics.tmp_disbursements prior_loan
      ON prior_loan.msisdn = current_loan.msisdn
     AND prior_loan.disbursement_ts < current_loan.disbursement_ts
),

-- Also pulls the most recent PRIOR loan's own label_eligible_30d (queried
-- directly against tmp_loan_label_assessment, NOT pre-filtered the way
-- max_dpd_within_30d is) so a NULL dpd outcome can be split into "genuinely
-- confirmed never past due" vs. "we don't actually know" (censored, or
-- never sampled as a target loan at all -- see the NEVER_PAST_DUE SPLIT
-- note below).
most_recent_prior_dpd AS (
    SELECT
        r.target_disbursement_fid,
        dpd.max_dpd_within_30d AS most_recent_prior_dpd_within_30d,
        COALESCE(pla.label_eligible_30d, 0)
            AS most_recent_prior_loan_label_eligible_30d
    FROM prior_loans_ranked r
    LEFT JOIN max_dpd_within_30d dpd
      ON dpd.disbursement_fid = r.prior_disbursement_fid
    LEFT JOIN hive.analytics.tmp_loan_label_assessment pla
      ON pla.disbursement_fid = r.prior_disbursement_fid
    WHERE r.recency_rn = 1
),

-- Mirrors prior_penalty_features's join shape exactly: computed strictly
-- from loans disbursed BEFORE the target loan, so this is genuinely
-- non-leaky (the target loan's own outcome is never read here).
prior_loan_history_candidates AS (
    SELECT
        current_loan.disbursement_fid AS target_disbursement_fid,

        MAX(fs.loan_seq) AS prior_max_loan_seq,
        COUNT(prior_loan.disbursement_fid) AS prior_loan_count,

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

        -- Rate, not raw count -- doesn't mechanically increase just from
        -- having more prior loans (see DURATION FRAMING note above).
        CASE
            WHEN COUNT(prior_loan.disbursement_fid) > 0
            THEN CAST(COUNT_IF(dpd.max_dpd_within_30d > 3) AS DOUBLE)
                 / COUNT(prior_loan.disbursement_fid)
        END AS prior_dpd_exceed_3_rate

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
    COALESCE(c.prior_loan_count, 0) AS prior_loan_count,
    c.prior_max_principal_outstanding_ugx,
    c.prior_max_total_outstanding_ugx,
    c.prior_avg_collection_ratio,
    COALESCE(c.prior_principal_unsettled_count, 0)
        AS prior_principal_unsettled_count,
    c.prior_dpd_exceed_3_rate,

    -- NEVER_PAST_DUE SPLIT: a NULL dpd outcome above previously meant
    -- either "genuinely never late" or "we have no confirmed outcome for
    -- this loan at all" (censored, or the prior loan was never itself
    -- sampled as a target loan in tmp_target_loans -- both leave no row in
    -- max_dpd_within_30d). Collapsing those together inflated the
    -- "NEVER_PAST_DUE" bucket's blacklist rate above the mild-lateness
    -- buckets in the first version of this query -- split them explicitly.
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

FROM latest_eligible_loan_ranked e
JOIN agent_bad_history bh
  ON bh.msisdn = e.msisdn
LEFT JOIN prior_loan_history_candidates c
  ON c.target_disbursement_fid = e.disbursement_fid
LEFT JOIN most_recent_prior_dpd mrd
  ON mrd.target_disbursement_fid = e.disbursement_fid
WHERE e.latest_rn = 1;
