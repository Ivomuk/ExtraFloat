-- ============================================================================
-- prior_loan_leakage_quantification.sql -- ad-hoc diagnostic, not part of
-- the committed validation suite. Direct, cheap-to-run sanity check of the
-- look-ahead leak described in prior_loan_history_candidates_vs_blacklist_
-- export_corrected.sql's header, scoped to just the MOST RECENT prior loan
-- per agent (the same population most_recent_prior_dpd_within_30d already
-- uses) rather than the full cross-join over every prior loan -- run this
-- first, since it's far cheaper than the full corrected export, to get a
-- quick read on how big the leak actually is before running that heavier
-- query.
--
-- TWO INDEPENDENT QUERIES BELOW -- run each one separately (a CTE only
-- lives for the single statement that defines it).
-- ============================================================================


-- ----------------------------------------------------------------------
-- Query 1: for each agent's most recent prior loan (relative to their
-- latest label-eligible loan), how much does the CURRENT (leaky, unbounded)
-- "final state" differ from the CORRECTED (bounded to the target's own
-- loan_date) state?
-- ----------------------------------------------------------------------
WITH latest_eligible_loan_ranked AS (
    SELECT
        l.msisdn, l.disbursement_fid, l.loan_date,
        ROW_NUMBER() OVER (PARTITION BY l.msisdn ORDER BY l.loan_date DESC) AS latest_rn
    FROM hive.analytics.tmp_loan_label_assessment l
    WHERE l.label_eligible_30d = 1
),
latest_eligible_loan AS (
    SELECT msisdn, disbursement_fid, loan_date
    FROM latest_eligible_loan_ranked WHERE latest_rn = 1
),
most_recent_prior_loan AS (
    SELECT
        e.disbursement_fid AS target_disbursement_fid,
        e.loan_date AS target_loan_date,
        p.disbursement_fid AS prior_disbursement_fid,
        ROW_NUMBER() OVER (PARTITION BY e.msisdn ORDER BY p.disbursement_ts DESC) AS recency_rn
    FROM latest_eligible_loan e
    JOIN hive.analytics.tmp_disbursements p
      ON p.msisdn = e.msisdn
     AND p.disbursement_ts < (
            SELECT d.disbursement_ts FROM hive.analytics.tmp_disbursements d
            WHERE d.disbursement_fid = e.disbursement_fid
         )
),
xtrafloat_state_dated AS (
    SELECT
        disbursement_fid, principal_outstanding_ugx, total_outstanding_ugx,
        is_principal_settled, inserted_ts,
        CAST(DATE_PARSE(CAST(date_key AS VARCHAR), '%Y%m%d') AS DATE) AS state_date
    FROM analytics.momo_loan_book_tracker_xtrafloat_loan_state_daily
    WHERE ova = 'XTRAFLOAT-AGENT' AND date_key >= 20260101
),
leaky_state_ranked AS (
    SELECT
        m.target_disbursement_fid,
        s.principal_outstanding_ugx, s.total_outstanding_ugx, s.is_principal_settled,
        ROW_NUMBER() OVER (PARTITION BY m.target_disbursement_fid ORDER BY s.state_date DESC, s.inserted_ts DESC) AS rn
    FROM most_recent_prior_loan m
    JOIN xtrafloat_state_dated s ON s.disbursement_fid = m.prior_disbursement_fid
    WHERE m.recency_rn = 1
),
corrected_state_ranked AS (
    SELECT
        m.target_disbursement_fid,
        s.principal_outstanding_ugx, s.total_outstanding_ugx, s.is_principal_settled,
        ROW_NUMBER() OVER (PARTITION BY m.target_disbursement_fid ORDER BY s.state_date DESC, s.inserted_ts DESC) AS rn
    FROM most_recent_prior_loan m
    JOIN xtrafloat_state_dated s
      ON s.disbursement_fid = m.prior_disbursement_fid
     AND s.state_date <= m.target_loan_date
    WHERE m.recency_rn = 1
),
compared AS (
    SELECT
        l.target_disbursement_fid,
        l.principal_outstanding_ugx AS leaky_principal_outstanding_ugx,
        c.principal_outstanding_ugx AS corrected_principal_outstanding_ugx,
        l.is_principal_settled AS leaky_is_principal_settled,
        c.is_principal_settled AS corrected_is_principal_settled
    FROM leaky_state_ranked l
    LEFT JOIN corrected_state_ranked c
      ON c.target_disbursement_fid = l.target_disbursement_fid AND c.rn = 1
    WHERE l.rn = 1
)
SELECT
    COUNT(*) AS n_pairs_checked,
    COUNT(corrected_principal_outstanding_ugx) AS n_with_a_corrected_row_at_all,
    COUNT_IF(corrected_principal_outstanding_ugx IS NULL) AS n_corrected_row_missing_entirely,
    COUNT_IF(
        corrected_principal_outstanding_ugx IS NOT NULL
        AND leaky_principal_outstanding_ugx <> corrected_principal_outstanding_ugx
    ) AS n_principal_outstanding_differs,
    COUNT_IF(
        corrected_is_principal_settled IS NOT NULL
        AND leaky_is_principal_settled <> corrected_is_principal_settled
    ) AS n_is_principal_settled_differs,
    AVG(ABS(leaky_principal_outstanding_ugx - corrected_principal_outstanding_ugx))
        AS avg_abs_principal_outstanding_diff
FROM compared;


-- ----------------------------------------------------------------------
-- Query 2: run separately. How often does the CURRENT (coverage-based)
-- label_eligible_30d gate say "eligible" while the prior loan's own 30-day
-- window had NOT actually elapsed relative to the TARGET's own loan_date --
-- i.e. how often the current logic silently reads future information for
-- most_recent_prior_days_past_due_within_30d.
-- ----------------------------------------------------------------------
WITH latest_eligible_loan_ranked AS (
    SELECT
        l.msisdn, l.disbursement_fid, l.loan_date,
        ROW_NUMBER() OVER (PARTITION BY l.msisdn ORDER BY l.loan_date DESC) AS latest_rn
    FROM hive.analytics.tmp_loan_label_assessment l
    WHERE l.label_eligible_30d = 1
),
latest_eligible_loan AS (
    SELECT msisdn, disbursement_fid, loan_date
    FROM latest_eligible_loan_ranked WHERE latest_rn = 1
),
most_recent_prior_loan AS (
    SELECT
        e.disbursement_fid AS target_disbursement_fid,
        e.loan_date AS target_loan_date,
        p.disbursement_fid AS prior_disbursement_fid,
        ROW_NUMBER() OVER (PARTITION BY e.msisdn ORDER BY p.disbursement_ts DESC) AS recency_rn
    FROM latest_eligible_loan e
    JOIN hive.analytics.tmp_disbursements p
      ON p.msisdn = e.msisdn
     AND p.disbursement_ts < (
            SELECT d.disbursement_ts FROM hive.analytics.tmp_disbursements d
            WHERE d.disbursement_fid = e.disbursement_fid
         )
),
prior_loan_label AS (
    SELECT
        m.target_disbursement_fid,
        m.target_loan_date,
        l.disbursement_fid AS prior_disbursement_fid,
        l.loan_date AS prior_loan_date,
        l.label_eligible_30d AS coverage_based_eligible
    FROM most_recent_prior_loan m
    JOIN hive.analytics.tmp_loan_label_assessment l
      ON l.disbursement_fid = m.prior_disbursement_fid
    WHERE m.recency_rn = 1
)
SELECT
    COUNT(*) AS n_pairs_checked,
    COUNT_IF(coverage_based_eligible = 1) AS n_coverage_says_eligible,
    COUNT_IF(
        coverage_based_eligible = 1
        AND DATE_DIFF('day', prior_loan_date, target_loan_date) < 30
    ) AS n_leak_affected_rows,
    CAST(
        COUNT_IF(
            coverage_based_eligible = 1
            AND DATE_DIFF('day', prior_loan_date, target_loan_date) < 30
        ) AS DOUBLE
    ) / NULLIF(COUNT_IF(coverage_based_eligible = 1), 0) AS pct_of_eligible_rows_leak_affected
FROM prior_loan_label;
