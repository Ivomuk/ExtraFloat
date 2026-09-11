-- ============================================================================
-- label_vs_penalty_comparison.sql
-- ============================================================================
-- Diagnostic follow-up to the prior-loan-penalty-history feature change
-- (see git log for loan_state_query_updated_materialized.txt /
-- loan_history_snapshot_query.txt / loan_history_features.py). That change
-- was feature-only -- the training label (bad_state_3dpd_30d) was left
-- untouched -- and after retraining, cal_pd's AUC against the business's
-- August whitelist/blacklist ground truth got WORSE, not better (overall
-- 0.6204 -> 0.5886; "Defaulter not paid back in the last 30 days" reason
-- 0.4830 -> 0.4458, moving further from 0.5).
--
-- This cross-tabs the current label (bad_state_3dpd_30d = 1 if the target
-- loan reaches days_aging > 3 within 30 days, or becomes ANOMALY_OPEN --
-- see loan_state_query_updated_materialized.txt lines ~555-558) against
-- whether that SAME target loan itself ever got the 48-hour late penalty
-- (ever_penalty_2_due, sourced from
-- analytics.momo_loan_book_tracker_xtrafloat_loan_state_daily's
-- authoritative penalty ledger -- not the target loan's own attribution-
-- heuristic-derived fields).
--
-- HOW TO READ THE RESULT:
--   - If bad_state_3dpd_30d=1/ever_penalty_2_due=true and
--     bad_state_3dpd_30d=0/ever_penalty_2_due=false dominate (the
--     "diagonal"), the two definitions mostly agree -- the label is
--     probably fine, and the worse post-retrain result is more likely
--     retrain variance than a label problem.
--   - If there's a large bad_state_3dpd_30d=0/ever_penalty_2_due=true
--     count -- loans the current label calls "good" that the business's
--     own penalty system flagged as late -- that's a real label mismatch,
--     and it would explain both the original 0.886-internal-vs-0.62-
--     blacklist AUC gap and why a feature-only fix couldn't close it (the
--     model's own training target doesn't match what's being checked
--     against).
--
-- Restricted to label_eligible_30d = 1 -- the same population the model
-- actually trains/validates on (see run_phase_2_2_loan_history_pd_features
-- in pd_model/preprocessing/loan_history_features.py).
-- ============================================================================

WITH target_penalty AS (
    SELECT
        disbursement_fid,
        bool_or(penalty_1_due) AS ever_penalty_1_due,
        bool_or(penalty_2_due) AS ever_penalty_2_due
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY disbursement_fid, date_key
                ORDER BY inserted_ts DESC
            ) AS rn
        FROM analytics.momo_loan_book_tracker_xtrafloat_loan_state_daily
        WHERE ova = 'XTRAFLOAT-AGENT'
          AND date_key >= 20260101
    )
    WHERE rn = 1
    GROUP BY disbursement_fid
)
SELECT
    l.bad_state_3dpd_30d,
    tp.ever_penalty_2_due,
    COUNT(*) AS n
FROM hive.analytics.tmp_loan_label_assessment l
LEFT JOIN target_penalty tp
    ON tp.disbursement_fid = l.disbursement_fid
WHERE l.label_eligible_30d = 1
GROUP BY 1, 2
ORDER BY 1, 2;
