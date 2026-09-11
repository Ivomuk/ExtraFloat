-- ============================================================================
-- new_label_vs_blacklist_prevalidation.sql
-- ============================================================================
-- Pre-retrain sanity check for the bad_state_3dpd_30d label fix (removed
-- "OR ANOMALY_OPEN" -- see loan_state_query_updated_materialized.txt's
-- ANOMALY_OPEN LABEL DECISION comment). Run this AFTER re-running Statement
-- 4 of that file (rebuilding hive.analytics.tmp_loan_label_assessment with
-- the new label formula) but BEFORE spending a full retrain cycle -- the
-- prior-loan-penalty-history feature change already burned one retrain on
-- an untested hypothesis that turned out to hurt; validate this one first.
--
-- Exports one row per agent with whether they have any bad_state_3dpd_30d=1
-- loan under the NEW definition. Cross-reference against
-- data/blacklist_aug_20260804.csv / data/whitelist_aug_20260804.csv (same
-- approach as scripts/check_rollover_only_bad_vs_blacklist.py) to confirm
-- the new label's bad/good split aligns with the blacklist at least as
-- well as -- ideally much better than -- the old one before touching the
-- pipeline further.
-- ============================================================================

SELECT
    l.msisdn,
    MAX(l.bad_state_3dpd_30d) AS has_bad_loan_new_label,
    COUNT(*) AS n_label_eligible_loans
FROM hive.analytics.tmp_loan_label_assessment l
WHERE l.label_eligible_30d = 1
GROUP BY l.msisdn;
