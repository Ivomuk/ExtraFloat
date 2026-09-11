-- ============================================================================
-- rollover_only_bad_vs_blacklist_export.sql
-- ============================================================================
-- Direct test of whether bad_state_3dpd_30d's "OR ANOMALY_OPEN" clause
-- should stay in the label, using the business's own blacklist ground
-- truth rather than inference alone.
--
-- dpd_bucket_split_by_anomaly_open.sql found: within loans where
-- max_dpd_within_30d <= 2 (i.e. days_aging could NOT have exceeded 3),
-- rollover_observed_30d=1 predicts bad_state_3dpd_30d=1 with ZERO
-- exceptions across ~3.4M loans -- 689,283 + 48,443 = 737,726 of the
-- 1,072,897 total "bad" loans (68.8%) are bad ONLY via the ANOMALY_OPEN
-- branch, with no elevated repayment lateness at all. Combined with this
-- session's earlier finding that ever_anomaly_open is actually MORE common
-- among whitelisted agents (82.3%) than Defaulters (77.1%) -- a loan-volume
-- proxy, not a badness signal -- there's a real question of whether "OR
-- ANOMALY_OPEN" belongs in the label at all.
--
-- This exports one row per agent (msisdn) with:
--   - has_genuinely_aging_bad_loan: at least one bad_state_3dpd_30d=1 loan
--     where max_days_aging_30d > 3 was ALSO independently true (i.e. would
--     still be bad under a stricter days_aging-only definition)
--   - has_rollover_only_bad_loan: at least one bad_state_3dpd_30d=1 loan
--     that is bad ONLY via rollover_observed_30d=1 (max_days_aging_30d IS
--     NULL or <= 3) -- would NOT be bad under a stricter definition
--
-- Export this to CSV and cross-reference against
-- data/blacklist_aug_20260804.csv / data/whitelist_aug_20260804.csv (same
-- MSISDN normalization as pd_model.postprocessing.whitelist_eval) to see
-- whether agents whose ONLY bad loans are rollover-only show up on the
-- blacklist at a similar rate to genuinely-aging-bad agents, or not.
-- If rollover-only-bad agents are NOT disproportionately blacklisted
-- (i.e. behave more like the whitelist population), that's direct
-- evidence the OR ANOMALY_OPEN clause is adding false positives to the
-- label rather than catching real Defaulters.
-- ============================================================================

SELECT
    l.msisdn,

    MAX(
        CASE
            WHEN l.bad_state_3dpd_30d = 1
             AND l.max_days_aging_30d > 3
            THEN 1 ELSE 0
        END
    ) AS has_genuinely_aging_bad_loan,

    MAX(
        CASE
            WHEN l.bad_state_3dpd_30d = 1
             AND COALESCE(l.max_days_aging_30d, 0) <= 3
            THEN 1 ELSE 0
        END
    ) AS has_rollover_only_bad_loan,

    COUNT(*) AS n_label_eligible_loans,
    SUM(l.bad_state_3dpd_30d) AS n_bad_loans

FROM hive.analytics.tmp_loan_label_assessment l
WHERE l.label_eligible_30d = 1
GROUP BY l.msisdn;
