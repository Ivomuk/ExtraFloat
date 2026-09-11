-- ============================================================================
-- penalty_unpaid_vs_penalty_assessed_rate.sql
-- ============================================================================
-- Follow-up to label_vs_penalty_comparison.sql's finding: ever_penalty_2_due
-- ("was a 48h late penalty ever ASSESSED") fires for 92.3% of ALL loans --
-- both bad_state_3dpd_30d=0 (90.5% of them) and bad_state_3dpd_30d=1 (100%
-- of them) -- meaning it barely discriminates and likely explains why
-- prior_late_fee_1_count/prior_late_fee_2_count (built on "ever assessed")
-- made cal_pd's AUC against the blacklist WORSE, not better: near-universal,
-- low-information features add noise, not signal.
--
-- Hypothesis this tests: "assessed" is the wrong cut -- almost every loan
-- gets a penalty assessed and pays it off within a day or two (normal
-- 24-hour-product friction, not risk). What should actually be rare and
-- discriminating is "assessed AND NEVER PAID OFF" -- genuine non-payment,
-- using penalty_2_outstanding_ugx at the loan's LATEST/closing snapshot
-- (not just any day it was ever nonzero, which would trivially be true
-- for one calendar day after every assessment).
--
-- Mirrors label_vs_penalty_comparison.sql's dedup pattern exactly.
-- ============================================================================

WITH target_penalty_ranked AS (
    SELECT
        disbursement_fid,
        date_key,
        penalty_2_due,
        penalty_2_owed_ugx,
        penalty_2_outstanding_ugx,
        ROW_NUMBER() OVER (
            PARTITION BY disbursement_fid
            ORDER BY date_key DESC, inserted_ts DESC
        ) AS latest_rn
    FROM analytics.momo_loan_book_tracker_xtrafloat_loan_state_daily
    WHERE ova = 'XTRAFLOAT-AGENT'
      AND date_key >= 20260101
),

target_penalty_latest AS (
    -- Latest known state per loan -- if penalty_2_outstanding_ugx is still
    -- > 0 as of the last observed day, that penalty was never paid off.
    SELECT
        disbursement_fid,
        COALESCE(penalty_2_due, false) AS ever_penalty_2_due_latest,
        COALESCE(penalty_2_outstanding_ugx, 0) > 0 AS penalty_2_unpaid_at_latest
    FROM target_penalty_ranked
    WHERE latest_rn = 1
)

SELECT
    l.bad_state_3dpd_30d,
    tp.penalty_2_unpaid_at_latest,
    COUNT(*) AS n
FROM hive.analytics.tmp_loan_label_assessment l
LEFT JOIN target_penalty_latest tp
    ON tp.disbursement_fid = l.disbursement_fid
WHERE l.label_eligible_30d = 1
GROUP BY 1, 2
ORDER BY 1, 2;
