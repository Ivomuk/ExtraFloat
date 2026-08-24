-- ============================================================================
-- charge_fields_all_status_check.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. interest_and_penalty_ugx AND expected_total_
-- charge_ugx/charge_variance_ugx/charge_variance_pct are all confirmed
-- 100% NULL for SETTLED and CLOSED loans. This checks whether they're
-- populated for OPEN/ANOMALY_OPEN loans instead -- if yes, these are
-- "live, currently accruing" fields the pipeline clears once a loan
-- settles (a real, explainable behavior, just not usable for Section A/B3's
-- retrospective "closed and charged" check). If no, none of loan_state_
-- daily's charge fields are populated for ANY status at this snapshot, a
-- bigger data-population gap.
-- ============================================================================

SELECT
loan_status,
COUNT(*) AS n,
SUM(CASE WHEN interest_and_penalty_ugx IS NOT NULL THEN 1 ELSE 0 END) AS n_interest_and_penalty_populated,
SUM(CASE WHEN expected_total_charge_ugx IS NOT NULL THEN 1 ELSE 0 END) AS n_expected_charge_populated,
SUM(CASE WHEN charge_variance_ugx IS NOT NULL THEN 1 ELSE 0 END) AS n_charge_variance_populated
FROM :validation_schema.vw_bh_loan_state_snapshot
GROUP BY loan_status
ORDER BY n DESC;
-- RESULT:
-- INTERPRETATION: if OPEN/ANOMALY_OPEN show nonzero n_*_populated counts
-- while SETTLED/CLOSED show 0 (already confirmed), the charge fields are
-- live-balance fields cleared on settlement -- Section A/B3 cannot use
-- loan_state_daily's charge columns for a closed-loan population at all,
-- regardless of which literal/column is chosen, and need a different
-- approach (e.g. deriving "was this loan charged" from cure-timing the same
-- way borrower_history.txt/loan_summary_query.txt already do, rather than
-- from an authoritative loan_state total). If OPEN also shows 0, the gap is
-- broader still and worth raising with the table owner directly.
