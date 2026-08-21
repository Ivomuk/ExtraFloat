-- ============================================================================
-- expected_charge_check.sql -- ad-hoc diagnostic, not part of the committed
-- validation suite. interest_and_penalty_ugx is confirmed 100% NULL for
-- both SETTLED and CLOSED loans (data/interest_and_penalty_check.sql) --
-- likely a currently-outstanding balance that naturally nulls out once a
-- loan is fully repaid, not a lifetime cumulative charge. vw_bh_loan_state_
-- snapshot also exposes expected_total_charge_ugx/charge_variance_ugx/
-- charge_variance_pct/is_charge_anomaly, which sound like better
-- candidates for "was this loan ever charged interest/penalty" on a closed
-- loan. This checks whether they're actually populated (non-null) for
-- SETTLED/CLOSED loans, before rewriting Section A/B3 to use one of them.
-- ============================================================================

SELECT
loan_status,
COUNT(*) AS n,
SUM(CASE WHEN expected_total_charge_ugx > 0 THEN 1 ELSE 0 END) AS n_expected_charge_positive,
SUM(CASE WHEN expected_total_charge_ugx = 0 THEN 1 ELSE 0 END) AS n_expected_charge_zero,
SUM(CASE WHEN expected_total_charge_ugx IS NULL THEN 1 ELSE 0 END) AS n_expected_charge_null,
approx_percentile(expected_total_charge_ugx, 0.5) AS median_expected_total_charge_ugx,
SUM(CASE WHEN charge_variance_ugx IS NULL THEN 1 ELSE 0 END) AS n_charge_variance_null,
approx_percentile(charge_variance_ugx, 0.5) AS median_charge_variance_ugx,
approx_percentile(charge_variance_pct, 0.5) AS median_charge_variance_pct,
SUM(CASE WHEN is_charge_anomaly THEN 1 ELSE 0 END) AS n_is_charge_anomaly
FROM :validation_schema.vw_bh_loan_state_snapshot
WHERE loan_status IN ('SETTLED', 'CLOSED')
GROUP BY loan_status;
-- RESULT:
-- INTERPRETATION: if n_expected_charge_positive is a meaningful nonzero
-- count (not ~0/all-null like interest_and_penalty_ugx was), expected_
-- total_charge_ugx > 0 is a viable replacement filter for Section A/B3's
-- "closed and charged" population. If it's ALSO entirely null/zero, none of
-- loan_state_daily's charge-related fields are usable this way for this
-- snapshot, and Section A/B3's whole "isolate charged loans via loan_state"
-- approach needs to be rethought with the table owner rather than patched
-- further by literal-swapping.
