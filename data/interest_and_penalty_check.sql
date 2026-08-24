-- ============================================================================
-- interest_and_penalty_check.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. Section A / B3 still returned 0 rows after
-- fixing loan_status to include SETTLED, so the OTHER filter in both
-- queries -- interest_and_penalty_ugx > 0 -- is the next suspect. This
-- checks the actual distribution of interest_and_penalty_ugx among
-- SETTLED/CLOSED loans: if it's ~0 (or negative/NULL) for nearly all of
-- them, that filter alone zeroes out both sections regardless of the
-- loan_status fix.
-- ============================================================================

SELECT
loan_status,
COUNT(*) AS n,
SUM(CASE WHEN interest_and_penalty_ugx > 0 THEN 1 ELSE 0 END) AS n_positive,
SUM(CASE WHEN interest_and_penalty_ugx = 0 THEN 1 ELSE 0 END) AS n_zero,
SUM(CASE WHEN interest_and_penalty_ugx < 0 THEN 1 ELSE 0 END) AS n_negative,
SUM(CASE WHEN interest_and_penalty_ugx IS NULL THEN 1 ELSE 0 END) AS n_null,
approx_percentile(interest_and_penalty_ugx, 0.5) AS median_interest_and_penalty_ugx,
approx_percentile(interest_and_penalty_ugx, 0.95) AS p95_interest_and_penalty_ugx,
MAX(interest_and_penalty_ugx) AS max_interest_and_penalty_ugx
FROM :validation_schema.vw_bh_loan_state_snapshot
WHERE loan_status IN ('SETTLED', 'CLOSED')
GROUP BY loan_status;
-- RESULT:
-- INTERPRETATION: if n_positive is ~0 for both statuses, interest_and_
-- penalty_ugx > 0 is the actual blocking filter, not loan_status. Compare
-- median/p95/max to see whether SOME loans do carry a charge (a real but
-- narrow population -- fine, just report a smaller n_closed_charged_loans)
-- or whether the field is essentially always 0/null for this snapshot (a
-- different problem: either no loan has been charged interest/penalty yet,
-- or the field means something other than expected).
