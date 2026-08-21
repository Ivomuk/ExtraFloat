-- ============================================================================
-- settled_vs_closed_check.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. B3/Section A in borrower_history_validation_
-- queries.sql filter loan_status = 'CLOSED' expecting "fully repaid and
-- done", but among 9,013 single-loan borrowers with a loan_state match,
-- ZERO hit 'CLOSED' -- all must be SETTLED/ANOMALY_OPEN/OPEN/NEVER_BORROWED
-- instead (loan_status distinct-value dump: SETTLED=3,457,110,
-- CLOSED=727,866, ANOMALY_OPEN=141,331, OPEN=41,009, NEVER_BORROWED=47).
-- SETTLED outnumbers CLOSED ~4.7x, suggesting SETTLED is the "fully repaid"
-- terminal status and CLOSED means something else (write-off, manual
-- closure, etc.). This checks that hypothesis directly: for each status,
-- what fraction of loans actually reached full principal repayment
-- (lifetime_repaid_ugx / lifetime_disbursed_ugx >= ~1)?
-- ============================================================================

SELECT
loan_status,
COUNT(*) AS n,
approx_percentile(lifetime_repaid_ugx / NULLIF(lifetime_disbursed_ugx, 0), 0.5) AS median_repaid_over_principal,
approx_percentile(lifetime_gross_repaid_ugx / NULLIF(lifetime_disbursed_ugx, 0), 0.5) AS median_gross_repaid_over_principal,
SUM(CASE WHEN lifetime_repaid_ugx >= lifetime_disbursed_ugx THEN 1 ELSE 0 END) AS n_repaid_reaches_principal_raw,
SUM(CASE WHEN lifetime_gross_repaid_ugx >= lifetime_disbursed_ugx THEN 1 ELSE 0 END) AS n_gross_repaid_reaches_principal,
ROUND(100.0 * SUM(CASE WHEN lifetime_gross_repaid_ugx >= lifetime_disbursed_ugx THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS pct_reaching_principal
FROM :validation_schema.vw_bh_loan_state_snapshot
GROUP BY loan_status
ORDER BY n DESC;
-- RESULT:
-- INTERPRETATION: if SETTLED shows pct_reaching_principal close to 100% and
-- CLOSED shows something much lower, that confirms SETTLED (not CLOSED) is
-- the "fully repaid" terminal status, and B3/Section A's literal should be
-- changed. If both look similar, the distinction is something else entirely
-- (ask the table owner) and this fix is wrong.
