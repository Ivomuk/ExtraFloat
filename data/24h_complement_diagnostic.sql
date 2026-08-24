-- ============================================================================
-- 24h_complement_diagnostic.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. GATE 1b's bad_24h_complement_identity check
-- found 29,667 borrowers (out of 127,409 -- ~23%) where lifetime_on_time_
-- 24h_rate + lifetime_default_24h_rate != 1.0 in vw_bh_output. That's a
-- borrower-level AVG() over possibly many loans, so it can't show WHICH
-- loan(s) actually violate the per-loan complement invariant. This queries
-- tbl_bh_loan_final directly -- the per-loan checkpoint table still sitting
-- in your schema from the last vw_bh_output.sql run -- to find the actual
-- offending loan rows.
--
-- on_time_24h_flag is derived from principal_cure_ts (a timestamp
-- comparison: was the crossing event at/before disbursement_ts + 24h?).
-- default_24h_flag is derived from recovery_24h (a cashflow comparison: was
-- cumulative recovery AT the 24h checkpoint already >= disbursed_amount?).
-- These are two independently computed measures of the same underlying
-- threshold that should agree given monotonic non-decreasing cumulative
-- recovery (confirmed: 0% negative repayments, Section B0) -- but evidently
-- don't always.
-- ============================================================================

-- Step 1: pattern summary. Which combination of (on_time_24h_flag,
-- default_24h_flag) is producing the mismatch, and does it correlate with
-- how close principal_cure_ts sits to the exact 24h boundary?
SELECT
on_time_24h_flag,
default_24h_flag,
COUNT(*) AS n,
SUM(CASE WHEN principal_cure_ts IS NULL THEN 1 ELSE 0 END) AS n_principal_cure_ts_null,
approx_percentile(hours_to_principal_cure, 0.5) AS median_hours_to_principal_cure,
approx_percentile(ABS(hours_to_principal_cure - 24.0), 0.5) AS median_abs_hours_from_24h_boundary,
MIN(hours_to_principal_cure) AS min_hours_to_principal_cure,
MAX(hours_to_principal_cure) AS max_hours_to_principal_cure
FROM analytics.tbl_bh_loan_final
WHERE on_time_24h_flag IS NOT NULL
AND default_24h_flag IS NOT NULL
AND on_time_24h_flag + default_24h_flag != 1
GROUP BY on_time_24h_flag, default_24h_flag;
-- RESULT:

-- Step 2: a sample of the actual offending rows, ordered by closeness to
-- the 24h boundary -- if the mismatch is a boundary/precision issue, the
-- worst offenders should cluster very close to hours_to_principal_cure = 24.
SELECT
requestid, phonenumber, disbursement_ts, disbursed_amount,
principal_cure_ts, hours_to_principal_cure, recovery_24h,
on_time_24h_flag, default_24h_flag
FROM analytics.tbl_bh_loan_final
WHERE on_time_24h_flag IS NOT NULL
AND default_24h_flag IS NOT NULL
AND on_time_24h_flag + default_24h_flag != 1
ORDER BY ABS(COALESCE(hours_to_principal_cure, 999) - 24.0) ASC
LIMIT 20;
-- RESULT:
-- INTERPRETATION: if Step 1 shows one dominant (on_time,default) combination
-- (e.g. both 0, meaning "not on time" AND "not default" simultaneously) and
-- median_abs_hours_from_24h_boundary is small (a fraction of an hour), this
-- is a boundary-precision mismatch between the two independently-derived
-- thresholds. If principal_cure_ts is NULL for the offending rows instead,
-- the issue is in how recovery_24h/disbursed_amount handle unresolved
-- (never-cured) loans differently between the two flags. Step 2's raw rows
-- should make the actual mechanism visible either way.
