-- ============================================================================
-- b3_eligibility_funnel.sql -- ad-hoc diagnostic, not part of the committed
-- validation suite. B3's `eligible` CTE (borrower_history_validation_
-- queries.sql lines 842-863) returned 0 rows on the latest run. This peels
-- back each of its filters one at a time to find which one zeroes out the
-- population -- run GATE 0 first (this reads vw_bh_disb_dedup/
-- vw_bh_loan_state_snapshot, same as the production validation run).
-- ============================================================================

-- Step 0: what loan_status values actually exist? B3 filters on the exact
-- string 'CLOSED' -- if the real value is differently cased or spelled,
-- this filter alone zeroes everything out downstream.
SELECT loan_status, COUNT(*) AS n
FROM :validation_schema.vw_bh_loan_state_snapshot
GROUP BY loan_status
ORDER BY n DESC;
-- RESULT:

-- Step 1: single-loan borrowers (exactly one deduped disbursement in-window)
SELECT COUNT(*) AS n_single_loan_borrowers
FROM (
SELECT phonenumber
FROM :validation_schema.vw_bh_disb_dedup
GROUP BY phonenumber
HAVING COUNT(*) = 1
);
-- RESULT:

-- Step 2: of those single-loan borrowers' one disbursement, how many have a
-- matching loan_state_snapshot row at all (regardless of status)?
WITH single_loan_borrowers AS (
SELECT phonenumber
FROM :validation_schema.vw_bh_disb_dedup
GROUP BY phonenumber
HAVING COUNT(*) = 1
)
SELECT COUNT(*) AS n_with_loan_state_match
FROM :validation_schema.vw_bh_disb_dedup d
JOIN single_loan_borrowers slb ON slb.phonenumber = d.phonenumber
JOIN :validation_schema.vw_bh_loan_state_snapshot ls ON ls.disbursement_fid = d.disbursement_fid;
-- RESULT:

-- Step 3: of those, how many are loan_status = 'CLOSED'?
WITH single_loan_borrowers AS (
SELECT phonenumber
FROM :validation_schema.vw_bh_disb_dedup
GROUP BY phonenumber
HAVING COUNT(*) = 1
)
SELECT COUNT(*) AS n_closed
FROM :validation_schema.vw_bh_disb_dedup d
JOIN single_loan_borrowers slb ON slb.phonenumber = d.phonenumber
JOIN :validation_schema.vw_bh_loan_state_snapshot ls ON ls.disbursement_fid = d.disbursement_fid
WHERE ls.loan_status = 'CLOSED';
-- RESULT:

-- Step 4: of those CLOSED, how many have interest_and_penalty_ugx > 0?
WITH single_loan_borrowers AS (
SELECT phonenumber
FROM :validation_schema.vw_bh_disb_dedup
GROUP BY phonenumber
HAVING COUNT(*) = 1
)
SELECT COUNT(*) AS n_closed_and_charged
FROM :validation_schema.vw_bh_disb_dedup d
JOIN single_loan_borrowers slb ON slb.phonenumber = d.phonenumber
JOIN :validation_schema.vw_bh_loan_state_snapshot ls ON ls.disbursement_fid = d.disbursement_fid
WHERE ls.loan_status = 'CLOSED'
AND ls.interest_and_penalty_ugx > 0;
-- RESULT:

-- Step 5: of those, how many pass the final has_pre_window_history filter
-- (the full B3 eligibility set)?
WITH single_loan_borrowers AS (
SELECT phonenumber
FROM :validation_schema.vw_bh_disb_dedup
GROUP BY phonenumber
HAVING COUNT(*) = 1
)
SELECT COUNT(*) AS n_fully_eligible
FROM :validation_schema.vw_bh_disb_dedup d
JOIN single_loan_borrowers slb ON slb.phonenumber = d.phonenumber
JOIN :validation_schema.vw_bh_loan_state_snapshot ls ON ls.disbursement_fid = d.disbursement_fid
WHERE ls.loan_status = 'CLOSED'
AND ls.interest_and_penalty_ugx > 0
AND COALESCE(ls.has_pre_window_history, false) = false;
-- RESULT:
-- INTERPRETATION: whichever step count drops to 0 first is the filter
-- responsible. Step 0's distinct loan_status list is the most likely
-- culprit (a case/spelling mismatch against the literal 'CLOSED') --
-- check that first.
