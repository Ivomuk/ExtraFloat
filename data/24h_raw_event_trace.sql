-- ============================================================================
-- 24h_raw_event_trace.sql -- ad-hoc diagnostic, not part of the committed
-- validation suite. 24h_complement_diagnostic.sql's sample showed a
-- suspicious, consistent pattern: principal_cure_ts lands 24h + 2-3 minutes
-- after disbursement_ts (just PAST the 24h mark), yet recovery_24h already
-- equals (or slightly exceeds -- often by ~1%) disbursed_amount, as if the
-- crossing event were counted as within-24h when it wasn't. This traces the
-- raw transaction-grain events feeding cumulative_cure/hours_since_
-- disbursement directly from tbl_bh_classified (checkpoint 0's output) for
-- a handful of the offending loans, to see the actual event stream instead
-- of guessing at the mechanism.
-- ============================================================================

SELECT
requestid,
phonenumber,
event_ts,
amount,
txn_type,
date_diff('second',
(SELECT MIN(disbursement_ts) FROM analytics.tbl_bh_loan_final lf WHERE lf.requestid = c.requestid AND lf.phonenumber = c.phonenumber),
event_ts) / 3600.0 AS hours_since_disbursement_recomputed,
SUM(CASE WHEN txn_type IN ('recovery','reimbursement') THEN amount ELSE 0 END) OVER (
PARTITION BY requestid, phonenumber ORDER BY event_ts
) AS cumulative_cure_recomputed
FROM analytics.tbl_bh_classified c
WHERE requestid IN (
38485872800, 39328881491, 37673344680, 38485874433, 37952520161
)
ORDER BY requestid, event_ts;
-- RESULT:
-- INTERPRETATION: look at each requestid's full event list. Is there more
-- than one recovery/reimbursement row? Does any row have a NEGATIVE amount
-- (a reimbursement-type reversal/correction that classified's cure_amount
-- CASE would still add in, unlike repayments_daily which showed 0%
-- negative in Section B0 -- reimbursement events were never checked for
-- sign separately)? Does cumulative_cure_recomputed reach disbursed_amount
-- at an event where hours_since_disbursement_recomputed is <= 24, even
-- though the LATEST/crossing event sits just past 24? If so, an earlier
-- event already satisfied the threshold and something about how
-- principal_cure_ts vs recovery_24h use event ordering/ties is the actual
-- bug -- the raw rows here should make it visible directly.
