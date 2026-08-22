-- ============================================================================
-- ever_anomalous_exclusion_test.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. borrower_history.txt's loan_state_anomalies
-- exclusion checks only the LATEST loan_state_daily snapshot per
-- disbursement_fid (loan_state_snapshot picks MAX(date_key)). Our traced
-- example (disbursement_fid 39631605547) was ANOMALY_OPEN for ~2 weeks but
-- resolved to CLOSED by its latest snapshot -- so it currently SURVIVES
-- the exclusion and sits in vw_bh_surviving_loans, still carrying the
-- reconciliation chaos from its anomaly period (it was one of B2b's 10
-- sampled mismatches: attributed=0 vs lifetime_repaid_ugx=5,250,000).
--
-- Tests a simpler candidate fix: exclude any disbursement_fid that was
-- EVER flagged ANOMALY_OPEN at ANY point in its full loan_state_daily
-- history, not just its latest snapshot -- before deciding whether it's
-- worth changing loan_state_anomalies' definition in borrower_history.txt/
-- borrower_history_validation_queries.sql.
-- ============================================================================

WITH ever_anomalous_fids AS (
SELECT DISTINCT disbursement_fid
FROM analytics.momo_loan_book_tracker_loan_state_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND disbursement_fid IS NOT NULL
AND date_key <= 20260731
AND inserted_ts <= TIMESTAMP '2026-08-20 00:00:00.000'
AND (loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true)
),
attributed AS (
SELECT w.disbursement_fid, r.repayment_amount
FROM :validation_schema.vw_bh_repay_dedup r
JOIN :validation_schema.vw_bh_disb_windows w
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
WHERE w.disbursement_fid IN (
SELECT disbursement_fid FROM :validation_schema.vw_bh_surviving_loans
WHERE disbursement_fid NOT IN (SELECT disbursement_fid FROM ever_anomalous_fids)
)
),
per_loan_attributed AS (
SELECT
w.disbursement_fid,
w.disbursed_amount,
COALESCE(SUM(ABS(a.repayment_amount)), 0) AS attributed_repaid_abs
FROM :validation_schema.vw_bh_surviving_loans w
LEFT JOIN attributed a ON a.disbursement_fid = w.disbursement_fid
WHERE w.disbursement_fid NOT IN (SELECT disbursement_fid FROM ever_anomalous_fids)
GROUP BY w.disbursement_fid, w.disbursed_amount
),
joined AS (
SELECT
pla.disbursement_fid,
pla.attributed_repaid_abs,
lsl.lifetime_repaid_ugx,
CASE WHEN pla.disbursement_fid IS NULL THEN 1 ELSE 0 END AS state_only,
CASE WHEN lsl.disbursement_fid IS NULL THEN 1 ELSE 0 END AS disbursement_only
FROM per_loan_attributed pla
FULL OUTER JOIN (
SELECT lsl.* FROM :validation_schema.vw_bh_loan_state_snapshot lsl
JOIN :validation_schema.vw_bh_surviving_loans sl ON sl.disbursement_fid = lsl.disbursement_fid
WHERE sl.disbursement_fid NOT IN (SELECT disbursement_fid FROM ever_anomalous_fids)
) lsl
ON lsl.disbursement_fid = pla.disbursement_fid
)
SELECT
(SELECT COUNT(*) FROM ever_anomalous_fids) AS total_ever_anomalous_fids,
(SELECT COUNT(*) FROM ever_anomalous_fids ea JOIN :validation_schema.vw_bh_surviving_loans sl ON sl.disbursement_fid = ea.disbursement_fid) AS ever_anomalous_fids_currently_surviving,
COUNT(*) AS n_total_after_exclusion,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0 THEN 1 ELSE 0 END) AS n_matched_after_exclusion,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_exact_match_after_exclusion,
ROUND(100.0 * SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END)
/ NULLIF(SUM(CASE WHEN state_only = 0 AND disbursement_only = 0 THEN 1 ELSE 0 END), 0), 2) AS pct_exact_match_after_exclusion
FROM joined;
-- RESULT:
-- INTERPRETATION: ever_anomalous_fids_currently_surviving is how many
-- loans this new filter would additionally remove from vw_bh_surviving_
-- loans beyond the current latest-snapshot-only exclusion. Compare pct_
-- exact_match_after_exclusion against the original 68.6% -- if it jumps
-- substantially (more than the +2-2.4% seen from the loan_uid-aware and
-- gross-vs-net tests), this simple filter change is a strong, low-risk
-- candidate: it only tightens an EXISTING exclusion rule (ANOMALY_OPEN =
-- system error, already the accepted rationale for excluding these loans
-- entirely) rather than introducing new attribution logic.
