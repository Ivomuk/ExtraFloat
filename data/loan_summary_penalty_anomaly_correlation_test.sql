-- ============================================================================
-- loan_summary_penalty_anomaly_correlation_test.sql -- ad-hoc diagnostic,
-- not part of the committed validation suite. Read loan_summary_query.txt
-- in full and traced its attribution logic: unlike borrower_history.txt's
-- total_recovered, this file's repayment_val_1M/3M/6M features are
-- computed straight from deduped raw repayments (repay_dedup), NOT the
-- vulnerable phonenumber-window attribution heuristic -- agent-level
-- features don't need per-loan attribution. The ONLY place the vulnerable
-- heuristic (attribution_timeline/attribution_filled/repay_attributed)
-- actually feeds an output feature is via loan_cure -> principal_cure_ts
-- -> penalty_events -> penalties_1M/3M/6M (synthetic 24h/48h penalty
-- EVENT COUNTS). For an ever-ANOMALY_OPEN loan, repayments landing after
-- the second loan's disbursement get misattributed away from the first
-- loan's cumulative_repaid, so the first loan can look perpetually
-- uncured (spurious penalty event) even if it was genuinely repaid on
-- time -- and the second loan can get artificial early credit (a
-- suppressed penalty event it should have gotten).
--
-- This quantifies that specific effect: for loans that DID eventually
-- reach CLOSED/SETTLED (ever_resolved -- authoritative ground truth that
-- the loan really was fully repaid, not a genuine default), what fraction
-- still get flagged as "not cured by 24h/48h" by the file's own
-- attribution-derived logic, cross-tabulated by ever_anomaly_open. A flag
-- on a loan that DID fully repay is, by construction, either a genuinely
-- late (but real) repayment or a misattribution artifact -- comparing the
-- rate between exposed and unexposed groups isolates how much of it is
-- the latter, the same differential-correlation approach used for
-- anomaly_open_b2b_correlation_test.sql.
--
-- Bounded to disbursement_ts <= 2026-06-02 so every loan has definitely
-- had 48h to mature (same maturity-buffer convention as
-- silently_stuck_loans_characterization.sql/anomaly_open_aging_only_test.sql).
-- Self-contained (raw tables), bounded to date_key <= 20260609.
-- ============================================================================

WITH disb_dedup AS (
SELECT disbursement_fid, phonenumber, disbursement_ts, disbursed_amount
FROM (
SELECT d.*,
ROW_NUMBER() OVER (PARTITION BY disbursement_fid ORDER BY inserted_ts DESC) rn
FROM (
SELECT
disbursement_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(disbursement_ts AS timestamp) AS disbursement_ts,
cast(disbursement_amount_ugx AS double) AS disbursed_amount,
inserted_ts
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND try_cast(disbursement_ts AS timestamp) IS NOT NULL
AND disbursement_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date_key <= 20260609
) d
)
WHERE rn = 1
),
disb_windows AS (
SELECT
disbursement_fid, phonenumber, disbursement_ts, disbursed_amount,
LEAD(disbursement_ts) OVER (PARTITION BY phonenumber ORDER BY disbursement_ts, disbursement_fid) AS next_disbursement_ts
FROM disb_dedup
),
repay_dedup AS (
SELECT phonenumber, repayment_ts, repayment_amount
FROM (
SELECT r.*,
ROW_NUMBER() OVER (PARTITION BY repayment_fid ORDER BY inserted_ts DESC) rn
FROM (
SELECT
repayment_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(repayment_ts AS timestamp) AS repayment_ts,
cast(repayment_amount_ugx AS double) AS repayment_amount,
inserted_ts
FROM analytics.momo_loan_book_tracker_repayments_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND try_cast(repayment_ts AS timestamp) IS NOT NULL
AND repayment_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date_key <= 20260609
) r
)
WHERE rn = 1
),
-- Mirrors loan_summary_query.txt's attribution_timeline/attribution_filled/
-- repay_attributed exactly (forward-fill LAST_VALUE form, logically
-- identical to the range-join form used elsewhere in this session).
attribution_timeline AS (
SELECT phonenumber, disbursement_ts AS event_ts, disbursed_amount AS raw_amount, 0 AS is_repayment, disbursement_fid
FROM disb_windows
UNION ALL
SELECT phonenumber, repayment_ts AS event_ts, repayment_amount AS raw_amount, 1 AS is_repayment, NULL AS disbursement_fid
FROM repay_dedup
),
attribution_filled AS (
SELECT
phonenumber, event_ts, raw_amount, is_repayment,
LAST_VALUE(disbursement_fid) IGNORE NULLS OVER (
PARTITION BY phonenumber ORDER BY event_ts, is_repayment, disbursement_fid
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
) AS disbursement_fid
FROM attribution_timeline
),
repay_attributed AS (
SELECT disbursement_fid, phonenumber, event_ts AS repayment_ts, ABS(raw_amount) AS repayment_amount
FROM attribution_filled
WHERE is_repayment = 1 AND disbursement_fid IS NOT NULL
),
loan_cure AS (
SELECT
w.disbursement_fid, w.phonenumber, w.disbursement_ts, w.disbursed_amount,
MIN(CASE WHEN cum.cumulative_repaid >= w.disbursed_amount THEN cum.repayment_ts END) AS principal_cure_ts
FROM disb_windows w
LEFT JOIN (
SELECT disbursement_fid, repayment_ts,
SUM(repayment_amount) OVER (PARTITION BY disbursement_fid ORDER BY repayment_ts) AS cumulative_repaid
FROM repay_attributed
) cum ON cum.disbursement_fid = w.disbursement_fid
WHERE w.disbursement_ts <= TIMESTAMP '2026-06-02 00:00:00.000'
GROUP BY w.disbursement_fid, w.phonenumber, w.disbursement_ts, w.disbursed_amount
),
loan_state_history AS (
SELECT disbursement_fid, loan_status, is_anomaly_open
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (PARTITION BY disbursement_fid, date_key ORDER BY inserted_ts DESC) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE ova = 'XTRAFLOAT-AGENT'
AND disbursement_fid IS NOT NULL
AND date_key <= 20260609
)
WHERE rn = 1
),
loan_flags AS (
SELECT
disbursement_fid,
MAX(CASE WHEN loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true THEN 1 ELSE 0 END) AS ever_anomaly_open,
MAX(CASE WHEN loan_status IN ('CLOSED', 'SETTLED') THEN 1 ELSE 0 END) AS ever_resolved
FROM loan_state_history
GROUP BY disbursement_fid
),
joined AS (
SELECT
lc.disbursement_fid,
lc.disbursement_ts,
lc.principal_cure_ts,
COALESCE(lf.ever_anomaly_open, 0) AS ever_anomaly_open,
COALESCE(lf.ever_resolved, 0) AS ever_resolved,
CASE WHEN lc.principal_cure_ts IS NULL
OR lc.principal_cure_ts > lc.disbursement_ts + INTERVAL '24' HOUR
THEN 1 ELSE 0 END AS derived_penalty_24h,
CASE WHEN lc.principal_cure_ts IS NULL
OR lc.principal_cure_ts > lc.disbursement_ts + INTERVAL '48' HOUR
THEN 1 ELSE 0 END AS derived_penalty_48h
FROM loan_cure lc
LEFT JOIN loan_flags lf ON lf.disbursement_fid = lc.disbursement_fid
)
SELECT
ever_anomaly_open,
COUNT(*) AS n_loans,
SUM(derived_penalty_24h) AS n_penalty_24h_overall,
ROUND(100.0 * SUM(derived_penalty_24h) / NULLIF(COUNT(*), 0), 2) AS pct_penalty_24h_overall,
SUM(derived_penalty_48h) AS n_penalty_48h_overall,
ROUND(100.0 * SUM(derived_penalty_48h) / NULLIF(COUNT(*), 0), 2) AS pct_penalty_48h_overall,
COUNT_IF(ever_resolved = 1) AS n_eventually_resolved,
COUNT_IF(ever_resolved = 1 AND derived_penalty_24h = 1) AS n_false_penalty_24h_among_resolved,
ROUND(100.0 * COUNT_IF(ever_resolved = 1 AND derived_penalty_24h = 1)
/ NULLIF(COUNT_IF(ever_resolved = 1), 0), 2) AS pct_false_penalty_24h_among_resolved,
COUNT_IF(ever_resolved = 1 AND derived_penalty_48h = 1) AS n_false_penalty_48h_among_resolved,
ROUND(100.0 * COUNT_IF(ever_resolved = 1 AND derived_penalty_48h = 1)
/ NULLIF(COUNT_IF(ever_resolved = 1), 0), 2) AS pct_false_penalty_48h_among_resolved
FROM joined
GROUP BY ever_anomaly_open
ORDER BY ever_anomaly_open;
-- RESULT:
-- INTERPRETATION: pct_false_penalty_24h/48h_among_resolved is the key
-- comparison -- these are loans PROVEN to have eventually fully repaid
-- (ever_resolved = 1, authoritative loan_status), so any penalty flag on
-- them is either a genuinely late-but-real repayment or a misattribution
-- artifact. If ever_anomaly_open = 1's rate is meaningfully higher than
-- ever_anomaly_open = 0's, that gap is direct, quantified evidence of the
-- misattribution mechanism inflating penalties_1M/3M/6M for exposed loans
-- beyond what their genuine repayment behavior would produce -- the
-- loan_summary_query.txt analog of the total_recovered mismatch already
-- fixed in borrower_history.txt. If the two rates are close, the effect is
-- small enough that documenting it as the same class of known limitation
-- as borrower_history.txt's cure-timing flags (no authoritative
-- replacement VALUE for exactly when a misattributed transaction should
-- have posted) is likely sufficient, without needing a code change here.
