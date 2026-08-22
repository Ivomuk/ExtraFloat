-- ============================================================================
-- loan_uid_aware_reconciliation_test.sql -- ad-hoc diagnostic, not part of
-- the committed validation suite, not a change to borrower_history.txt.
-- Tests whether bounding repayment-attribution windows by "next disbursement
-- with a DIFFERENT loan_uid" (instead of "next disbursement, period")
-- meaningfully improves B2b's reconciliation match rate (currently 68.6%
-- against lifetime_repaid_ugx), before deciding whether it's worth porting
-- into borrower_history.txt's actual repay_attributed logic.
--
-- Mechanism: loan_uid lives on loan_state_daily, not disbursements_daily,
-- and ~3.9% of disbursement_fids (the "hidden" same-day-merged-away ones)
-- have NO loan_state_daily row at all -- so loan_uid is forward-filled per
-- phonenumber (LAST_VALUE ... IGNORE NULLS), same technique already used in
-- borrower_history.txt's repay_attributed rewrite. A "gaps and islands"
-- grouping then finds, for each disbursement, the timestamp of the next
-- disbursement whose (filled) loan_uid actually differs -- that becomes the
-- new window boundary instead of vw_bh_disb_windows' plain next_
-- disbursement_ts.
-- ============================================================================

WITH loan_state_with_uid AS (
-- Mirrors vw_bh_loan_state_snapshot's own dedup logic exactly (same
-- filters, same double ROW_NUMBER pass), just also keeping loan_uid, which
-- that view doesn't expose.
SELECT disbursement_fid, loan_uid
FROM (
SELECT lsld.*,
ROW_NUMBER() OVER (
PARTITION BY disbursement_fid
ORDER BY date_key DESC
) rn2
FROM (
SELECT *
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (
PARTITION BY disbursement_fid, date_key
ORDER BY inserted_ts DESC
) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE ova = 'XTRAFLOAT-AGENT'
AND disbursement_fid IS NOT NULL
AND date_key <= 20260731
AND inserted_ts <= TIMESTAMP '2026-08-20 00:00:00.000'
)
WHERE rn = 1
) lsld
)
WHERE rn2 = 1
),
disb_with_raw_uid AS (
SELECT
d.disbursement_fid,
d.phonenumber,
d.disbursement_ts,
d.disbursed_amount,
lu.loan_uid
FROM :validation_schema.vw_bh_disb_dedup d
LEFT JOIN loan_state_with_uid lu ON lu.disbursement_fid = d.disbursement_fid
),
disb_with_filled_uid AS (
SELECT
*,
LAST_VALUE(loan_uid) IGNORE NULLS OVER (
PARTITION BY phonenumber
ORDER BY disbursement_ts, disbursement_fid
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
) AS filled_loan_uid
FROM disb_with_raw_uid
),
grouped AS (
SELECT
*,
SUM(CASE WHEN is_new_uid THEN 1 ELSE 0 END) OVER (
PARTITION BY phonenumber
ORDER BY disbursement_ts, disbursement_fid
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
) AS loan_uid_group
FROM (
SELECT
*,
CASE
WHEN LAG(filled_loan_uid) OVER (
PARTITION BY phonenumber ORDER BY disbursement_ts, disbursement_fid
) IS DISTINCT FROM filled_loan_uid THEN true
ELSE false
END AS is_new_uid
FROM disb_with_filled_uid
)
),
per_group AS (
SELECT phonenumber, loan_uid_group, MIN(disbursement_ts) AS group_start_ts
FROM grouped
GROUP BY phonenumber, loan_uid_group
),
per_group_with_next AS (
SELECT
phonenumber, loan_uid_group, group_start_ts,
LEAD(group_start_ts) OVER (PARTITION BY phonenumber ORDER BY loan_uid_group) AS next_group_start_ts
FROM per_group
),
disb_windows_uid_aware AS (
SELECT
g.disbursement_fid,
g.phonenumber,
g.disbursement_ts,
g.disbursed_amount,
pgn.next_group_start_ts AS next_different_loan_uid_ts
FROM grouped g
JOIN per_group_with_next pgn
ON pgn.phonenumber = g.phonenumber AND pgn.loan_uid_group = g.loan_uid_group
),
attributed AS (
SELECT w.disbursement_fid, r.repayment_amount
FROM :validation_schema.vw_bh_repay_dedup r
JOIN disb_windows_uid_aware w
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_different_loan_uid_ts IS NULL OR r.repayment_ts < w.next_different_loan_uid_ts)
WHERE w.disbursement_fid IN (SELECT disbursement_fid FROM :validation_schema.vw_bh_surviving_loans)
),
per_loan_attributed AS (
SELECT
w.disbursement_fid,
w.disbursed_amount,
COALESCE(SUM(ABS(a.repayment_amount)), 0) AS attributed_repaid_abs
FROM :validation_schema.vw_bh_surviving_loans w
LEFT JOIN attributed a ON a.disbursement_fid = w.disbursement_fid
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
) lsl
ON lsl.disbursement_fid = pla.disbursement_fid
)
SELECT
COUNT(*) AS n_total,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0 THEN 1 ELSE 0 END) AS n_matched,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_exact_match_loan_uid_aware,
ROUND(100.0 * SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END)
/ NULLIF(SUM(CASE WHEN state_only = 0 AND disbursement_only = 0 THEN 1 ELSE 0 END), 0), 2) AS pct_exact_match_loan_uid_aware
FROM joined;
-- RESULT:
-- INTERPRETATION: compare pct_exact_match_loan_uid_aware against B2b's
-- original 68.6%. A large improvement (e.g. into the 90s) confirms the
-- loan_uid-aware window boundary is the right fix and worth porting into
-- borrower_history.txt's repay_attributed. A small or no improvement means
-- either the mechanism is more complex than this model captures, or the
-- fix needs to also address the "ripple" case differently (e.g. genuinely
-- separate loans downstream of a long-lived merged loan_uid that this
-- grouping doesn't fully capture).
