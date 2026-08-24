-- ============================================================================
-- b2b_gross_vs_net_test.sql -- ad-hoc diagnostic, not part of the committed
-- validation suite. loan_state_sharing_trace.sql's phonenumber 256772
-- example (disbursement_fid 39604855038, no same-day merge) showed
-- attributed_repaid_abs=7,500,000 far from lifetime_repaid_ugx=750,000, but
-- very close to lifetime_repaid_ugx + lifetime_gross_repaid_ugx =
-- 750,000 + 6,750,000 = 7,500,000 exactly. Hypothesis: our raw-repayment-
-- sum attribution heuristic naturally measures something closer to a
-- gross/account-level total than to lifetime_repaid_ugx's net,
-- principal-allocated-to-this-loan figure -- meaning B2b may have been
-- reconciling against the wrong column. This recomputes B2b's exact-match
-- rate against lifetime_gross_repaid_ugx instead, to test directly.
-- ============================================================================

WITH attributed AS (
SELECT w.disbursement_fid, r.repayment_amount
FROM :validation_schema.vw_bh_repay_dedup r
JOIN :validation_schema.vw_bh_disb_windows w
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
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
pla.disbursed_amount,
pla.attributed_repaid_abs,
lsl.lifetime_repaid_ugx,
lsl.lifetime_gross_repaid_ugx,
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
-- ORIGINAL B2b comparison: attributed_repaid_abs vs lifetime_repaid_ugx (net)
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_exact_match_vs_net,
-- NEW comparison: attributed_repaid_abs vs lifetime_gross_repaid_ugx alone
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - lifetime_gross_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_exact_match_vs_gross,
-- NEW comparison: attributed_repaid_abs vs (net + gross) -- matches the
-- 256772 example exactly (750,000 + 6,750,000 = 7,500,000)
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - (lifetime_repaid_ugx + lifetime_gross_repaid_ugx)) <= 1 THEN 1 ELSE 0 END) AS n_exact_match_vs_net_plus_gross,
ROUND(100.0 * SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END)
/ NULLIF(SUM(CASE WHEN state_only = 0 AND disbursement_only = 0 THEN 1 ELSE 0 END), 0), 2) AS pct_match_vs_net,
ROUND(100.0 * SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - lifetime_gross_repaid_ugx) <= 1 THEN 1 ELSE 0 END)
/ NULLIF(SUM(CASE WHEN state_only = 0 AND disbursement_only = 0 THEN 1 ELSE 0 END), 0), 2) AS pct_match_vs_gross,
ROUND(100.0 * SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - (lifetime_repaid_ugx + lifetime_gross_repaid_ugx)) <= 1 THEN 1 ELSE 0 END)
/ NULLIF(SUM(CASE WHEN state_only = 0 AND disbursement_only = 0 THEN 1 ELSE 0 END), 0), 2) AS pct_match_vs_net_plus_gross
FROM joined;
-- RESULT:
-- INTERPRETATION: if pct_match_vs_gross or pct_match_vs_net_plus_gross is
-- dramatically higher than pct_match_vs_net (68.6%, the original B2b
-- number), that confirms the attribution heuristic's raw repayment sum
-- naturally measures a gross/account-level total, not the net,
-- principal-allocated-to-this-loan figure lifetime_repaid_ugx represents --
-- B2b has been comparing against the wrong column, and the real
-- reconciliation rate is much healthier than 68.6% once compared correctly.
-- If none of the three are much better, the 256772 example was a
-- coincidence and the mismatch has a different or mixed cause (e.g. the
-- ANOMALY_OPEN ripple-effect hypothesis from the same-day-merge case).
