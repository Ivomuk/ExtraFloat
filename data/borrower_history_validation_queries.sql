-- ============================================================================
-- borrower_history_validation_queries.sql
-- ============================================================================
-- Diagnostics for the borrower_history.txt rewrite (xtrafloat_daily_trans ->
-- momo_loan_book_tracker_* tables). None of these have been run -- this repo
-- has no live warehouse connection, so every query here is prepared, not
-- executed. Run each section in Athena/Trino, read the interpretation note,
-- and treat the rewrite as unverified until this file's findings are filled
-- in (see the "RESULT:" placeholders).
--
-- Set these to match the borrower_history.txt run being validated:
--   :snapshot_dt      -- e.g. 20260531
--   :as_of_load_ts    -- e.g. TIMESTAMP '2026-06-01 00:00:00.000'
-- ============================================================================
--
-- ============================================================================
-- ACCEPTANCE THRESHOLDS (proposed defaults -- ratify with the team before
-- using any of these as a go/no-go gate; they are starting points, not
-- something I can unilaterally declare correct)
-- ============================================================================
--   Attributed repayment coverage        (B1: attributed_repayments /
--                                          total_repayments)             >= 99%
--   Boundary-sensitive repayment rate    (B1: near_next_disbursement_
--                                          boundary / attributed_
--                                          repayments)                   <= 1%
--   Rapid-reborrow rate                  (B1: rapid_reborrow_disbursements
--                                          / total_disbursements -- proxy
--                                          for "overlapping/near-
--                                          simultaneous loans", since there
--                                          is no direct concurrency signal
--                                          without an exact loan key)     <= 2%
--   Reconciliation match rate            (B2: n_exact_match_abs / n_matched
--                                          -- B3: n_abs_matches_lifetime_repaid
--                                          / n_single_loan_closed_charged)  >= 98%
--   p95 abs reconciliation error         (B2: p95_abs_diff_abs_ugx, relative
--                                          to median principal size)        <= 2%
--   Negative repayment share             (B0: negative_repayments /
--                                          total_repayments)             ~0%,
--                                          investigate any nonzero count
--                                          before trusting B1-B3's totals
--   Dedup tie collisions                 (F: keys with >1 row at the max
--                                          inserted_ts)                  0
--                                          (any nonzero count means dedup
--                                          is picking a row nondetermin-
--                                          istically for that key)
-- ============================================================================


-- ============================================================================
-- SECTION A -- Repayment amount semantics: gross or principal-only?
-- ============================================================================
-- Blocks cure-timing feature validity if wrong (see review). Uses ONLY
-- loan_state_daily's own authoritative totals -- independent of the
-- repay_attributed heuristic in borrower_history.txt, so this is a clean
-- check even if attribution turns out to be imperfect.
--
-- CAVEAT: this section tests what lifetime_repaid_ugx/lifetime_gross_repaid_ugx
-- mean, NOT directly what the atomic repayment_amount_ugx field in
-- repayments_daily means -- those are different columns in different tables.
-- If lifetime_repaid_ugx is itself a derived allocation (tracker-side split of
-- gross cash flow into a principal-recovered component) rather than a raw sum
-- of repayment_amount_ugx, it would read as principal-only by construction
-- regardless of what the atomic field contains. Section B3 below tests the
-- atomic field directly, on a subset where attribution is unambiguous, and is
-- the more decisive check -- treat A as corroborating context, not proof.
--
-- Logic: for CLOSED loans with a non-zero interest_and_penalty_ugx, compare
-- lifetime_repaid_ugx and lifetime_gross_repaid_ugx against (a) principal
-- alone and (b) principal + charge. This also directly checks whether the
-- "repaid" vs "gross_repaid" split loan_state_daily already exposes behaves
-- the way its naming implies (gross_repaid - repaid ~= interest_and_penalty_ugx),
-- rather than assuming and deriving that split myself.
WITH closed_charged_loans AS (
SELECT
disbursement_fid,
lifetime_disbursed_ugx,
lifetime_repaid_ugx,
lifetime_gross_repaid_ugx,
interest_and_penalty_ugx,
lifetime_repaid_ugx / NULLIF(lifetime_disbursed_ugx, 0) AS repaid_over_principal,
lifetime_repaid_ugx / NULLIF(lifetime_disbursed_ugx + interest_and_penalty_ugx, 0) AS repaid_over_gross,
lifetime_gross_repaid_ugx / NULLIF(lifetime_disbursed_ugx, 0) AS gross_repaid_over_principal,
lifetime_gross_repaid_ugx / NULLIF(lifetime_disbursed_ugx + interest_and_penalty_ugx, 0) AS gross_repaid_over_gross,
(lifetime_gross_repaid_ugx - lifetime_repaid_ugx) / NULLIF(interest_and_penalty_ugx, 0) AS gross_minus_repaid_over_charge
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (
PARTITION BY disbursement_fid
ORDER BY date_key DESC, inserted_ts DESC
) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE loan_status = 'CLOSED'
AND interest_and_penalty_ugx > 0
AND date_key <= :snapshot_dt
)
WHERE rn = 1
)
SELECT
COUNT(*) AS n_closed_charged_loans,
approx_percentile(repaid_over_principal, 0.5) AS median_repaid_over_principal,
approx_percentile(repaid_over_gross, 0.5) AS median_repaid_over_gross,
approx_percentile(gross_repaid_over_principal, 0.5) AS median_gross_repaid_over_principal,
approx_percentile(gross_repaid_over_gross, 0.5) AS median_gross_repaid_over_gross,
approx_percentile(gross_minus_repaid_over_charge, 0.5) AS median_gross_minus_repaid_over_charge,
SUM(CASE WHEN ABS(repaid_over_principal - 1.0) <= 0.02 THEN 1 ELSE 0 END) AS n_repaid_matches_principal,
SUM(CASE WHEN ABS(gross_repaid_over_gross - 1.0) <= 0.02 THEN 1 ELSE 0 END) AS n_gross_repaid_matches_gross,
SUM(CASE WHEN ABS(gross_minus_repaid_over_charge - 1.0) <= 0.02 THEN 1 ELSE 0 END) AS n_gap_matches_charge
FROM closed_charged_loans;
-- RESULT:
-- INTERPRETATION:
-- median_gross_minus_repaid_over_charge near 1.0 (and n_gap_matches_charge
-- high) confirms lifetime_repaid_ugx = principal-recovered and
-- lifetime_gross_repaid_ugx = principal + charge, exactly as the field names
-- imply -- gives confidence in interest_and_penalty_ugx as the right charge
-- figure to use elsewhere (e.g. Section C/D, or the latest_* enrichment
-- columns). If it does NOT hold, don't trust the "gross" naming at face
-- value -- fall back to Section B3 as the primary evidence.
-- median_repaid_over_principal near 1.0 with median_gross_repaid_over_principal
-- clearly above 1.0 is the expected pattern regardless of what the ATOMIC
-- repayment_amount_ugx field contains, per the caveat above -- do not treat
-- this alone as resolving whether borrower_history.txt's cure-timing logic
-- (which sums the atomic field) is correct. Go to Section B3 for that.


-- ============================================================================
-- SECTION B0 -- Repayment sign diagnostic
-- ============================================================================
-- classified in borrower_history.txt applies ABS() unconditionally to both
-- disbursement and repayment amounts (inherited from the original query's
-- pattern). That's only safe for repayments if every atomic row is genuine
-- positive cash flow and any negative rows are purely a sign convention. If
-- repayments_daily instead contains reversals, refunds, chargebacks, or
-- correction entries as negative rows, ABS() silently converts those into
-- additional positive recovery -- overstating repayment and making loans
-- cure too early.
--
-- CAVEAT: repayments_daily (per the schema sample this file was built from)
-- has no explicit transaction-type/status/reversal-code column, so this
-- diagnostic can only detect negative SIGN as a proxy -- it cannot attribute
-- a negative row to a specific cause. If negative rows exist in meaningful
-- volume, escalate to whoever owns the tracker tables' data dictionary
-- rather than guessing at the cause from this query alone.
SELECT
COUNT(*) AS total_repayments,
SUM(CASE WHEN repayment_amount_ugx < 0 THEN 1 ELSE 0 END) AS negative_repayments,
SUM(CASE WHEN repayment_amount_ugx < 0 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS negative_repayment_share,
SUM(CASE WHEN repayment_amount_ugx < 0 THEN repayment_amount_ugx ELSE 0 END) AS negative_repayment_total_ugx,
SUM(repayment_amount_ugx) AS raw_sum_ugx,
SUM(ABS(repayment_amount_ugx)) AS abs_sum_ugx,
SUM(ABS(repayment_amount_ugx)) - SUM(repayment_amount_ugx) AS overstatement_from_abs_ugx,
-- the only lifecycle-adjacent columns on this table, as a coarse proxy
-- breakdown in the absence of a real status/type field
SUM(CASE WHEN repayment_amount_ugx < 0 AND optout_transaction_id IS NOT NULL THEN 1 ELSE 0 END) AS negative_with_optout_txn,
SUM(CASE WHEN repayment_amount_ugx < 0 AND cancel_preapproval_id IS NOT NULL THEN 1 ELSE 0 END) AS negative_with_cancel_preapproval
FROM analytics.momo_loan_book_tracker_repayments_daily
WHERE try_cast(repayment_ts AS timestamp) IS NOT NULL
AND date(try_cast(repayment_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND inserted_ts <= :as_of_load_ts;
-- RESULT:
-- INTERPRETATION: negative_repayment_share should be ~0% per the acceptance
-- threshold above. overstatement_from_abs_ugx is exactly how much extra
-- "recovery" borrower_history.txt's ABS()-based classified CTE adds versus
-- the raw signed total -- if this is nonzero and negative_repayments is not
-- explained as a benign sign convention (confirm with the data owner), the
-- unconditional ABS() in borrower_history.txt needs to change to
-- source-semantic handling (e.g. excluding or separately classifying
-- negative rows) rather than blindly flipping their sign.


-- ============================================================================
-- SECTION B -- Repayment attribution quality (repay_attributed heuristic)
-- ============================================================================
-- Reproduces the borrower_history.txt attribution join, then checks it
-- against loan_state_daily's authoritative per-loan repaid total.
--
-- NOTE ON RUNNING THIS SECTION: B1, B2, and B3 are three independent
-- statements, each terminated by its own semicolon. A WITH clause's CTEs
-- only stay in scope for the single statement they precede -- they do NOT
-- carry over to the next semicolon-terminated statement in Trino/Athena. So
-- each of B1/B2/B3 repeats the same disb/disb_windows/repay/attributed
-- prefix rather than sharing one; that's intentional, not copy-paste debt.
--
-- B1/B2/B3 report BOTH the raw signed sum and the ABS()-basis sum (matching
-- what borrower_history.txt's classified CTE actually computes) -- see
-- Section B0 for why these can differ.

-- B1. Coverage: how many repayments got dropped for lack of a matching
-- disbursement, and how many disbursements look like rapid reborrows (a
-- proxy for "overlapping loans", since there is no exact concurrency signal)?
WITH disb AS (
SELECT
disbursement_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(disbursement_ts AS timestamp) AS disbursement_ts,
cast(disbursement_amount_ugx AS double) AS disbursed_amount
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE try_cast(disbursement_ts AS timestamp) IS NOT NULL
AND disbursement_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date(try_cast(disbursement_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND inserted_ts <= :as_of_load_ts
),
disb_windows AS (
SELECT
disbursement_fid,
phonenumber,
disbursement_ts,
disbursed_amount,
LEAD(disbursement_ts) OVER (
PARTITION BY phonenumber ORDER BY disbursement_ts, disbursement_fid
) AS next_disbursement_ts
FROM disb
),
repay AS (
SELECT
repayment_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(repayment_ts AS timestamp) AS repayment_ts,
cast(repayment_amount_ugx AS double) AS repayment_amount
FROM analytics.momo_loan_book_tracker_repayments_daily
WHERE try_cast(repayment_ts AS timestamp) IS NOT NULL
AND repayment_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date(try_cast(repayment_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND inserted_ts <= :as_of_load_ts
),
attributed AS (
SELECT
w.disbursement_fid,
r.repayment_fid,
r.repayment_ts,
r.repayment_amount,
w.disbursement_ts,
w.next_disbursement_ts,
-- flag repayments landing within 1 hour of the *next* disbursement --
-- ambiguous cases where the attribution boundary is a close call
CASE WHEN w.next_disbursement_ts IS NOT NULL
AND date_diff('second', r.repayment_ts, w.next_disbursement_ts) <= 3600
THEN 1 ELSE 0 END AS near_boundary_flag
FROM repay r
JOIN disb_windows w
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
)
SELECT
(SELECT COUNT(*) FROM repay) AS total_repayments,
(SELECT COUNT(*) FROM attributed) AS attributed_repayments,
(SELECT COUNT(*) FROM repay) - (SELECT COUNT(*) FROM attributed) AS dropped_repayments,
(SELECT COUNT(*) FROM attributed WHERE near_boundary_flag = 1) AS near_next_disbursement_boundary,
(SELECT COUNT(*) FROM disb_windows) AS total_disbursements,
-- proxy for "overlapping loans": a second disbursement to the same customer
-- before their prior loan could plausibly have cured (26h grace period)
(SELECT COUNT(*) FROM disb_windows
 WHERE next_disbursement_ts IS NOT NULL
 AND date_diff('hour', disbursement_ts, next_disbursement_ts) <= 26) AS rapid_reborrow_disbursements;
-- RESULT:
-- INTERPRETATION: dropped_repayments should be small and explainable (e.g.
-- repayments for loans disbursed before the source table's coverage starts).
-- A large dropped count means the attribution logic is silently discarding
-- real cash flow. near_next_disbursement_boundary flags repayments that could
-- plausibly belong to either the current or the next loan -- inspect these by
-- hand; a high count here is the clearest sign the sequential-loans
-- assumption is being stressed (e.g. rapid top-up/reborrow behavior).
-- rapid_reborrow_disbursements / total_disbursements is the closest available
-- proxy for the "rate of borrowers with overlapping or near-simultaneous
-- loans" metric -- see the acceptance thresholds block at the top of this
-- file. It is a proxy, not a direct measurement, since neither table exposes
-- an explicit "loan closed" event to test true concurrency against.


-- B2. Reconciliation against loan_state_daily's authoritative repaid total.
-- (mixes two possible explanations for a mismatch -- see the B3 caveat below)
WITH disb AS (
SELECT
disbursement_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(disbursement_ts AS timestamp) AS disbursement_ts,
cast(disbursement_amount_ugx AS double) AS disbursed_amount
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE try_cast(disbursement_ts AS timestamp) IS NOT NULL
AND disbursement_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date(try_cast(disbursement_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND inserted_ts <= :as_of_load_ts
),
disb_windows AS (
SELECT
disbursement_fid,
phonenumber,
disbursement_ts,
disbursed_amount,
LEAD(disbursement_ts) OVER (
PARTITION BY phonenumber ORDER BY disbursement_ts, disbursement_fid
) AS next_disbursement_ts
FROM disb
),
repay AS (
SELECT
repayment_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(repayment_ts AS timestamp) AS repayment_ts,
cast(repayment_amount_ugx AS double) AS repayment_amount
FROM analytics.momo_loan_book_tracker_repayments_daily
WHERE try_cast(repayment_ts AS timestamp) IS NOT NULL
AND repayment_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date(try_cast(repayment_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND inserted_ts <= :as_of_load_ts
),
attributed AS (
SELECT
w.disbursement_fid,
r.repayment_amount
FROM repay r
JOIN disb_windows w
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
),
-- LEFT JOIN from disb_windows (not GROUP BY attributed) so every known
-- disbursement gets a row here, including loans with zero repayments so
-- far -- otherwise those would wrongly show up as "state-only" below purely
-- for having no repayments yet, not for being outside the disbursement
-- population.
per_loan_attributed AS (
SELECT
w.disbursement_fid,
COALESCE(SUM(a.repayment_amount), 0) AS attributed_repaid_raw,
COALESCE(SUM(ABS(a.repayment_amount)), 0) AS attributed_repaid_abs
FROM disb_windows w
LEFT JOIN attributed a ON a.disbursement_fid = w.disbursement_fid
GROUP BY w.disbursement_fid
),
loan_state_latest AS (
SELECT disbursement_fid, lifetime_repaid_ugx
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (
PARTITION BY disbursement_fid
ORDER BY date_key DESC, inserted_ts DESC
) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE date_key <= :snapshot_dt
)
WHERE rn = 1
),
-- FULL OUTER JOIN so state-only and disbursement-only loans are visible
-- and excluded from the reconciliation rate, not silently folded into it
-- via COALESCE(...,0) as if a missing side were a real zero.
joined AS (
SELECT
COALESCE(pla.disbursement_fid, lsl.disbursement_fid) AS disbursement_fid,
pla.attributed_repaid_raw,
pla.attributed_repaid_abs,
lsl.lifetime_repaid_ugx,
CASE WHEN pla.disbursement_fid IS NULL THEN 1 ELSE 0 END AS state_only,
CASE WHEN lsl.disbursement_fid IS NULL THEN 1 ELSE 0 END AS disbursement_only
FROM per_loan_attributed pla
FULL OUTER JOIN loan_state_latest lsl ON lsl.disbursement_fid = pla.disbursement_fid
)
SELECT
COUNT(*) AS n_total,
SUM(state_only) AS n_state_only,
SUM(disbursement_only) AS n_disbursement_only,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0 THEN 1 ELSE 0 END) AS n_matched,
-- reconciliation rate computed ONLY over n_matched, both bases
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_raw - lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_exact_match_raw,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_exact_match_abs,
approx_percentile(
CASE WHEN state_only = 0 AND disbursement_only = 0
THEN ABS(attributed_repaid_raw - lifetime_repaid_ugx) END, 0.95) AS p95_abs_diff_raw_ugx,
approx_percentile(
CASE WHEN state_only = 0 AND disbursement_only = 0
THEN ABS(attributed_repaid_abs - lifetime_repaid_ugx) END, 0.95) AS p95_abs_diff_abs_ugx
FROM joined;
-- RESULT:
-- INTERPRETATION: n_state_only and n_disbursement_only should both be small
-- and explainable (e.g. disbursement_only = very recent loans not yet
-- reflected in loan_state_daily's next snapshot) -- if either is large, the
-- two source tables disagree on coverage and that's a data-pipeline issue
-- to raise separately, not something the reconciliation rate below should
-- absorb. n_exact_match_raw vs n_exact_match_abs / n_matched should both be
-- high per the acceptance threshold. CAVEAT: a mismatch on EITHER basis is
-- still ambiguous between two causes -- (1) the time-window heuristic
-- misattributing payments across loans, or (2) repayment_amount_ugx being
-- gross-of-charges while lifetime_repaid_ugx is principal-only (see Section
-- A's caveat). B3 below isolates cause (2) by removing cause (1) entirely,
-- so read B2 and B3 together: if B3 (no attribution ambiguity possible)
-- still shows a gap against lifetime_repaid_ugx, that gap is
-- amount-semantics, not attribution -- and B2's mismatch rate net of B3's
-- gap is the true attribution-error signal. Comparing n_exact_match_raw
-- against n_exact_match_abs also directly answers Section B0's question: if
-- raw matches much better than abs, negative rows are real adjustments that
-- ABS() is wrongly inflating into recovery.


-- B3. Isolate amount semantics from attribution error: single-loan borrowers only.
-- A borrower with exactly one loan in the query window has no second loan to
-- misattribute a repayment to -- disbursement_ts is a lower bound and there
-- is no next_disbursement_ts upper bound, so every one of their repayments
-- unambiguously belongs to that one loan. Any gap between the summed ATOMIC
-- repayment_amount_ugx and loan_state_daily's totals for this subset is
-- attributable to amount semantics (gross vs principal), not attribution
-- error -- this is the most direct test of what repayment_amount_ugx means.
WITH disb AS (
SELECT
disbursement_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(disbursement_ts AS timestamp) AS disbursement_ts,
cast(disbursement_amount_ugx AS double) AS disbursed_amount
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE try_cast(disbursement_ts AS timestamp) IS NOT NULL
AND disbursement_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date(try_cast(disbursement_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND inserted_ts <= :as_of_load_ts
),
single_loan_borrowers AS (
SELECT phonenumber
FROM disb
GROUP BY phonenumber
HAVING COUNT(*) = 1
),
single_loan_disb AS (
SELECT d.disbursement_fid, d.phonenumber, d.disbursement_ts, d.disbursed_amount
FROM disb d
JOIN single_loan_borrowers slb ON slb.phonenumber = d.phonenumber
),
repay AS (
SELECT
repayment_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(repayment_ts AS timestamp) AS repayment_ts,
cast(repayment_amount_ugx AS double) AS repayment_amount
FROM analytics.momo_loan_book_tracker_repayments_daily
WHERE try_cast(repayment_ts AS timestamp) IS NOT NULL
AND repayment_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date(try_cast(repayment_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND inserted_ts <= :as_of_load_ts
),
unambiguous_repaid AS (
SELECT
sld.disbursement_fid,
SUM(r.repayment_amount) AS atomic_repaid_raw,
SUM(ABS(r.repayment_amount)) AS atomic_repaid_abs
FROM single_loan_disb sld
JOIN repay r
ON r.phonenumber = sld.phonenumber
AND r.repayment_ts >= sld.disbursement_ts
GROUP BY sld.disbursement_fid
),
loan_state_latest AS (
SELECT disbursement_fid, lifetime_disbursed_ugx, lifetime_repaid_ugx,
lifetime_gross_repaid_ugx, interest_and_penalty_ugx
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (
PARTITION BY disbursement_fid
ORDER BY date_key DESC, inserted_ts DESC
) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE loan_status = 'CLOSED'
AND interest_and_penalty_ugx > 0
AND date_key <= :snapshot_dt
)
WHERE rn = 1
)
SELECT
COUNT(*) AS n_single_loan_closed_charged,
approx_percentile(ur.atomic_repaid_abs / NULLIF(ls.lifetime_disbursed_ugx, 0), 0.5) AS median_atomic_over_principal,
approx_percentile(ur.atomic_repaid_abs / NULLIF(ls.lifetime_disbursed_ugx + ls.interest_and_penalty_ugx, 0), 0.5) AS median_atomic_over_gross,
SUM(CASE WHEN ABS(ur.atomic_repaid_abs - ls.lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_abs_matches_lifetime_repaid,
SUM(CASE WHEN ABS(ur.atomic_repaid_abs - ls.lifetime_gross_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_abs_matches_lifetime_gross_repaid,
SUM(CASE WHEN ABS(ur.atomic_repaid_raw - ls.lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_raw_matches_lifetime_repaid,
SUM(CASE WHEN ABS(ur.atomic_repaid_raw - ls.lifetime_gross_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_raw_matches_lifetime_gross_repaid
FROM unambiguous_repaid ur
JOIN loan_state_latest ls ON ls.disbursement_fid = ur.disbursement_fid;
-- RESULT:
-- INTERPRETATION: this is the decisive check, not Section A.
-- Amount semantics: if n_abs_matches_lifetime_repaid (or n_raw_matches_
-- lifetime_repaid) is high, repayment_amount_ugx is principal-only. If the
-- *_gross_repaid variants are high instead, repayments are gross-of-charges.
-- If neither matches cleanly, repayment allocation is more complex than a
-- simple gross/principal split -- ask whoever owns the tracker tables' data
-- dictionary rather than guessing further.
-- Sign handling: comparing the raw-basis matches against the abs-basis
-- matches directly answers Section B0's question on this unambiguous
-- subset -- if raw matches noticeably better, negative rows are real
-- adjustments (reversals/refunds) that ABS() is wrongly turning into
-- recovery, and borrower_history.txt's classified CTE needs to stop
-- applying ABS() unconditionally to repayments.
-- CORRECTION to an earlier version of this note: if repayments turn out to
-- be gross-of-charges, "floor the cure comparison at disbursed_amount" is
-- NOT by itself a fix. Flooring only changes the threshold gross cash must
-- cross -- it says nothing about the ORDER charges are allocated in. If a
-- payment applies to interest/penalty before principal, cumulative gross
-- cash flow crossing disbursed_amount still does not prove principal itself
-- has been repaid; it could mean charges were partly repaid while principal
-- remains outstanding. A correct fix needs the tracker's actual allocation
-- waterfall (principal-first vs charge-first vs pro-rata), not just a
-- different threshold -- if that isn't documented anywhere, ask for it
-- explicitly rather than assuming principal-first.


-- ============================================================================
-- SECTION C -- ANOMALY_OPEN exclusion impact
-- ============================================================================
-- NOTE ON POPULATION: total_loans/total_borrowers here are loan_state_daily's
-- own population (every disbursement_fid it currently carries a snapshot
-- for), not necessarily identical to the disbursement-driven loan_level
-- population borrower_history.txt actually produces -- appropriate for this
-- section's specific question ("what does excluding ANOMALY_OPEN cost
-- against everything loan_state_daily knows about"), but don't read
-- anomaly_open_loans/total_loans as exactly the fraction of loan_level rows
-- that get dropped. Section D below is held to the exact-population standard
-- since it's aggregated onto the final borrower_history.txt output shape.
SELECT
COUNT(*) AS total_loans,
SUM(CASE WHEN loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true THEN 1 ELSE 0 END) AS anomaly_open_loans,
COUNT(DISTINCT customer_msisdn) AS total_borrowers,
COUNT(DISTINCT CASE WHEN loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true THEN customer_msisdn END) AS borrowers_with_an_anomaly_loan
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (
PARTITION BY disbursement_fid
ORDER BY date_key DESC, inserted_ts DESC
) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE date_key <= :snapshot_dt
)
WHERE rn = 1;
-- RESULT:
-- INTERPRETATION: report anomaly_open_loans / total_loans and
-- borrowers_with_an_anomaly_loan / total_borrowers as the population-shift
-- cost of the exclusion decision -- record this figure alongside the
-- decision, since it changes every affected borrower's prior-loan windows,
-- streaks, and lifetime rates, not just the anomalous loan itself.


-- ============================================================================
-- SECTION D -- has_pre_window_history impact
-- ============================================================================
-- Rebuilt on the SAME surviving population borrower_history.txt actually
-- produces (normalized phonenumber, deduped disbursements, ANOMALY_OPEN
-- excluded) -- the previous version grouped raw loan_state_daily.
-- customer_msisdn directly, which could include borrowers absent from the
-- disbursement population, unnormalized duplicate phone representations,
-- borrowers represented only by excluded anomaly loans, and a possible
-- null-customer group. That made it a source-table diagnostic, not the
-- actual percentage in the produced borrower dataset.
WITH disb AS (
SELECT
disbursement_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE try_cast(disbursement_ts AS timestamp) IS NOT NULL
AND disbursement_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date(try_cast(disbursement_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND inserted_ts <= :as_of_load_ts
),
loan_state_snapshot AS (
SELECT disbursement_fid, loan_status, is_anomaly_open, has_pre_window_history
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (
PARTITION BY disbursement_fid
ORDER BY date_key DESC, inserted_ts DESC
) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE disbursement_fid IS NOT NULL
AND date_key <= :snapshot_dt
)
WHERE rn = 1
),
surviving_loans AS (
SELECT d.disbursement_fid, d.phonenumber, ls.has_pre_window_history
FROM disb d
LEFT JOIN loan_state_snapshot ls ON ls.disbursement_fid = d.disbursement_fid
WHERE ls.disbursement_fid IS NULL
OR NOT (ls.loan_status = 'ANOMALY_OPEN' OR ls.is_anomaly_open = true)
)
SELECT
COUNT(*) AS total_borrowers,
SUM(CASE WHEN flagged THEN 1 ELSE 0 END) AS borrowers_with_pre_window_history,
SUM(CASE WHEN flagged THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS pct_left_censored
FROM (
SELECT
phonenumber,
bool_or(has_pre_window_history) AS flagged
FROM surviving_loans
GROUP BY phonenumber
);
-- RESULT:
-- INTERPRETATION: pct_left_censored is the share of borrowers whose
-- total_loans/first_loan_ts/lifetime rates from borrower_history.txt should
-- be treated as incomplete, not zero-history. Feed this into model
-- monitoring; if material, consider segmenting or excluding these borrowers
-- from lifetime-rate-dependent policy decisions until resolved.


-- ============================================================================
-- SECTION E -- Candidate old->new loan identity bridge
-- ============================================================================
-- disbursement_external_id / repayment_external_id may tie back to the old
-- transaction log's transactionid/requestid. Existence check only -- if these
-- match, they're a far better reconciliation key than the msisdn+timestamp
-- proxy used elsewhere in this file.
SELECT
COUNT(*) AS new_disbursements_with_external_id,
COUNT(DISTINCT d.disbursement_external_id) AS distinct_external_ids,
SUM(CASE WHEN t.requestid IS NOT NULL THEN 1 ELSE 0 END) AS matched_to_old_requestid
FROM analytics.momo_loan_book_tracker_disbursements_daily d
LEFT JOIN devdata.xtrafloat_daily_trans t
ON cast(t.requestid AS varchar) = cast(d.disbursement_external_id AS varchar)
WHERE d.disbursement_external_id IS NOT NULL
AND date(try_cast(d.disbursement_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d');
-- RESULT:
-- INTERPRETATION: if matched_to_old_requestid is a high fraction of
-- new_disbursements_with_external_id, disbursement_external_id is a real
-- bridge to the old requestid space -- use it (not msisdn+timestamp
-- proximity) for any old-vs-new loan-level comparison, including a tighter
-- version of Section B's reconciliation. If near-zero, external_id is not the
-- bridge and old-vs-new comparison must stay at the aggregate/population
-- level (Section C/D style), not loan-by-loan.


-- ============================================================================
-- SECTION F -- Dedup tie-breaker collisions
-- ============================================================================
-- Every dedup in this file (and in borrower_history.txt) orders by
-- inserted_ts DESC only. If two rows for the same business key share the
-- exact same max inserted_ts, ROW_NUMBER() picks between them
-- nondeterministically -- there's no known secondary field (sequence,
-- batch id, version) in the schema this file was built from to break the
-- tie reliably. Rather than assume that never happens, count it.
SELECT 'disbursements_daily' AS source_table, COUNT(*) AS keys_with_tie
FROM (
SELECT disbursement_fid, MAX(inserted_ts) AS max_ts, COUNT(*) AS n_at_max_ts
FROM analytics.momo_loan_book_tracker_disbursements_daily d
WHERE inserted_ts = (
SELECT MAX(inserted_ts) FROM analytics.momo_loan_book_tracker_disbursements_daily d2
WHERE d2.disbursement_fid = d.disbursement_fid
)
GROUP BY disbursement_fid
HAVING COUNT(*) > 1
)
UNION ALL
SELECT 'repayments_daily', COUNT(*)
FROM (
SELECT repayment_fid
FROM analytics.momo_loan_book_tracker_repayments_daily r
WHERE inserted_ts = (
SELECT MAX(inserted_ts) FROM analytics.momo_loan_book_tracker_repayments_daily r2
WHERE r2.repayment_fid = r.repayment_fid
)
GROUP BY repayment_fid
HAVING COUNT(*) > 1
)
UNION ALL
SELECT 'loan_state_daily (per disbursement_fid, date_key)', COUNT(*)
FROM (
SELECT disbursement_fid, date_key
FROM analytics.momo_loan_book_tracker_loan_state_daily l
WHERE inserted_ts = (
SELECT MAX(inserted_ts) FROM analytics.momo_loan_book_tracker_loan_state_daily l2
WHERE l2.disbursement_fid = l.disbursement_fid AND l2.date_key = l.date_key
)
GROUP BY disbursement_fid, date_key
HAVING COUNT(*) > 1
);
-- RESULT:
-- INTERPRETATION: keys_with_tie should be 0 for all three rows per the
-- acceptance threshold above. Any nonzero count means dedup is currently
-- choosing arbitrarily for that many business keys -- ask the table owner
-- whether a real tie-breaker field exists (sequence number, batch id,
-- ingestion offset) before shipping; if none exists, this needs to be
-- documented as a known nondeterminism rather than silently accepted.
