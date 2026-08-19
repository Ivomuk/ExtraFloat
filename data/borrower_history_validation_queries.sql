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

-- B1. Coverage: how many repayments got dropped for lack of a matching disbursement?
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
(SELECT COUNT(*) FROM attributed WHERE near_boundary_flag = 1) AS near_next_disbursement_boundary;
-- RESULT:
-- INTERPRETATION: dropped_repayments should be small and explainable (e.g.
-- repayments for loans disbursed before the source table's coverage starts).
-- A large dropped count means the attribution logic is silently discarding
-- real cash flow. near_next_disbursement_boundary flags repayments that could
-- plausibly belong to either the current or the next loan -- inspect these by
-- hand; a high count here is the clearest sign the sequential-loans
-- assumption is being stressed (e.g. rapid top-up/reborrow behavior).


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
per_loan_attributed AS (
SELECT disbursement_fid, SUM(repayment_amount) AS attributed_repaid
FROM attributed
GROUP BY disbursement_fid
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
)
SELECT
COUNT(*) AS n_loans_compared,
SUM(CASE WHEN ABS(COALESCE(a.attributed_repaid, 0) - COALESCE(ls.lifetime_repaid_ugx, 0)) <= 1 THEN 1 ELSE 0 END) AS n_exact_match,
SUM(CASE WHEN ABS(COALESCE(a.attributed_repaid, 0) - COALESCE(ls.lifetime_repaid_ugx, 0)) > 1 THEN 1 ELSE 0 END) AS n_mismatch,
approx_percentile(ABS(COALESCE(a.attributed_repaid, 0) - COALESCE(ls.lifetime_repaid_ugx, 0)), 0.5) AS median_abs_diff_ugx,
approx_percentile(ABS(COALESCE(a.attributed_repaid, 0) - COALESCE(ls.lifetime_repaid_ugx, 0)), 0.95) AS p95_abs_diff_ugx
FROM loan_state_latest ls
LEFT JOIN per_loan_attributed a ON a.disbursement_fid = ls.disbursement_fid;
-- RESULT:
-- INTERPRETATION: n_mismatch should be low and p95_abs_diff_ugx small relative
-- to typical loan size. CAVEAT: a mismatch here is ambiguous between two
-- causes -- (1) the time-window heuristic misattributing payments across
-- loans, or (2) repayment_amount_ugx being gross-of-charges while
-- lifetime_repaid_ugx is principal-only (see Section A's caveat). B3 below
-- isolates cause (2) by removing cause (1) entirely, so read B2 and B3
-- together: if B3 (no attribution ambiguity possible) still shows a gap
-- against lifetime_repaid_ugx, that gap is amount-semantics, not attribution
-- -- and B2's mismatch rate net of B3's gap is the true attribution-error
-- signal.


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
SELECT sld.disbursement_fid, SUM(r.repayment_amount) AS atomic_repaid
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
approx_percentile(ur.atomic_repaid / NULLIF(ls.lifetime_disbursed_ugx, 0), 0.5) AS median_atomic_over_principal,
approx_percentile(ur.atomic_repaid / NULLIF(ls.lifetime_disbursed_ugx + ls.interest_and_penalty_ugx, 0), 0.5) AS median_atomic_over_gross,
SUM(CASE WHEN ABS(ur.atomic_repaid - ls.lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_atomic_matches_lifetime_repaid,
SUM(CASE WHEN ABS(ur.atomic_repaid - ls.lifetime_gross_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_atomic_matches_lifetime_gross_repaid
FROM unambiguous_repaid ur
JOIN loan_state_latest ls ON ls.disbursement_fid = ur.disbursement_fid;
-- RESULT:
-- INTERPRETATION: this is the decisive check, not Section A. If
-- n_atomic_matches_lifetime_repaid is high (and median_atomic_over_principal
-- ~= 1.0), repayment_amount_ugx is principal-only and borrower_history.txt's
-- cure-timing logic is correct as written. If
-- n_atomic_matches_lifetime_gross_repaid is high instead (and
-- median_atomic_over_gross ~= 1.0), repayments are gross-of-charges --
-- principal_cure_ts reads early, and the fix is to floor the cure comparison
-- at disbursed_amount only (excluding the charge component) rather than the
-- full attributed repayment sum. If neither matches cleanly, repayment
-- allocation is more complex than a simple gross/principal split and needs a
-- direct question to whoever owns the tracker tables' data dictionary.


-- ============================================================================
-- SECTION C -- ANOMALY_OPEN exclusion impact
-- ============================================================================
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
SELECT
COUNT(*) AS total_borrowers,
SUM(CASE WHEN flagged THEN 1 ELSE 0 END) AS borrowers_with_pre_window_history,
SUM(CASE WHEN flagged THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS pct_left_censored
FROM (
SELECT
customer_msisdn,
bool_or(has_pre_window_history) AS flagged
FROM analytics.momo_loan_book_tracker_loan_state_daily
WHERE date_key <= :snapshot_dt
GROUP BY customer_msisdn
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
