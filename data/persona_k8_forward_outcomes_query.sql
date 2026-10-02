-- =============================================================================
-- persona_k8_forward_outcomes_query.sql
-- =============================================================================
-- Forward-looking outcome panel for the frozen K=8 persona segmentation:
--
--     Persona(t) -> Performance(t+1 .. n)
--
-- where t = 2026-07-31, the same MoMo-mart snapshot date
-- scripts/profile_persona_k8.py used to assign C0-C7
-- (see segmentation_outputs/persona_k8_profile/k8_cluster_assignments.csv
-- for the persona_cluster/persona_name each customer_msisdn already has).
-- This does NOT re-run or touch the clustering itself -- Persona(t) is
-- already fixed; this query only pulls what happened AFTER t.
--
-- GRAIN: one row per customer_msisdn (borrower), aggregating loan activity
-- over the forward window t0+1 day through window_end. window_end is
-- MAX(state_date) actually present in the data, not a hardcoded date --
-- so the achieved horizon stays visible (fwd_window_days) and this query
-- keeps working unmodified as more data accumulates.
--
-- DESIGN CHOICE -- loan_state_daily only, no disbursements_daily bridge:
-- loan_state_query_updated.txt (the PD-training query) needs to bridge
-- disbursement_fid -> loan_uid because it scores a specific target
-- disbursement using state known strictly before it. That bridge is ~150
-- lines of same-day positional matching and is the most failure-prone part
-- of that query. This query doesn't need it: loan_state_daily already
-- carries loan_uid and current_loan_start_date directly, which is enough
-- to identify "a new loan originated in the forward window" without ever
-- touching disbursements_daily. Deliberately simpler and lower-risk for a
-- first draft that hasn't been run against the real warehouse yet.
--
-- DESIGN CHOICE -- does not replicate lifetime_on_time_24h_rate exactly:
-- that formula lives further upstream (borrower_history.txt /
-- extrafloat_limit_engine_features.py) and wasn't worth re-deriving here.
-- Instead this reports directly-computable, honestly-labeled signals:
-- new borrowing volume/frequency, repayment coverage on that new
-- borrowing, worst aging reached by ANY loan (new or carried over) during
-- the window, and a 3+ day-past-due "bad" flag over the window -- the same
-- threshold data/loan_state_query_updated.txt's own bad_state_3dpd_30d
-- uses, just at a borrower/open-window grain instead of per-loan/30-day.
--
-- KNOWN CAVEATS CARRIED OVER FROM loan_state_query_updated.txt (same repo,
-- same source tables -- read that file's header for the full history):
--   - ANOMALY_OPEN does NOT mean debt transfer (confirmed: the flagged
--     loan's own lifetime_repaid_ugx keeps climbing normally under its own
--     disbursement_fid) -- treated here as a flag worth reporting
--     (fwd_any_anomaly_open), not folded into the bad-loan definition.
--   - closure_date is NOT reliable on its own (populated on some
--     non-terminal ANOMALY_OPEN rows too) -- every "closed" check below
--     requires loan_status IN ('SETTLED','CLOSED','OVERPAID') alongside
--     closure_date IS NOT NULL, matching that file's own guard.
--
-- USAGE: run against the warehouse, export to CSV, then join locally
-- against k8_cluster_assignments.csv (phonenumber == customer_msisdn
-- after digit-normalization) via
-- scripts/analyze_persona_k8_forward_outcomes.py.
--
-- FIX HISTORY
--   fwd_new_loans_closed_good_count / fwd_new_loans_closed_bad_count --
--   a real run on the full population (143,217 borrowers, 42-day window)
--   came back with closed_bad_count_total == 0 for every one of the 8
--   personas, despite fwd_any_bad_3dpd_rate_pct showing 14-86% bad rates
--   for that same population. Root cause: these two columns classified a
--   loan as good/bad using days_aging off its CLOSURE row only, and
--   loan_state_daily almost certainly resets days_aging once a loan
--   reaches SETTLED/CLOSED/OVERPAID (there's no "current" aging state for
--   a closed loan) -- so every closed loan silently read as "never late,"
--   regardless of how overdue it ran beforehand. Fixed in the
--   loan_worst_aging_in_window CTE (step 4b): now takes the MAX
--   days_aging across EVERY daily row that loan has in the window, the
--   same approach step 7's fwd_any_bad_3dpd already used correctly.
--   Every other column was unaffected by this bug.
-- =============================================================================

WITH

snapshot AS (
    SELECT DATE '2026-07-31' AS t0   -- the persona-assignment snapshot date
),

-- ---------------------------------------------------------------------
-- 1. Forward window boundary: latest state_date actually available.
-- ---------------------------------------------------------------------
window_bounds AS (
    SELECT
        s.t0,
        MAX(
            CAST(DATE_PARSE(CAST(ls.date_key AS VARCHAR), '%Y%m%d') AS DATE)
        ) AS window_end
    FROM snapshot s
    CROSS JOIN analytics.momo_loan_book_tracker_loan_state_daily ls
    WHERE ls.ova = 'XTRAFLOAT-AGENT'
    GROUP BY s.t0
),

-- ---------------------------------------------------------------------
-- 2. One physical state record per (customer, loan_uid, date) -- same
--    dedup discipline as loan_state_query_updated.txt's loan_state_ranked.
-- ---------------------------------------------------------------------
loan_state_daily_ranked AS (
    SELECT
        ls.customer_msisdn,
        ls.loan_uid,
        ls.loan_seq,
        ls.current_loan_start_date,
        ls.closure_date,
        TRY_CAST(ls.lifetime_disbursed_ugx AS DOUBLE) AS loan_disbursed_ugx,
        TRY_CAST(ls.lifetime_repaid_ugx AS DOUBLE) AS loan_repaid_ugx,
        ls.days_aging,
        ls.aging_bucket,
        ls.loan_status,
        ls.is_anomaly_open,
        ls.is_active_loan,
        CAST(
            DATE_PARSE(CAST(ls.date_key AS VARCHAR), '%Y%m%d') AS DATE
        ) AS state_date,
        ROW_NUMBER() OVER (
            PARTITION BY ls.customer_msisdn, ls.loan_uid,
                         CAST(DATE_PARSE(CAST(ls.date_key AS VARCHAR), '%Y%m%d') AS DATE)
            ORDER BY ls.inserted_ts DESC, ls.loan_seq DESC
        ) AS state_day_rn
    FROM analytics.momo_loan_book_tracker_loan_state_daily ls
    WHERE ls.ova = 'XTRAFLOAT-AGENT'
      AND ls.loan_status <> 'NEVER_BORROWED'
),

loan_state_daily_dedup AS (
    SELECT * FROM loan_state_daily_ranked WHERE state_day_rn = 1
),

-- ---------------------------------------------------------------------
-- 3. Every loan touching the forward window at all (new, or pre-existing
--    and still showing activity during the window).
-- ---------------------------------------------------------------------
loans_in_window AS (
    SELECT lsd.*
    FROM loan_state_daily_dedup lsd
    CROSS JOIN window_bounds w
    WHERE lsd.state_date > w.t0
      AND lsd.state_date <= w.window_end
),

-- ---------------------------------------------------------------------
-- 4. Latest-in-window state row per loan_uid -- this loan's outcome AS OF
--    window_end, matching the "pick most recent state" pattern used
--    throughout loan_state_query_updated.txt.
-- ---------------------------------------------------------------------
latest_state_in_window AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY customer_msisdn, loan_uid
            ORDER BY state_date DESC
        ) AS latest_rn
    FROM loans_in_window
),

loan_final_state AS (
    SELECT * FROM latest_state_in_window WHERE latest_rn = 1
),

-- ---------------------------------------------------------------------
-- 4b. Worst-ever days_aging each loan reached ANYWHERE in the window --
--     NOT its closure-row value. A real run surfaced why this matters:
--     fwd_new_loans_closed_bad_count came back exactly 0 for every single
--     persona despite fwd_any_bad_3dpd_rate_pct (step 7, which already
--     takes a MAX across every daily row) showing 14-86% bad rates for
--     the same population. Root cause: loan_state_daily almost certainly
--     resets days_aging to 0/NULL once a loan reaches SETTLED/CLOSED/
--     OVERPAID -- "days currently aging" is undefined for a closed loan
--     -- so reading days_aging off only the closure row (as step 5 used
--     to) silently classifies every closed loan as "never late"
--     regardless of how overdue it ran beforehand. Fixed by computing
--     the worst value reached across ALL of that loan's rows in the
--     window, mirroring step 7's own (correct) approach, and joining it
--     into classified_loans below instead of trusting the closure row.
-- ---------------------------------------------------------------------
loan_worst_aging_in_window AS (
    SELECT
        customer_msisdn,
        loan_uid,
        MAX(COALESCE(days_aging, 0)) AS worst_days_aging_in_window
    FROM loans_in_window
    GROUP BY customer_msisdn, loan_uid
),

-- ---------------------------------------------------------------------
-- 5. Classify each loan touching the window as NEW (originated after t0)
--    or CARRIED_OVER (already open at or before t0).
-- ---------------------------------------------------------------------
classified_loans AS (
    SELECT
        lfs.*,
        wla.worst_days_aging_in_window,
        CASE
            WHEN lfs.current_loan_start_date > w.t0 THEN 'new'
            ELSE 'carried_over'
        END AS loan_window_role
    FROM loan_final_state lfs
    CROSS JOIN window_bounds w
    JOIN loan_worst_aging_in_window wla
      ON wla.customer_msisdn = lfs.customer_msisdn
     AND wla.loan_uid = lfs.loan_uid
),

-- ---------------------------------------------------------------------
-- 6. Borrower-grain aggregates over NEW loans only (forward borrowing
--    volume/frequency and repayment coverage on that new borrowing).
-- ---------------------------------------------------------------------
new_loan_aggregates AS (
    SELECT
        customer_msisdn,
        COUNT(*) AS fwd_new_loan_count,
        SUM(COALESCE(loan_disbursed_ugx, 0)) AS fwd_new_loans_disbursed_ugx,
        SUM(COALESCE(loan_repaid_ugx, 0)) AS fwd_new_loans_repaid_ugx,
        COUNT_IF(
            closure_date IS NOT NULL
            AND loan_status IN ('SETTLED', 'CLOSED', 'OVERPAID')
            AND worst_days_aging_in_window <= 3
        ) AS fwd_new_loans_closed_good_count,
        COUNT_IF(
            closure_date IS NOT NULL
            AND loan_status IN ('SETTLED', 'CLOSED', 'OVERPAID')
            AND worst_days_aging_in_window > 3
        ) AS fwd_new_loans_closed_bad_count
    FROM classified_loans
    WHERE loan_window_role = 'new'
    GROUP BY customer_msisdn
),

-- ---------------------------------------------------------------------
-- 7. Borrower-grain signals over ALL loans touching the window (new AND
--    carried-over) -- worst aging reached, any 3+ day-past-due event, any
--    anomaly-open event, still-active status at window_end. Worst-aging
--    and the bad flag look at EVERY state row in the window (not just
--    each loan's final row), so a transient spike that later recovered or
--    closed still counts -- a final-row-only MAX would miss that.
-- ---------------------------------------------------------------------
all_window_rows AS (
    SELECT liw.*
    FROM loans_in_window liw
),

window_wide_aggregates AS (
    SELECT
        customer_msisdn,
        MAX(COALESCE(days_aging, 0)) AS fwd_worst_days_aging,
        MAX(
            CASE WHEN COALESCE(days_aging, 0) > 3 THEN 1 ELSE 0 END
        ) AS fwd_any_bad_3dpd,
        MAX(
            CASE WHEN COALESCE(is_anomaly_open, FALSE) THEN 1 ELSE 0 END
        ) AS fwd_any_anomaly_open
    FROM all_window_rows
    GROUP BY customer_msisdn
),

-- Still-active status specifically AS OF window_end (latest row per loan),
-- across both new and carried-over loans.
still_active_aggregates AS (
    SELECT
        customer_msisdn,
        MAX(
            CASE
                WHEN COALESCE(is_active_loan, FALSE)
                  OR loan_status IN ('OPEN', 'OPEN_PRE_WINDOW', 'ANOMALY_OPEN')
                THEN 1 ELSE 0
            END
        ) AS fwd_still_active_at_window_end
    FROM classified_loans
    GROUP BY customer_msisdn
)

-- ---------------------------------------------------------------------
-- 8. Final output -- one row per borrower who had ANY loan activity
--    (new or carried-over) during the forward window. A borrower with NO
--    row here had zero loan-state activity in the window (fully dormant
--    through window_end, or the window is too short to have caught them
--    yet) -- report that as a population split in the analysis script,
--    not silently as zero/NULL here.
-- ---------------------------------------------------------------------
SELECT
    w.t0 AS fwd_window_start_exclusive,
    w.window_end AS fwd_window_end,
    DATE_DIFF('day', w.t0, w.window_end) AS fwd_window_days,
    COALESCE(sa.customer_msisdn, na.customer_msisdn, wa.customer_msisdn) AS customer_msisdn,
    COALESCE(na.fwd_new_loan_count, 0) AS fwd_new_loan_count,
    COALESCE(na.fwd_new_loans_disbursed_ugx, 0) AS fwd_new_loans_disbursed_ugx,
    COALESCE(na.fwd_new_loans_repaid_ugx, 0) AS fwd_new_loans_repaid_ugx,
    CASE
        WHEN COALESCE(na.fwd_new_loans_disbursed_ugx, 0) > 0
        THEN na.fwd_new_loans_repaid_ugx / na.fwd_new_loans_disbursed_ugx
    END AS fwd_new_loans_repayment_ratio,
    COALESCE(na.fwd_new_loans_closed_good_count, 0) AS fwd_new_loans_closed_good_count,
    COALESCE(na.fwd_new_loans_closed_bad_count, 0) AS fwd_new_loans_closed_bad_count,
    COALESCE(wa.fwd_worst_days_aging, 0) AS fwd_worst_days_aging,
    COALESCE(wa.fwd_any_bad_3dpd, 0) AS fwd_any_bad_3dpd,
    COALESCE(wa.fwd_any_anomaly_open, 0) AS fwd_any_anomaly_open,
    COALESCE(sa.fwd_still_active_at_window_end, 0) AS fwd_still_active_at_window_end
FROM still_active_aggregates sa
FULL OUTER JOIN new_loan_aggregates na ON na.customer_msisdn = sa.customer_msisdn
FULL OUTER JOIN window_wide_aggregates wa ON wa.customer_msisdn = COALESCE(sa.customer_msisdn, na.customer_msisdn)
CROSS JOIN window_bounds w
ORDER BY customer_msisdn
