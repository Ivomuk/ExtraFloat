-- =============================================================================
-- agent_profile_snapshot_query.sql
-- =============================================================================
-- Produces the AGENT PROFILE SNAPSHOT used by both:
--   1. PD model Phase 2.1 feature engineering  (--pd-model-file)
--   2. ExtraFloat credit limit engine           (--transaction-file)
--
-- Both inputs consume the same underlying data; the only difference is that
-- load_transaction_capacity_features() renames agent_msisdn → msisdn for
-- the engine side, while run_inference_pipeline() keeps agent_msisdn.
-- You may pass the same exported CSV to both --pd-model-file and
-- --transaction-file.
--
-- Output columns (57 total):
--   agent_msisdn, snapshot_dt, agent_profile,
--   account_balance, average_balance, commission,
--   cash_out_vol_{1m,3m,6m}, cash_out_value_{1m,3m,6m},
--   cash_out_peers_{1m,3m,6m}, cash_out_comm_{1m,3m,6m}, cash_out_cust_{1m,3m,6m},
--   cash_in_vol_{1m,3m,6m},  cash_in_value_{1m,3m,6m},
--   cash_in_peers_{1m,3m,6m}, cash_in_comm_{1m,3m,6m},  cash_in_cust_{1m,3m,6m},
--   payment_vol_{1m,3m,6m}, payment_value_{1m,3m,6m},
--   payment_peers_{1m,3m,6m}, payment_comm_{1m,3m,6m}, payment_cust_{1m,3m,6m},
--   cust_{1m,3m,6m}, vol_{1m,3m,6m}
--
-- Table dependencies (Trino / Athena / Presto dialect):
--   devdata.account_holder_dump       — agent MSISDN roster + profile tier
--   devdata.agent_daily_snapshot      — daily balance, commission, activation_dt, dob
--   devdata.momo_transactions_daily   — raw MoMo cash-in / cash-out / payment rows
--
-- Replace the table names above with your actual catalogue.schema.table paths.
-- The column-level comments below document which source column maps to each alias.
--
-- Scoring usage  (current-date snapshot, no forward-looking label):
--   Set snapshot_dt to today's date; the WHERE on outcome_observed_30D
--   from the repayment query does NOT apply here.
--
-- Training usage  (historical snapshot with label):
--   Use the repayment query (data/loan_summary_query.txt) to supply the label
--   (bad_state_30D) and repayment behavioural features as a separate file
--   passed to --repayment-file.
-- =============================================================================

WITH

-- ---------------------------------------------------------------------------
-- 1. Snapshot configuration — change snapshot_dt for production runs
-- ---------------------------------------------------------------------------
snapshots AS (
    SELECT
        CAST(20260731 AS BIGINT) AS snapshot_dt,   -- YYYYMMDD integer
        'scoring'                AS split
),

-- ---------------------------------------------------------------------------
-- 2. Agent base population
--    Source: devdata.account_holder_dump
--    One row per active MoMo agent as of the most recent tbl_dt load.
-- ---------------------------------------------------------------------------
base AS (
    SELECT
        b0.msisdn,
        b0.profile                     AS agent_profile,
        s.snapshot_dt,
        s.split
    FROM (
        SELECT DISTINCT
            msisdn,
            profile
        FROM devdata.account_holder_dump
        WHERE tbl_dt = (
            SELECT MAX(tbl_dt)
            FROM devdata.account_holder_dump
            WHERE account_type = 'MOBILE MONEY'
              AND LOWER(profile) LIKE '%agent%'
        )
          AND account_type = 'MOBILE MONEY'
          AND LOWER(profile) LIKE '%agent%'
    ) b0
    CROSS JOIN snapshots s
),

-- ---------------------------------------------------------------------------
-- 3. Lookback window boundaries (YYYYMMDD integers)
--    dt_m1 = 1 month ago, dt_m3 = 3 months ago, dt_m6 = 6 months ago
-- ---------------------------------------------------------------------------
win AS (
    SELECT
        b.msisdn,
        b.agent_profile,
        b.snapshot_dt,
        b.split,
        TRY_CAST(
            DATE_FORMAT(
                DATE_ADD('month', -1, DATE_PARSE(CAST(b.snapshot_dt AS VARCHAR), '%Y%m%d')),
                '%Y%m%d'
            ) AS BIGINT
        ) AS dt_m1,
        TRY_CAST(
            DATE_FORMAT(
                DATE_ADD('month', -3, DATE_PARSE(CAST(b.snapshot_dt AS VARCHAR), '%Y%m%d')),
                '%Y%m%d'
            ) AS BIGINT
        ) AS dt_m3,
        TRY_CAST(
            DATE_FORMAT(
                DATE_ADD('month', -6, DATE_PARSE(CAST(b.snapshot_dt AS VARCHAR), '%Y%m%d')),
                '%Y%m%d'
            ) AS BIGINT
        ) AS dt_m6
    FROM base b
),

-- ---------------------------------------------------------------------------
-- 4. Agent daily profile: balance, commission, activation_dt, date_of_birth
--    Source: devdata.agent_daily_snapshot
--    Take the most recent row at or before snapshot_dt for each agent.
--    Expected columns in source table:
--      msisdn, tbl_dt, account_balance, average_balance_30d (→ average_balance),
--      commission_mtd (→ commission), activation_dt, date_of_birth
-- ---------------------------------------------------------------------------
profile AS (
    SELECT
        p.msisdn,
        p.account_balance,
        p.average_balance_30d                       AS average_balance,
        p.commission_mtd                            AS commission,
        TRY_CAST(p.activation_dt  AS DATE)          AS activation_dt,
        TRY_CAST(p.date_of_birth  AS DATE)          AS date_of_birth
    FROM (
        SELECT
            t.*,
            ROW_NUMBER() OVER (
                PARTITION BY t.msisdn
                ORDER BY t.tbl_dt DESC
            ) AS rn
        FROM devdata.agent_daily_snapshot t
        JOIN win w
            ON w.msisdn   = t.msisdn
           AND t.tbl_dt  <= w.snapshot_dt
    ) p
    WHERE p.rn = 1
),

-- ---------------------------------------------------------------------------
-- 5. Raw MoMo transactions in the 6-month lookback window
--    Source: devdata.momo_transactions_daily
--    Expected columns:
--      msisdn (or phonenumber), tbl_dt, trantype, amount,
--      counterparty_msisdn (for peer detection), comm_amount, cust_msisdn
--    trantype values: 'CASH_IN', 'CASH_OUT', 'PAYMENT'
--    (Replace with your actual trantype labels if they differ.)
-- ---------------------------------------------------------------------------
txn_pull AS (
    SELECT
        w.msisdn,
        w.snapshot_dt,
        w.dt_m1,
        w.dt_m3,
        w.dt_m6,
        t.tbl_dt,
        t.trantype,
        TRY_CAST(t.amount       AS DOUBLE)          AS amount,
        TRY_CAST(t.comm_amount  AS DOUBLE)          AS comm_amount,
        t.counterparty_msisdn,
        t.cust_msisdn
    FROM win w
    JOIN devdata.momo_transactions_daily t
        ON  t.msisdn   = w.msisdn
        AND t.tbl_dt  >  w.dt_m6
        AND t.tbl_dt  <= w.snapshot_dt
        AND t.trantype IN ('CASH_IN', 'CASH_OUT', 'PAYMENT')
),

-- ---------------------------------------------------------------------------
-- 6. Transaction feature aggregations by type × horizon
-- ---------------------------------------------------------------------------
features AS (
    SELECT
        msisdn,
        snapshot_dt,

        -- ── CASH-OUT ──────────────────────────────────────────────────────
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_OUT' AND tbl_dt > dt_m1 THEN tbl_dt || '-' || cust_msisdn END), 0)  AS cash_out_vol_1m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_OUT' AND tbl_dt > dt_m3 THEN tbl_dt || '-' || cust_msisdn END), 0)  AS cash_out_vol_3m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_OUT'                     THEN tbl_dt || '-' || cust_msisdn END), 0)  AS cash_out_vol_6m,

        COALESCE(SUM(CASE WHEN trantype = 'CASH_OUT' AND tbl_dt > dt_m1 THEN amount END), 0)   AS cash_out_value_1m,
        COALESCE(SUM(CASE WHEN trantype = 'CASH_OUT' AND tbl_dt > dt_m3 THEN amount END), 0)   AS cash_out_value_3m,
        COALESCE(SUM(CASE WHEN trantype = 'CASH_OUT'                     THEN amount END), 0)   AS cash_out_value_6m,

        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_OUT' AND tbl_dt > dt_m1 AND counterparty_msisdn IS NOT NULL THEN counterparty_msisdn END), 0) AS cash_out_peers_1m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_OUT' AND tbl_dt > dt_m3 AND counterparty_msisdn IS NOT NULL THEN counterparty_msisdn END), 0) AS cash_out_peers_3m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_OUT'                     AND counterparty_msisdn IS NOT NULL THEN counterparty_msisdn END), 0) AS cash_out_peers_6m,

        COALESCE(SUM(CASE WHEN trantype = 'CASH_OUT' AND tbl_dt > dt_m1 THEN comm_amount END), 0) AS cash_out_comm_1m,
        COALESCE(SUM(CASE WHEN trantype = 'CASH_OUT' AND tbl_dt > dt_m3 THEN comm_amount END), 0) AS cash_out_comm_3m,
        COALESCE(SUM(CASE WHEN trantype = 'CASH_OUT'                     THEN comm_amount END), 0) AS cash_out_comm_6m,

        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_OUT' AND tbl_dt > dt_m1 THEN cust_msisdn END), 0) AS cash_out_cust_1m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_OUT' AND tbl_dt > dt_m3 THEN cust_msisdn END), 0) AS cash_out_cust_3m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_OUT'                     THEN cust_msisdn END), 0) AS cash_out_cust_6m,

        -- ── CASH-IN ───────────────────────────────────────────────────────
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_IN' AND tbl_dt > dt_m1 THEN tbl_dt || '-' || cust_msisdn END), 0)  AS cash_in_vol_1m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_IN' AND tbl_dt > dt_m3 THEN tbl_dt || '-' || cust_msisdn END), 0)  AS cash_in_vol_3m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_IN'                     THEN tbl_dt || '-' || cust_msisdn END), 0)  AS cash_in_vol_6m,

        COALESCE(SUM(CASE WHEN trantype = 'CASH_IN' AND tbl_dt > dt_m1 THEN amount END), 0)   AS cash_in_value_1m,
        COALESCE(SUM(CASE WHEN trantype = 'CASH_IN' AND tbl_dt > dt_m3 THEN amount END), 0)   AS cash_in_value_3m,
        COALESCE(SUM(CASE WHEN trantype = 'CASH_IN'                     THEN amount END), 0)   AS cash_in_value_6m,

        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_IN' AND tbl_dt > dt_m1 AND counterparty_msisdn IS NOT NULL THEN counterparty_msisdn END), 0) AS cash_in_peers_1m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_IN' AND tbl_dt > dt_m3 AND counterparty_msisdn IS NOT NULL THEN counterparty_msisdn END), 0) AS cash_in_peers_3m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_IN'                     AND counterparty_msisdn IS NOT NULL THEN counterparty_msisdn END), 0) AS cash_in_peers_6m,

        COALESCE(SUM(CASE WHEN trantype = 'CASH_IN' AND tbl_dt > dt_m1 THEN comm_amount END), 0) AS cash_in_comm_1m,
        COALESCE(SUM(CASE WHEN trantype = 'CASH_IN' AND tbl_dt > dt_m3 THEN comm_amount END), 0) AS cash_in_comm_3m,
        COALESCE(SUM(CASE WHEN trantype = 'CASH_IN'                     THEN comm_amount END), 0) AS cash_in_comm_6m,

        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_IN' AND tbl_dt > dt_m1 THEN cust_msisdn END), 0) AS cash_in_cust_1m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_IN' AND tbl_dt > dt_m3 THEN cust_msisdn END), 0) AS cash_in_cust_3m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'CASH_IN'                     THEN cust_msisdn END), 0) AS cash_in_cust_6m,

        -- ── PAYMENT ───────────────────────────────────────────────────────
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'PAYMENT' AND tbl_dt > dt_m1 THEN tbl_dt || '-' || cust_msisdn END), 0)  AS payment_vol_1m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'PAYMENT' AND tbl_dt > dt_m3 THEN tbl_dt || '-' || cust_msisdn END), 0)  AS payment_vol_3m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'PAYMENT'                     THEN tbl_dt || '-' || cust_msisdn END), 0)  AS payment_vol_6m,

        COALESCE(SUM(CASE WHEN trantype = 'PAYMENT' AND tbl_dt > dt_m1 THEN amount END), 0)   AS payment_value_1m,
        COALESCE(SUM(CASE WHEN trantype = 'PAYMENT' AND tbl_dt > dt_m3 THEN amount END), 0)   AS payment_value_3m,
        COALESCE(SUM(CASE WHEN trantype = 'PAYMENT'                     THEN amount END), 0)   AS payment_value_6m,

        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'PAYMENT' AND tbl_dt > dt_m1 AND counterparty_msisdn IS NOT NULL THEN counterparty_msisdn END), 0) AS payment_peers_1m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'PAYMENT' AND tbl_dt > dt_m3 AND counterparty_msisdn IS NOT NULL THEN counterparty_msisdn END), 0) AS payment_peers_3m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'PAYMENT'                     AND counterparty_msisdn IS NOT NULL THEN counterparty_msisdn END), 0) AS payment_peers_6m,

        COALESCE(SUM(CASE WHEN trantype = 'PAYMENT' AND tbl_dt > dt_m1 THEN comm_amount END), 0) AS payment_comm_1m,
        COALESCE(SUM(CASE WHEN trantype = 'PAYMENT' AND tbl_dt > dt_m3 THEN comm_amount END), 0) AS payment_comm_3m,
        COALESCE(SUM(CASE WHEN trantype = 'PAYMENT'                     THEN comm_amount END), 0) AS payment_comm_6m,

        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'PAYMENT' AND tbl_dt > dt_m1 THEN cust_msisdn END), 0) AS payment_cust_1m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'PAYMENT' AND tbl_dt > dt_m3 THEN cust_msisdn END), 0) AS payment_cust_3m,
        COALESCE(COUNT(DISTINCT CASE WHEN trantype = 'PAYMENT'                     THEN cust_msisdn END), 0) AS payment_cust_6m,

        -- ── ALL-TYPE TOTALS ───────────────────────────────────────────────
        COALESCE(COUNT(DISTINCT CASE WHEN tbl_dt > dt_m1 THEN cust_msisdn END), 0) AS cust_1m,
        COALESCE(COUNT(DISTINCT CASE WHEN tbl_dt > dt_m3 THEN cust_msisdn END), 0) AS cust_3m,
        COALESCE(COUNT(DISTINCT                                cust_msisdn    ), 0) AS cust_6m,

        COALESCE(COUNT(DISTINCT CASE WHEN tbl_dt > dt_m1 THEN tbl_dt || '-' || cust_msisdn END), 0) AS vol_1m,
        COALESCE(COUNT(DISTINCT CASE WHEN tbl_dt > dt_m3 THEN tbl_dt || '-' || cust_msisdn END), 0) AS vol_3m,
        COALESCE(COUNT(DISTINCT                                tbl_dt || '-' || cust_msisdn    ), 0) AS vol_6m

    FROM txn_pull
    GROUP BY msisdn, snapshot_dt
),

-- ---------------------------------------------------------------------------
-- 7. Cluster-level averages (used by Phase 2.1 commission comparison features)
--    agent_profile (Silver Class / Gold Class / etc.) is the cluster label.
--    These become commission_cluster_mean, vol_3m_cluster_mean, etc.
-- ---------------------------------------------------------------------------
cluster_stats AS (
    SELECT
        w.agent_profile,
        AVG(p.commission)        AS commission_cluster_mean,
        AVG(f.vol_3m)            AS vol_3m_cluster_mean,
        AVG(p.commission)        AS cluster_avg_commission,
        AVG(f.vol_3m)            AS cluster_avg_vol_3m
    FROM win w
    JOIN profile p  ON p.msisdn = w.msisdn
    JOIN features f ON f.msisdn = w.msisdn AND f.snapshot_dt = w.snapshot_dt
    GROUP BY w.agent_profile
)

-- ---------------------------------------------------------------------------
-- 8. Final output — one row per agent
-- ---------------------------------------------------------------------------
SELECT
    w.msisdn                                    AS agent_msisdn,
    w.snapshot_dt,
    w.split,
    w.agent_profile,

    -- Balance & commission
    COALESCE(p.account_balance,  0)             AS account_balance,
    COALESCE(p.average_balance,  0)             AS average_balance,
    COALESCE(p.commission,       0)             AS commission,

    -- Date passthrough columns (excluded from model features, kept for audit)
    p.activation_dt,
    p.date_of_birth,

    -- Cluster reference columns (used by Phase 2.1 comparison features)
    cs.commission_cluster_mean,
    cs.vol_3m_cluster_mean,
    cs.cluster_avg_commission,
    cs.cluster_avg_vol_3m,

    -- Cash-out
    COALESCE(f.cash_out_vol_1m,   0)            AS cash_out_vol_1m,
    COALESCE(f.cash_out_vol_3m,   0)            AS cash_out_vol_3m,
    COALESCE(f.cash_out_vol_6m,   0)            AS cash_out_vol_6m,
    COALESCE(f.cash_out_value_1m, 0)            AS cash_out_value_1m,
    COALESCE(f.cash_out_value_3m, 0)            AS cash_out_value_3m,
    COALESCE(f.cash_out_value_6m, 0)            AS cash_out_value_6m,
    COALESCE(f.cash_out_peers_1m, 0)            AS cash_out_peers_1m,
    COALESCE(f.cash_out_peers_3m, 0)            AS cash_out_peers_3m,
    COALESCE(f.cash_out_peers_6m, 0)            AS cash_out_peers_6m,
    COALESCE(f.cash_out_comm_1m,  0)            AS cash_out_comm_1m,
    COALESCE(f.cash_out_comm_3m,  0)            AS cash_out_comm_3m,
    COALESCE(f.cash_out_comm_6m,  0)            AS cash_out_comm_6m,
    COALESCE(f.cash_out_cust_1m,  0)            AS cash_out_cust_1m,
    COALESCE(f.cash_out_cust_3m,  0)            AS cash_out_cust_3m,
    COALESCE(f.cash_out_cust_6m,  0)            AS cash_out_cust_6m,

    -- Cash-in
    COALESCE(f.cash_in_vol_1m,    0)            AS cash_in_vol_1m,
    COALESCE(f.cash_in_vol_3m,    0)            AS cash_in_vol_3m,
    COALESCE(f.cash_in_vol_6m,    0)            AS cash_in_vol_6m,
    COALESCE(f.cash_in_value_1m,  0)            AS cash_in_value_1m,
    COALESCE(f.cash_in_value_3m,  0)            AS cash_in_value_3m,
    COALESCE(f.cash_in_value_6m,  0)            AS cash_in_value_6m,
    COALESCE(f.cash_in_peers_1m,  0)            AS cash_in_peers_1m,
    COALESCE(f.cash_in_peers_3m,  0)            AS cash_in_peers_3m,
    COALESCE(f.cash_in_peers_6m,  0)            AS cash_in_peers_6m,
    COALESCE(f.cash_in_comm_1m,   0)            AS cash_in_comm_1m,
    COALESCE(f.cash_in_comm_3m,   0)            AS cash_in_comm_3m,
    COALESCE(f.cash_in_comm_6m,   0)            AS cash_in_comm_6m,
    COALESCE(f.cash_in_cust_1m,   0)            AS cash_in_cust_1m,
    COALESCE(f.cash_in_cust_3m,   0)            AS cash_in_cust_3m,
    COALESCE(f.cash_in_cust_6m,   0)            AS cash_in_cust_6m,

    -- Payment
    COALESCE(f.payment_vol_1m,    0)            AS payment_vol_1m,
    COALESCE(f.payment_vol_3m,    0)            AS payment_vol_3m,
    COALESCE(f.payment_vol_6m,    0)            AS payment_vol_6m,
    COALESCE(f.payment_value_1m,  0)            AS payment_value_1m,
    COALESCE(f.payment_value_3m,  0)            AS payment_value_3m,
    COALESCE(f.payment_value_6m,  0)            AS payment_value_6m,
    COALESCE(f.payment_peers_1m,  0)            AS payment_peers_1m,
    COALESCE(f.payment_peers_3m,  0)            AS payment_peers_3m,
    COALESCE(f.payment_peers_6m,  0)            AS payment_peers_6m,
    COALESCE(f.payment_comm_1m,   0)            AS payment_comm_1m,
    COALESCE(f.payment_comm_3m,   0)            AS payment_comm_3m,
    COALESCE(f.payment_comm_6m,   0)            AS payment_comm_6m,
    COALESCE(f.payment_cust_1m,   0)            AS payment_cust_1m,
    COALESCE(f.payment_cust_3m,   0)            AS payment_cust_3m,
    COALESCE(f.payment_cust_6m,   0)            AS payment_cust_6m,

    -- All-type totals
    COALESCE(f.cust_1m,           0)            AS cust_1m,
    COALESCE(f.cust_3m,           0)            AS cust_3m,
    COALESCE(f.cust_6m,           0)            AS cust_6m,
    COALESCE(f.vol_1m,            0)            AS vol_1m,
    COALESCE(f.vol_3m,            0)            AS vol_3m,
    COALESCE(f.vol_6m,            0)            AS vol_6m

FROM win w
LEFT JOIN profile p
    ON  p.msisdn = w.msisdn
LEFT JOIN features f
    ON  f.msisdn      = w.msisdn
    AND f.snapshot_dt = w.snapshot_dt
LEFT JOIN cluster_stats cs
    ON  cs.agent_profile = w.agent_profile
ORDER BY w.msisdn
