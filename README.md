# CreditRisk

End-to-end pipeline for MTN MoMo Uganda XtraFloat float limit assignment.

The pipeline runs in two phases:

1. **PD model** — trains XGBoost and LightGBM classifiers on historical agent repayment data to produce a calibrated probability of default (`cal_pd`) per agent.
2. **Credit limit engine** — combines four independent cap signals (capacity, recent usage, prior exposure, risk) and applies a policy layer to assign a UGX float limit and risk tier per agent. When `cal_pd` is available it replaces the engine's internal 7-signal risk blend.

---

## Architecture

```
Agent profile snapshot CSV
        │
        ├──► pd.read_csv()  ──────────────────────────────► PD model
        │     (agent_msisdn key, raw)                        Phase 2.1 — transactional behaviour features
        │                                                    Phase 2.2 — repayment features (optional)
        │                                                         │
        │                                                         ▼
        │                                                       cal_pd per agent
        │                                                         │
        └──► load_transaction_capacity_features() ──► Engine ◄───┘  (left-joined on msisdn)
              (msisdn key, renamed)                   capacity cap
                                                      recent usage cap  ◄── loan summary CSV
                                                      prior exposure cap ◄── borrower credit CSV
                                                      risk cap (1 − cal_pd)
                                                         │
                                                         ▼
                                              assigned_limit, risk_tier, cal_pd
```

---

## Input files

Three CSV files are required. All accept comma- or tab-delimited format (auto-detected).

| Argument | Default | Source query | Description |
|---|---|---|---|
| `--transaction-file` | `data/agent_profile_snapshot.csv` | `data/agent_profile_snapshot_query.sql` | Agent profile snapshot — 57-column MTN MoMo behavioural features. Read raw for PD model; reloaded via capacity loader for engine. |
| `--loan-file` | `data/loan_summary.csv` | `data/loan_summary_query.txt` | XtraFloat disbursement/repayment volumes and penalty counts over 1m/3m/6m windows. Fed to engine recent usage cap. |
| `--borrower-file` | `data/borrower_credit.csv` | `data/borrower_credit_limit_aggregated.txt` | Lifetime borrower credit history — on-time rates, default rates, prior loan sizes. Fed to engine prior exposure cap. |

One optional file:

| Argument | Description |
|---|---|
| `--repayment-file` | XtraFloat repayment history. Enables Phase 2.2 PD features (DPD, penalty roll-forward, repayment consistency). Used to compute `distinct_loan_months` and `total_loans_6m` — agents with fewer than 4 active months or fewer than 5 loans are classified thin-file and scored by the calibrated LR with a 0.12 PD floor. |

---

## Quick start

### Install

```bash
pip install -e ".[dev,monitor]"
```

### Phase 1 — Train the PD model

Export historical agent snapshots from the warehouse using `data/agent_profile_snapshot_query.sql` and `data/loan_summary_query.txt`, then:

```bash
make train \
  TRAIN=data/agent_snapshot_train.csv \
  VAL=data/agent_snapshot_val.csv \
  REPAYMENT=data/repayments.csv
```

This produces the model artifacts in `pd_model/artifacts/`:

```
xgb_model.joblib          lgbm_model.joblib
feature_order.json         pd_calibration_map.csv
xgb_policy_thresholds.csv  lgb_policy_thresholds.csv
transform_report.csv
```

### Phase 2 — Score agents and assign limits

Export current-date agent data from the warehouse, then:

```bash
make run \
  TRANSACTION_FILE=data/agent_profile_snapshot.csv \
  LOAN_FILE=data/loan_summary.csv \
  BORROWER_FILE=data/borrower_credit.csv
```

Output is written to `output/credit_risk_output.csv`.

### Run directly

```bash
python run_credit_risk_pipeline.py \
    --transaction-file  data/agent_profile_snapshot.csv \
    --loan-file         data/loan_summary.csv \
    --borrower-file     data/borrower_credit.csv \
    --artifacts-dir     pd_model/artifacts \
    --repayment-file    data/repayments.csv \
    --output            output/credit_risk_output.csv
```

---

## Output columns

| Column | Type | Description |
|---|---|---|
| `msisdn` | string | Agent MSISDN |
| `assigned_limit` | UGX | Recommended float limit (rounded to nearest 100) |
| `risk_tier` | tier_1 … tier_4 | Risk classification. tier_1 = best (cal_pd < 0.15), tier_4 = highest risk (cal_pd ≥ 0.65) |
| `cal_pd` | float [0, 1] | Calibrated probability of default from PD model |
| `final_decision_reason` | string | Primary reason determining the final limit |
| `policy_reason` | string | Policy-stage reason (tier assignment, floor override, regulatory cap) |
| `combined_reason` | string | Combination-stage reason (weighting scheme applied) |
| `combined_top_driver` | string | Which cap was the binding constraint |
| `thin_file_flag` | 0 / 1 | 1 if the agent was scored by the thin-file LR path (fewer than 4 active loan months or fewer than 5 total loans in the 6-month window). |
| `pd_decile` | int 1–10 \| NA | Population-relative risk rank from `cal_pd` (1 = lowest risk, 10 = highest). NA for agents on the 7-signal fallback. |

See `docs/engine_output_data_dictionary.md` for the full column reference and reason code catalogue.

---

## Engine cap logic

The engine evaluates four independent cap signals and combines them with configurable weights:

| Cap | Weight (thick-file) | Weight (thin-file) | What it measures |
|---|---|---|---|
| Capacity | 40% | 25% | Business volume — balance, commission, transaction throughput |
| Recent usage | 25% | 15% | XtraFloat utilisation and repayment in the last 1–3 months |
| Prior exposure | 15% | 10% | Maximum loan size the agent has successfully serviced |
| Risk | 20% | 50% | `1 − cal_pd` when PD model runs; 7-signal blend otherwise |

**Thin-file** agents — those with fewer than 4 distinct active loan months OR fewer than 5 total loans in the 6-month repayment window — receive higher risk weight (50%) because repayment history is sparse. They are scored by a calibrated Logistic Regression rather than XGBoost, and their `cal_pd` is floored at the `thin_file_pd_prior` (0.12) so agents with no loan history are never treated as zero-risk. The experience ramp in the 7-signal fallback path does **not** apply on the `cal_pd` path.

After combining, the engine applies:
- **Thin-file hard cap**: thin-file agents are capped at the Bronze business-category flat amount (100,000 UGX). Agents already in a lower tier (New Bronze / Unknown, ceiling 50,000 UGX) keep their own category value and are not raised. Configurable via `combination.thin_file_max_tier` in `DEFAULT_CAP_CONFIG`.
- Risk-tier policy multiplier (tier_1 = 100%, tier_4 = 40%)
- Agent-tier ceiling (Silver / Gold / Platinum class multipliers)
- Bank of Uganda regulatory cap: **5,000,000 UGX**
- Global floor: **0 UGX**

---

## Makefile targets

| Target | Description |
|---|---|
| `make install` | Install package and dev dependencies |
| `make train` | Phase 1 — train PD model, write artifacts |
| `make run` | Phase 2 — score agents, assign limits |
| `make test` | Run all tests (pipeline + engine + PD model) |
| `make test-pipeline` | Run pipeline integration tests only |
| `make check-artifacts` | Verify all 7 PD model artifacts exist |

Key variables (override on the command line):

```bash
make train TRAIN=path/to/train.csv VAL=path/to/val.csv REPAYMENT=path/to/repayments.csv
make run   TRANSACTION_FILE=path/to/snapshot.csv LOAN_FILE=... BORROWER_FILE=...
```

---

## Repository structure

```
CreditRisk/
├── run_credit_risk_pipeline.py   # Pipeline entry point
├── run_engine.py                 # Engine-only CLI (no PD model)
├── pyproject.toml
├── Makefile
│
├── pd_model/                     # PD model — train and score
│   ├── run_pipeline.py           # Training CLI
│   ├── run_drift_monitor.py      # Drift monitoring CLI
│   ├── artifacts/                # Trained model artifacts (gitignored after training)
│   ├── config/                   # Feature config, model config
│   ├── preprocessing/            # Phase 2.1 (transaction features), Phase 2.2 (repayment features)
│   ├── modeling/                 # XGBoost, LightGBM, calibration, inference
│   ├── postprocessing/           # Scorecard, whitelist/blacklist evaluation
│   ├── monitoring/               # Drift, stress, vintage monitoring
│   └── tests/                   # 16 PD model unit tests
│
├── extrafloat/                   # Credit limit engine
│   ├── engine/
│   │   ├── extrafloat_limit_engine_caps.py      # Cap logic + DEFAULT_CAP_CONFIG
│   │   ├── extrafloat_limit_engine_features.py  # Feature engineering
│   │   └── run_extrafloat_limit_engine.py       # Engine entry point
│   ├── io/
│   │   └── extrafloat_data_loaders.py           # CSV loaders for 3 input files
│   └── monitoring/
│       └── extrafloat_drift_monitor.py
│
├── data/                         # SQL queries for warehouse extraction
│   ├── agent_profile_snapshot_query.sql    # --transaction-file source
│   ├── loan_summary_query.txt              # --loan-file source
│   └── borrower_credit_limit_aggregated.txt # --borrower-file source
│
├── docs/
│   └── engine_output_data_dictionary.md   # Full output column reference
│
└── tests/
    ├── pipeline/   # 19 end-to-end pipeline tests
    ├── engine/     # Engine integration tests
    ├── io/         # Data loader tests
    └── monitoring/ # Drift monitor tests
```

---

## Tests

```bash
# All tests
make test

# Pipeline integration tests only
make test-pipeline

# PD model tests only
python -m pytest pd_model/tests/ -v
```

19 pipeline tests cover: `cal_pd` path vs 7-signal fallback, tier assignment, experience factor isolation, preflight artifact check, and correct data routing between PD model and engine.

---

## Configuration

Engine parameters are controlled by `DEFAULT_CAP_CONFIG` in `extrafloat/engine/extrafloat_limit_engine_caps.py`. Override at runtime by passing `engine_config` to `run_credit_risk_pipeline()`:

```python
from run_credit_risk_pipeline import run_credit_risk_pipeline

result = run_credit_risk_pipeline(
    transaction_file="data/agent_profile_snapshot.csv",
    loan_file="data/loan_summary.csv",
    borrower_file="data/borrower_credit.csv",
    artifacts_dir="pd_model/artifacts",
    engine_config={
        "combination": {"risk_weight": 0.30, "capacity_weight": 0.35,
                        "recent_usage_weight": 0.20, "prior_exposure_weight": 0.15},
    },
)
```

---

## Known limitations

- **Risk tier thresholds are not yet calibrated.** The current thresholds (tier_1: cal_pd < 0.15, tier_2: < 0.40, tier_3: < 0.65) were set before the PD model was trained. They should be recalibrated against `pd_calibration_map.csv` once the model has been trained on production data, aligning cut-points with observed default rate step-changes.
- **`pd_decile` uses dynamic (per-run) quantile cuts.** Decile boundaries are recomputed from each scoring batch, so an agent's decile can shift as the population mix changes. Once `pd_calibration_map.csv` is available, freeze the boundaries to training-derived thresholds for production consistency.
- **Cap blend weights were designed for the 7-signal fallback.** With `cal_pd` now driving the risk cap, the 20% risk weight may underweight the PD signal. Consider increasing it at the expense of capacity weight once live performance is available.
