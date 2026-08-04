# CreditRisk — Governance Framework

## 1. Data Classification

All data processed by this pipeline is classified into one of three tiers. Controls
apply at every stage: ingestion, processing, storage, and output.

| Tier | Examples | Controls |
|---|---|---|
| **Restricted** | MSISDN, national_id, date_of_birth, account_name/number | Never enters model features (enforced by `PD_FEATURE_BLACKLIST`); dropped at loader boundary (`_drop_extra_pii`); not in scoring outputs (`FINAL_OUTPUT_COLUMNS`); pseudonymised in training exports (`ops_scored.csv` uses SHA-256 prefix); never logged (enforced by `PIIRedactingFilter`) |
| **Sensitive** | cal_pd, assigned_limit, risk_tier, loan history, on-time rates | Output CSV is gitignored; access restricted to authorised personnel; not published to external systems without explicit approval |
| **Internal** | model_metadata.json, drift reports, alert JSON files, transform_report.csv | Stored in gitignored `pd_model/artifacts/` and `monitoring/`; retention per section 5 |

## 2. Model Approval Process

A new model version requires the following evidence before it may be promoted to the
production champion:

| Gate | Requirement |
|---|---|
| AUC | Val AUC ≥ 0.70 for both XGBoost and LightGBM |
| Bootstrap CI | Paired bootstrap 95% CI for AUC difference excludes zero (or champion is not significantly worse than challenger) |
| Calibration | Brier score and calibration curve reviewed by Model Owner |
| PSI | Score PSI on holdout < 0.10 (stable population) |
| Fairness review | Thin-file vs thick-file approval rates reviewed; any demographic slice analysis available must be reviewed |
| Sign-off | Model Owner approval (primary); Risk Lead awareness notification |

Approval is recorded as a PR comment on the model promotion commit referencing
`model_metadata.json` values (`git_commit`, `xgb_val_auc`, `lgb_val_auc`).

## 3. Policy Configuration Approval

Any change to `DEFAULT_CAP_CONFIG` in
`extrafloat/engine/extrafloat_limit_engine_caps.py` requires:

1. A pull request with a written impact analysis: which agents are affected, estimated
   limit change direction and magnitude (use `run_credit_risk_pipeline.py` on a shadow
   dataset if available).
2. Two approvals: Risk Lead (required) + Model Owner (required).
3. The CODEOWNERS file enforces this on GitHub.

Changes to tier multipliers, combination weights, or regulatory caps are treated as
policy changes and follow the same process.

## 4. Rollback Procedure

**Trigger:** CRITICAL drift alert (PSI ≥ 0.25) OR confirmed model defect.

**Steps:**

1. **Freeze** new scoring runs immediately. Disable the cron job or batch trigger.
2. **Confirm** the issue via `run_drift_monitor.py` and the alert JSON in `monitoring/`.
3. **Identify** root cause: data feed change? model artifact corruption? config change?
   Check `model_metadata.json` for the active `git_commit` and `training_completed_at`.
4. **Restore** the previous artifact set from backup storage (see ops team for artifact
   backup location). Replace the contents of `pd_model/artifacts/`.
5. **Verify** artifact integrity: `make check-artifacts`. Must pass without errors.
6. **Smoke test** on 10 rows: `python run_credit_risk_pipeline.py --pd-model-file <small_sample>`.
7. **Re-enable** scoring. Notify Risk Lead and Model Owner.
8. **Post-mortem** within 48 hours (see `docs/RUNBOOK.md` for template).

## 5. Data Retention and Deletion

| Asset | Retention | Deletion trigger |
|---|---|---|
| Training CSVs (raw agent data) | 2 years minimum (Bank of Uganda audit trail requirement) | On expiry, secure deletion by Data Custodian |
| Model artifacts (`pd_model/artifacts/`) | Retain all versions until 2 successive stable successor versions exist | Manual deletion by Model Owner after review |
| Scoring outputs (`output/`) | 90-day rolling window | Automated cleanup script (see `Makefile`) |
| `ops_scored.csv` | 1 year (pseudonymised — agent_msisdn replaced with SHA-256 prefix) | On expiry, delete by Data Custodian |
| Drift reports and alert JSON | 180 days | Automated cleanup |
| Application logs | 30 days | Log rotation managed at infrastructure level |

## 6. Operational Ownership

| Role | Responsibilities |
|---|---|
| **Model Owner** | Model approval, champion promotion, AUC monitoring, model post-mortems |
| **Risk Lead** | Policy configuration approval, regulatory cap compliance, limit policy review |
| **Data Custodian** | PII handling, data retention enforcement, data subject rights requests, PDPA compliance |
| **On-Call Engineer** | Drift alert response, artifact integrity checks, rollback execution, incident triage |

On-call rotation and escalation path are documented in `docs/RUNBOOK.md`.

## 7. Fairness and Equitable Treatment

This pipeline serves financially excluded populations. The following commitments apply:

- The thin-file scorecard provides a separate, conservative treatment path for agents
  with fewer than 3 lifetime loans, preventing the model from systematically
  disadvantaging new-to-credit agents.
- Any slicing analysis by agent geography, tenure, or agent tier that becomes available
  must be reviewed by the Model Owner before a new model version is promoted.
- Disparate impact testing across agent tier is a committed roadmap item (see
  `MODEL_CARD.md`). It becomes mandatory when demographic-proxy data is available.
- Limit decisions are explainable via SHAP values and `final_decision_reason` codes
  (see `docs/engine_output_data_dictionary.md`).

## 8. Regulatory Context

This system operates under:

- **Bank of Uganda National Payment Systems Act** — sets the regulatory cap of
  UGX 5,000,000 per transaction (enforced in `extrafloat_limit_engine_caps.py`
  `_apply_regulatory_cap()`).
- **Uganda Data Protection and Privacy Act 2019 (PDPA)** — governs personal data
  handling. See `PRIVACY.md` for compliance controls.
- **MTN Group Data Policy** — governs data received from MTN MoMo systems.

Any change to the regulatory cap value requires written confirmation from the
compliance team before being merged.
