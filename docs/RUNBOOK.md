# CreditRisk — Operations Runbook

## 1. Scheduled Jobs

| Job | Schedule (UTC) | Command | Owner |
|---|---|---|---|
| Daily drift monitor | `0 6 * * *` | `python run_drift_monitor.py --ref-file <baseline> --cur-file <latest>` | On-Call Engineer |
| Weekly artifact integrity | `0 0 * * 1` | `make check-artifacts` | On-Call Engineer |
| Daily scoring run | Per deployment schedule | `make run` | On-Call Engineer |

## 2. Alert Thresholds

Alerts are emitted by `pd_model/run_drift_monitor.py` and
`extrafloat/monitoring/extrafloat_drift_monitor.py` via Slack
(`SLACK_WEBHOOK` env var) and as JSON files in `monitoring/`.

| Level | PSI threshold | CSI threshold | Required action |
|---|---|---|---|
| **WARNING** | ≥ 0.10 | ≥ 0.10 | Investigate within 4 hours. Check data feed for anomalies. Review feature drift report. No immediate scoring freeze required. |
| **CRITICAL** | ≥ 0.25 | ≥ 0.25 | Freeze new scoring runs immediately. Escalate to Model Owner within 1 hour. Begin rollback evaluation. |

Engine-specific alerts (policy health, cap driver shifts) follow the same severity
levels and are documented in `extrafloat/monitoring/extrafloat_drift_monitor.py`
docstrings.

## 3. Incident Response Checklist

### Step 1 — Confirm the alert

```bash
# Check the latest drift report JSON
ls -lt monitoring/*.json | head -5
cat monitoring/<latest_report>.json | python -m json.tool
```

Check Slack for the structured alert payload. Note the `psi`, `stability`, and
`alert_features` fields.

### Step 2 — Assess severity

- **WARNING**: Continue monitoring. Check the data feed for the affected features.
  Log the investigation in the incident log.
- **CRITICAL**: Proceed to Step 3 immediately.

### Step 3 — Freeze scoring (CRITICAL only)

Disable the cron job or batch trigger for `make run`. Notify Risk Lead and Model Owner
via Slack and email within 1 hour.

### Step 4 — Identify root cause

Check each possible cause in order:

1. **Data feed change**: Compare current input CSV column stats to the previous run.
   Missing columns, format changes, or upstream system changes are common causes.
2. **Model artifact corruption**: Run `make check-artifacts`. A `MissingArtifactError`
   or `ArtifactVerificationError` confirms artifact issues.
3. **Population shift**: Run `pd_model/run_drift_monitor.py` with a larger reference
   window. Is the drift sustained or a one-time spike?
4. **Config change**: Check `git log --oneline -10` for recent changes to
   `DEFAULT_CAP_CONFIG` or feature configuration.

### Step 5 — Remediate

**Rollback** (model artifact issue or unexplained CRITICAL drift):
See section 4 below.

**Data feed fix** (upstream format change):
Coordinate with the data engineering team to restore the expected schema. Update
`extrafloat_data_loaders.py` column rename mappings if needed and deploy via PR.

**Retraining** (sustained concept drift — PSI ≥ 0.10 for 5+ consecutive days):
Follow the model approval process in `GOVERNANCE.md` section 2.

### Step 6 — Re-enable and verify

```bash
make check-artifacts          # must pass
make run                      # smoke test on latest data
python run_drift_monitor.py   # confirm PSI is below WARNING threshold
```

Notify Risk Lead and Model Owner that scoring has resumed.

### Step 7 — Post-mortem

Complete the post-mortem template (section 6) within 48 hours of resolution.
Distribute to Model Owner, Risk Lead, and On-Call Engineer.

## 4. Rollback Procedure

```bash
# 1. Stop the scoring cron job (environment-specific)

# 2. Restore previous artifact set from backup storage
#    (backup location: coordinate with ops team)
cp -r <backup_artifacts_dir>/* pd_model/artifacts/

# 3. Verify artifact integrity
make check-artifacts
# Expected: "All artifacts present and checksums verified."

# 4. Smoke test on a small sample
python run_credit_risk_pipeline.py \
    --pd-model-file     data/sample_10_agents.csv \
    --transaction-file  data/sample_10_txn.csv \
    --loan-file         data/sample_10_loans.csv \
    --borrower-file     data/sample_10_borrowers.csv \
    --artifacts-dir     pd_model/artifacts/

# 5. Re-enable the scoring cron job

# 6. Run drift monitor to confirm stability
python run_drift_monitor.py --ref-file <baseline> --cur-file <latest>
```

## 5. Escalation Path

| Situation | Primary contact | Secondary |
|---|---|---|
| Score drift (WARNING) | On-Call Engineer | Model Owner (notify) |
| Score drift (CRITICAL) | Model Owner | Risk Lead (notify within 1 hour) |
| PII incident (data exposure) | Data Custodian | Model Owner + Risk Lead (notify within 1 hour) |
| Model artifact corruption | On-Call Engineer | Model Owner |
| Regulatory cap breach | Risk Lead | Compliance team |
| Policy config error | Risk Lead | Model Owner |

Contacts for each role are maintained in the team's internal directory (not committed
to this repository).

**Slack channel:** `#credit-risk-alerts` (production alerts auto-posted here)

## 6. Post-Mortem Template

```
Incident post-mortem
====================
Date: YYYY-MM-DD
Duration: X hours Y minutes
Severity: WARNING | CRITICAL
Reported by: <name>

Summary
-------
<One paragraph: what happened, what was affected, how it was resolved>

Timeline
--------
HH:MM  Alert fired / issue detected
HH:MM  On-call engineer acknowledged
HH:MM  Root cause identified
HH:MM  Remediation started
HH:MM  Scoring resumed / incident resolved

Root cause
----------
<Technical description of the root cause>

Contributing factors
--------------------
<What conditions allowed this to happen>

Impact
------
- Scoring runs affected: N
- Agents without updated limits: N
- Duration of impact: X hours

Remediation steps taken
-----------------------
1.
2.

Prevention / follow-up actions
-------------------------------
[ ] Action item 1 — Owner — Due date
[ ] Action item 2 — Owner — Due date
```

## 7. Useful Commands

```bash
# Check artifact integrity
make check-artifacts

# Run drift monitor manually
python run_drift_monitor.py --ref-file <baseline_csv> --cur-file <current_csv>

# Run full pipeline
make run

# Run tests
make test

# Check for missing required artifacts
python -c "
from pathlib import Path
from run_credit_risk_pipeline import _check_artifacts
_check_artifacts(Path('pd_model/artifacts/'))
print('All artifacts OK')
"

# Inspect latest model metadata
python -c "
import json
from pathlib import Path
meta = json.loads(Path('pd_model/artifacts/model_metadata.json').read_text())
print('Trained at:', meta.get('training_completed_at'))
print('Git commit:', meta.get('git_commit', 'unknown')[:12])
print('Champion:', meta.get('champion'))
print('XGB AUC:', meta.get('xgb_val_auc'))
print('LGB AUC:', meta.get('lgb_val_auc'))
"
```
