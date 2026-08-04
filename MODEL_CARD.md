# Model Card — CreditRisk PD Model

## Model Details

| Property | Value |
|---|---|
| **Model type** | Ensemble: XGBoost + LightGBM probability-of-default classifiers |
| **Calibration** | Isotonic regression on quantile-binned scores (monotonic PD vs. score) |
| **Champion selection** | Paired bootstrap AUC comparison; configurable via `--champion xgb\|lgb` |
| **Target variable** | `bad_state` — agent failed to repay within the defined window (binary) |
| **Feature count** | ~50 behavioural features (see `pd_model/config/feature_config.py`) |
| **Version tracking** | `model_metadata.json` — `git_commit`, `training_completed_at`, SHA-256 checksums |
| **Framework** | scikit-learn, XGBoost, LightGBM, pandas |
| **Language** | Python 3.10+ |

## Intended Use

**Primary use case:** Compute a probability-of-default score (`cal_pd`) for registered
MTN MoMo agents in Uganda. The score is used as the risk signal in the ExtraFloat
credit limit engine, which assigns a float credit limit (`assigned_limit`) to each agent.

**Intended users:** MTN MoMo credit risk team, model risk function, and the automated
scoring pipeline.

**Deployment context:** Batch scoring — the model scores a snapshot of active agents
on a scheduled basis (typically daily). It is not designed for real-time transaction
scoring.

## Out-of-Scope Uses

The following uses are **not** intended and should not be attempted without a separate
model risk review:

- Consumer credit decisions for individuals (natural persons) — the model is trained
  on agent behavioural data, not personal consumer credit data.
- Credit bureau reporting or any use that triggers a regulated credit decision.
- Real-time fraud detection at the transaction level.
- Deployment in markets outside Uganda without retraining on local data and regulatory
  review.
- Decisions based on protected characteristics (gender, ethnicity, religion, political
  opinion) — the model does not use or infer these, but output limits must not be used
  as a proxy for such characteristics.

## Performance

Performance metrics are populated from `model_metadata.json` after training.
The table below shows the fields stored; values are training-run specific.

| Metric | Field in model_metadata.json | Acceptance threshold |
|---|---|---|
| XGBoost Val AUC | `xgb_val_auc` | ≥ 0.70 |
| LightGBM Val AUC | `lgb_val_auc` | ≥ 0.70 |
| Bootstrap AUC 95% CI | `bootstrap.auc_ci_lo`, `bootstrap.auc_ci_hi` | CI reviewed by Model Owner |
| Training bad rate | `bad_rate_train` | Consistent with historical portfolio |
| Validation bad rate | `bad_rate_val` | Within ±20% of training bad rate |
| Score PSI (holdout) | Computed at inference via drift monitor | < 0.10 (stable) |

## Limitations

1. **Thin-file agents** (fewer than 3 lifetime loans) are routed to a separate
   rule-based scorecard (`pd_model/scoring/scorecard.py`). The ML model is not
   applied to this population. Scorecard weights are set conservatively and should be
   reviewed with new data as the thin-file population grows.

2. **Batch-dependent `pd_decile`** — the `pd_decile` column (1–10) is computed from
   the current scoring batch, not from fixed training-derived thresholds. An agent's
   decile may shift between runs if the scored population changes. This is a known
   limitation documented in `README.md`.

3. **Static tier thresholds** — risk tier assignment uses fixed `risk_tier_*_score_min`
   thresholds in `DEFAULT_CAP_CONFIG`. These were set before the model was trained.
   They should be calibrated against the actual score distribution after the first
   production training run.

4. **Temporal stability** — the model is trained on a time split (train cutoff defined
   by `--train-cutoff`). Concept drift is monitored via `run_drift_monitor.py` (PSI
   and CSI). CRITICAL drift (PSI ≥ 0.25) triggers a Slack alert and a cron exit code
   of 1, requiring rollback or retraining.

5. **Feature leakage guard** — the leakage detection system prevents DPD-adjacent
   columns from entering features, but cannot prevent all forms of indirect leakage.
   Feature set changes require manual review against `DataLeakageError` patterns in
   `pd_model/config/feature_config.py`.

## Fairness

**Known gap:** No formal disparate impact analysis has been performed across demographic
groups, geographic regions, or agent tier classifications.

**Existing mitigating controls:**

- **Thin-file treatment**: Agents with fewer than 3 lifetime loans receive a separate
  conservative scoring path, preventing systematic underserving of new-to-credit agents.
- **Explainability**: Every scored agent receives a `final_decision_reason` code and
  SHAP values are computed when the `shap` package is available, enabling individual
  limit decisions to be explained.
- **No protected-class features**: Gender, ethnicity, religion, and similar attributes
  are not in the feature set. `date_of_birth` is explicitly excluded via `DATE_COLS`
  and `PD_FEATURE_BLACKLIST`.

**Committed roadmap item:** Sliced AUC analysis and approval rate comparison by agent
tier and region will be added as a mandatory pre-promotion gate once demographic-proxy
data is available. See `GOVERNANCE.md` section 7.

## Ethical Considerations

- This model affects the financial livelihoods of MTN MoMo agents, many of whom depend
  on float credit access for their business operations. False negatives (low limits for
  creditworthy agents) have direct economic impact.
- Model risk committee review is required before the first production deployment.
- Limit decisions must be explainable to affected agents upon request, consistent with
  the Uganda PDPA.
- The model should be retrained at least annually or whenever score PSI exceeds 0.10
  on a sustained basis.

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | — | Initial model card. Performance values TBD pending first training run. |
