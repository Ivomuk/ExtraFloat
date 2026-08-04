# CreditRisk — Privacy and Data Protection

This document describes how personal data is handled by the CreditRisk pipeline in
accordance with the Uganda Data Protection and Privacy Act 2019 (PDPA) and the
Bank of Uganda National Payment Systems Act.

## 1. Data Inventory

| Field | Classification | Source | Enters model features? | Appears in scoring output? | Retention |
|---|---|---|---|---|---|
| MSISDN / agent_msisdn | Restricted (PII) | Transaction CSV, Borrower CSV | No — `PD_FEATURE_BLACKLIST` | No — excluded from `FINAL_OUTPUT_COLUMNS` | Join key only; not persisted in outputs |
| national_id / nid | Restricted (PII) | Borrower CSV (if present) | No | No | Dropped at loader by `_drop_extra_pii` |
| date_of_birth | Restricted (PII) | Transaction CSV (if present) | No — `DATE_COLS` + blacklist | No | Dropped at loader by `_drop_extra_pii` |
| account_name | Restricted (PII) | Transaction CSV (if present) | No | No | Dropped at loader |
| account_number | Restricted (PII) | Transaction CSV (if present) | No | No | Dropped at loader |
| cal_pd | Sensitive | Model output | N/A — derived score | Yes — output CSV | 90 days |
| assigned_limit | Sensitive | Engine output | N/A | Yes — output CSV | 90 days |
| risk_tier | Sensitive | Engine output | N/A | Yes — output CSV | 90 days |
| Behavioural features (volumes, rates) | Internal | All input CSVs | Yes | No (kept_intermediate=False) | 2 years (training data) |

## 2. MSISDN Handling

The MSISDN is the primary identifier for each agent. The following controls apply at
each pipeline stage:

| Stage | Control |
|---|---|
| **Ingestion** | MSISDN is normalised (strip whitespace, remove `.0` suffix) and used as the join key. It is not transformed beyond normalisation at this stage. |
| **Feature engineering** | MSISDN is in `PD_FEATURE_BLACKLIST` and `NON_BEHAVIOURAL_COLS`. It is structurally prevented from entering the model feature matrix. |
| **Logging** | `PIIRedactingFilter` in `pd_model/logging_config.py` scrubs Ugandan MSISDNs in international (`256xxxxxxxxx`) and local (`07xxxxxxxx`) formats from all log records before emission. The filter is installed on the root logger handler so it covers all loggers including `extrafloat/` modules. |
| **Scoring output** | `_trim_output_columns()` in `run_extrafloat_limit_engine.py` retains only `FINAL_OUTPUT_COLUMNS` by default. MSISDN is not in this list. |
| **Training export** | `ops_scored.csv` replaces `agent_msisdn` with the first 16 hex characters of its SHA-256 hash before writing. The raw MSISDN is not persisted. |
| **Error messages** | Exception messages embed only column names (string constants) and integer counts — never raw MSISDN cell values. |

## 3. Additional PII Fields

Beyond MSISDN, the data loaders enforce the following:

`_drop_extra_pii()` is called in all three loaders
(`load_transaction_capacity_features`, `load_loan_summary_recent_features`,
`load_borrower_limit_features`) and drops the following columns if present in the
source CSV: `date_of_birth`, `dob`, `account_name`, `account_number`, `acct_no`,
`national_id`, `nid`, `passport_no`, `imei`, `imsi`, `device_id`.

These columns are not required by any downstream pipeline stage. Dropping them at
the loader boundary prevents silent propagation into intermediate DataFrames.

## 4. Encryption

The CreditRisk application does not implement application-level encryption. The
following controls are required of the deployment environment:

- **At rest**: Training CSVs, model artifacts, and scoring outputs must be stored on
  encrypted volumes (AES-256 or equivalent). This is a deployment infrastructure
  requirement, not an application responsibility.
- **In transit**: All data transfers (file uploads, API calls, Slack webhooks) must
  occur over TLS 1.2+. The Slack webhook uses HTTPS (`urllib.request.urlopen` with
  an HTTPS URL).
- **Artifact integrity**: SHA-256 checksums are stored in `model_metadata.json` and
  verified by `_check_artifacts()` before every inference run. This detects tampering
  or corruption at rest.

## 5. Purpose Limitation

Data received from MTN MoMo systems is used solely for:
- Computing mobile money float credit limits for registered MTN MoMo agents in Uganda.
- Training and evaluating the PD (probability of default) model.

It must not be used for: consumer credit bureau reporting, identity verification for
non-MoMo purposes, advertising, or sharing with third parties outside the scope of the
data sharing agreement.

## 6. Data Minimisation

The pipeline applies data minimisation at two points:
1. PII columns not required downstream are dropped at the loader boundary (section 3).
2. The default output contains only the 9 columns in `FINAL_OUTPUT_COLUMNS` — no
   intermediate features or raw inputs are written unless `--keep-intermediate` is
   explicitly passed by an authorised operator.

## 7. Data Subject Rights

Agents whose data is processed by this system have rights under the Uganda PDPA,
including access, correction, and deletion.

**Deletion request process:**
1. Request received by the Data Custodian (see `GOVERNANCE.md` for role definition).
2. Data Custodian identifies all records containing the agent's MSISDN across:
   training CSVs, scoring outputs, `ops_scored.csv` (note: stored as SHA-256 prefix —
   locate via hash of the MSISDN), drift report CSVs.
3. Records are deleted from all storage locations within 30 days of the request.
4. If the agent's data was used in training, the Model Owner determines whether
   model retraining is required under the PDPA.

## 8. Regulatory References

- Uganda Data Protection and Privacy Act 2019 (PDPA)
- Bank of Uganda National Payment Systems Act (Cap. 234)
- MTN Group Data Privacy Policy (internal reference)
- Bank of Uganda Mobile Money Guidelines (2013, as amended)
