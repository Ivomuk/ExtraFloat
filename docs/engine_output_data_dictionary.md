# ExtraFloat Limit Engine - Output Data Dictionary

## How the Engine Works (Summary)

The engine computes a recommended **float limit** (in UGX) for each mobile money agent by evaluating four independent perspectives, combining them, then applying policy guardrails:

| Stage | What It Answers | Weight (Standard) | Weight (New Borrower) |
|---|---|---|---|
| **Capacity** | How much business does this agent do? | 40% | 25% |
| **Recent Usage** | How much float have they recently used and repaid? | 25% | 15% |
| **Prior Exposure** | What loan sizes have they successfully handled? | 15% | 10% |
| **Risk** | How reliably do they repay? | 20% | 50% |

After combining, the engine applies a **risk-tier policy multiplier** and enforces hard ceilings (agent tier, Bank of Uganda regulatory cap).

---

## Final Output Columns (Key Business Columns)

These are the columns that matter most for reporting and decisions.

| Column | Type | Description |
|---|---|---|
| `assigned_limit` | UGX (rounded to nearest 100) | **The recommended float limit.** This is the final number after all caps, policy adjustments, and regulatory checks. |
| `assigned_limit_pre_round` | UGX | The limit before rounding to the nearest 100 UGX. Useful for auditing rounding effects. |
| `risk_tier` | tier_1 / tier_2 / tier_3 / tier_4 | **Agent risk classification.** tier_1 = best (score >= 0.85), tier_4 = highest risk (score < 0.35). Determines the policy multiplier applied to the limit. |
| `final_decision_reason` | Text | **Why the agent got this limit.** The primary policy or rule that determined the final outcome (see Reason Codes below). |
| `policy_reason` | Text | The specific policy-stage reason (tier assignment, floor override, or regulatory cap). |
| `combined_reason` | Text | The combination-stage reason (standard vs thin-file weighting, smoothing, risk guardrail). |
| `combined_top_driver` | Text | Which of the four cap stages contributed most to the combined limit: `capacity_component`, `recent_usage_component`, `prior_exposure_component`, or `risk_component`. |
| `regulatory_cap_applied` | 0 / 1 | 1 if the Bank of Uganda regulatory ceiling (5,000,000 UGX) was the binding constraint. |

---

## Reason Codes (final_decision_reason / policy_reason / combined_reason)

### Policy Reasons
| Code | Meaning |
|---|---|
| `tier_1_policy` | Best risk tier (score >= 0.85). Limit = 100% of combined cap. |
| `tier_2_policy` | Good risk tier (score 0.60-0.85). Limit = 85% of combined cap. |
| `tier_3_policy` | Moderate risk tier (score 0.35-0.60). Limit = 65% of combined cap. |
| `tier_4_policy` | Highest risk tier (score < 0.35). Limit = 40% of combined cap. |
| `proven_good_floor_override` | Agent has 3+ loans, 90%+ on-time rate, and <=5% default rate. Their limit was raised to at least 85% of the combined cap, overriding the tier haircut. |
| `active_borrower_min_floor_override` | Agent has recent activity but the calculated limit was below 500 UGX. Floored to 500 UGX minimum. |

### Combined Reasons
| Code | Meaning |
|---|---|
| `standard_weighting_applied` | Standard 4-cap weighting used (40/25/15/20). |
| `thin_file_weighting_applied` | Agent has fewer than 3 lifetime loans. Risk cap weighted more heavily (25/15/10/50). |
| `prior_limit_smoothing_applied` | Agent had a prior limit; new limit was smoothed toward it (max +25% / -25% swing). |
| `risk_cap_guardrail_binding` | The risk cap was lower than the weighted combination and became the binding constraint. |

### Finalization Reasons
| Code | Meaning |
|---|---|
| `bou_regulatory_cap_applied` | Bank of Uganda regulatory ceiling (5,000,000 UGX) was the binding constraint. |
| `finalized_from_combined_cap` | No policy adjustments changed the limit; it came directly from the combined cap. |

---

## Risk Score and Tier Details

| Column | Type | Description |
|---|---|---|
| `risk_score` | 0.00 - 1.00 | Composite repayment reliability score. Higher = safer borrower. Weighted blend of on-time rate (30%), lifetime default rate (20%), recent default rate (20%), 50-loan window default rate (10%), cure speed (8%), repayment stability (7%), and cure-time volatility (5%). |
| `risk_tier` | tier_1 to tier_4 | Classification based on risk_score thresholds. |
| `policy_multiplier` | 0.40 - 1.00 | The fraction of the combined cap allowed for this risk tier: tier_1=1.00, tier_2=0.85, tier_3=0.65, tier_4=0.40. |

### Risk Tier Thresholds
| Tier | Risk Score Range | Policy Multiplier | Interpretation |
|---|---|---|---|
| tier_1 | >= 0.85 | 1.00 (100%) | Excellent repayment history |
| tier_2 | 0.60 - 0.84 | 0.85 (85%) | Good history, minor concerns |
| tier_3 | 0.35 - 0.59 | 0.65 (65%) | Moderate risk, limit reduced |
| tier_4 | < 0.35 | 0.40 (40%) | High risk, significant reduction |

---

## Stage 1: Capacity Cap

Measures agent business throughput - how much transaction volume flows through this agent.

| Column | Type | Description |
|---|---|---|
| `capacity_cap` | UGX | The capacity-based limit. Derived from agent's transaction activity (balance, revenue, volume, payments, customers) with log-scaling to prevent outlier distortion. |
| `capacity_score` | 0.00 - 1.00 | Normalized capacity signal (capacity relative to the global ceiling). |
| `capacity_structural` | UGX | Pure capacity signal before activity adjustment. Represents what the agent's business can support. |
| `capacity_raw` | UGX | Sum of all six weighted capacity components before log-scaling. |
| `capacity_top_driver` | Text | Which business metric contributes most to this agent's capacity: `balance`, `revenue`, `txn`, `payments`, `customers`, `volume`, or `no_capacity_signal`. |
| `capacity_balance_component` | UGX | Contribution from average account balance (weight: 22%). |
| `capacity_revenue_component` | UGX | Contribution from monthly revenue / commissions (weight: 28%). |
| `capacity_txn_component` | UGX | Contribution from transaction count (weight: 12%). |
| `capacity_payments_component` | UGX | Contribution from payment values (weight: 14%). |
| `capacity_customers_component` | UGX | Contribution from unique active customers (weight: 9%). |
| `capacity_volume_component` | UGX | Contribution from total transaction value (weight: 15%). |
| `capacity_activity_score` | 0.00 - 1.00 | Blend of operational activity (70%) and recent credit activity (30%). Used to attenuate capacity for inactive agents. |
| `capacity_effective_ceiling` | UGX | Per-agent maximum limit based on their agent tier (e.g., Bronze=100,000, Diamond=1,000,000). |
| `capacity_season_multiplier` | 0.85 or 1.00 | 0.85 during peak months (Jan, Aug, Sep, Dec) to prevent seasonal spikes from inflating limits. |
| `capacity_missing_inputs` | 0-12 | Count of capacity input signals that were completely absent. Higher values mean less reliable capacity estimate. |
| `capacity_fallback_inputs` | 0-12 | Count of capacity inputs that fell back to a secondary data source. |

---

## Stage 2: Recent Usage Cap

Measures how actively the agent has been using XtraFloat and how well they repay.

| Column | Type | Description |
|---|---|---|
| `recent_usage_cap` | UGX | Limit based on recent XtraFloat loan activity (disbursements and repayments in last 1-3 months). Zero if agent falls below the activity gate. |
| `recent_usage_disbursement_component` | UGX | Weighted contribution from recent loan disbursement amounts. |
| `recent_usage_repayment_component` | UGX | Weighted contribution from recent loan repayment amounts. Repayments weighted more (55%) than disbursements (45%). |
| `recent_usage_coverage_multiplier` | 0.85 - 1.20 | Bonus/reduction based on repayment coverage ratio. Agents who repay more of what they borrow get a boost (up to 1.20x). |
| `recent_usage_penalty_multiplier` | 0.00 - 1.00 | Haircut for penalty events. Each penalty reduces by 10% (e.g., 2 penalties = 0.80x). |
| `recent_usage_repayment_ratio` | 0.00 - 1.00 | Ratio of repayment amount to disbursement amount. Higher = better repayment behavior. |
| `recent_usage_repayment_ratio_multiplier` | 0.55 - 1.00 | Attenuation factor from repayment ratio. Low ratio = lower cap. |
| `recent_usage_active_flag` | 0 / 1 | 1 if agent's recent disbursement + repayment >= 100 UGX (activity gate). Agents below this get recent_usage_cap = 0. |
| `recent_usage_top_driver` | Text | Whether disbursements or repayments contributed more: `disbursement` or `repayment`. |
| `recent_usage_reason` | Text | Why the cap was set this way: `active_recent_usage_policy`, `inactive_recent_usage_gate`, `coverage_bonus_applied`, `repayment_ratio_haircut_applied`, or `penalty_event_haircut_applied`. |

---

## Stage 3: Prior Exposure Cap

Measures what loan sizes the agent has successfully handled historically.

| Column | Type | Description |
|---|---|---|
| `prior_exposure_cap` | UGX | Limit based on the agent's historical loan sizes. Existing borrowers: weighted blend of average and max prior loan sizes. New borrowers: 50% of their current loan size. |
| `prior_exposure_avg_component` | UGX | Contribution from average prior loan size (weight: 60%, multiplier: 0.90x). |
| `prior_exposure_max_component` | UGX | Contribution from maximum prior loan size (weight: 55%, multiplier: 1.05x). |
| `prior_exposure_new_to_credit_component` | UGX | For new-to-credit agents only: 50% of current loan size as a conservative starting point. |
| `prior_exposure_above_max_penalty_multiplier` | 0.85 or 1.00 | 0.85 if the agent's current loan exceeds their historical maximum (penalizes unusual growth). |
| `prior_exposure_growth_penalty_multiplier` | 0.75 - 1.15 | Adjusts for the spread between max and avg loan sizes. Wide spread = potential volatility. |
| `prior_exposure_recent_performance_multiplier` | 0.70 - 1.00 | Haircut based on recent repayment performance. Poor recent performance reduces the cap. |
| `prior_exposure_existing_cap_before_new_to_credit_override` | UGX | Cap calculated for existing borrowers, shown even for new-to-credit agents for comparison. |
| `prior_exposure_top_driver` | Text | Main factor: `avg` (average loan size), `max` (max loan size), or `new_to_credit` (first-time borrower proxy). |
| `prior_exposure_reason` | Text | Why: `existing_exposure_policy`, `new_to_credit_proxy_cap`, `above_prior_max_penalty_applied`, or `recent_performance_haircut_applied`. |

---

## Stage 4: Risk Cap

Sets a ceiling based on the agent's repayment reliability score.

| Column | Type | Description |
|---|---|---|
| `risk_cap` | UGX | Maximum limit the risk score allows. Calculated as risk_score x 1,000,000 UGX x experience_factor. |
| `risk_score` | 0.00 - 1.00 | Composite reliability score (see Risk Score section above). |

---

## Stage 5: Cap Combination

Merges the four individual caps into a single number.

| Column | Type | Description |
|---|---|---|
| `combined_cap` | UGX | Final combined limit after weighting, risk guardrail, and optional prior-limit smoothing. |
| `capacity_component` | UGX | Capacity cap's weighted contribution to the combined limit. |
| `recent_usage_component` | UGX | Recent usage cap's weighted contribution. |
| `prior_exposure_component` | UGX | Prior exposure cap's weighted contribution. |
| `risk_component` | UGX | Risk cap's weighted contribution. |
| `combined_cap_before_risk_guardrail` | UGX | Weighted sum before applying the risk cap as a hard ceiling. |
| `risk_cap_binding` | 0 / 1 | 1 if the risk cap was lower than the weighted combination (risk is the constraining factor). |
| `combined_cap_after_risk_guardrail` | UGX | Combined cap after enforcing risk cap as upper bound. |
| `combined_cap_before_smoothing` | UGX | Cap before prior-limit smoothing was applied. |
| `prior_limit_smoothing_applied` | 0 / 1 | 1 if a prior limit existed and smoothing changed the cap (max +/-25% swing from prior limit). |
| `combined_top_driver` | Text | Which of the four stages contributed most. |
| `combined_reason` | Text | Combination logic applied (see Reason Codes above). |

---

## Stage 6: Policy Adjustments

Applies business policy rules on top of the combined cap.

| Column | Type | Description |
|---|---|---|
| `policy_cap` | UGX | Limit after applying risk-tier multiplier and floor overrides. |
| `risk_tier` | tier_1 to tier_4 | Risk classification (see table above). |
| `policy_multiplier` | 0.40 - 1.00 | The tier-based fraction of combined_cap. |
| `is_proven_good_borrower` | 0 / 1 | 1 if agent has 3+ loans, 90%+ on-time rate, and <=5% default rate. These agents get a floor override. |
| `proven_good_floor` | UGX | Minimum limit for proven-good borrowers (85% of combined_cap). Only applies if is_proven_good_borrower=1. |
| `policy_floor_applied` | 0 / 1 | 1 if the proven-good floor raised the limit above what the tier multiplier would have given. |
| `active_floor_eligible` | 0 / 1 | 1 if agent has any recent lending activity (disbursements + repayments >= 1 UGX). |
| `active_floor_applied` | 0 / 1 | 1 if the minimum active-borrower floor (500 UGX) raised the limit. |
| `policy_reason` | Text | Which policy rule determined the outcome (see Reason Codes above). |

---

## Agent Tier Ceilings

Each agent has a hard ceiling based on their profile tier. No limit can exceed this ceiling regardless of other calculations.

| Agent Tier | Ceiling (UGX) | Multiplier |
|---|---|---|
| Diamond | 1,000,000 | 1.00 |
| Titanium | 750,000 | 0.75 |
| Platinum | 500,000 | 0.50 |
| Gold | 350,000 | 0.35 |
| Silver / Silver Class | 250,000 | 0.25 |
| Bronze | 100,000 | 0.10 |
| New Bronze / Unknown | 50,000 | 0.05 |

---

## Key Input Feature Columns (Passed Through in Output)

These are inputs to the engine that appear in the output when `keep_intermediate=True`.

| Column | Source | Description |
|---|---|---|
| `msisdn` | All sources | Agent phone number (unique identifier). |
| `snapshot_dt` | Loan summary query | Date the features were computed for. |
| `agent_profile` | Transaction capacity | Agent tier classification (Diamond, Gold, Bronze, etc.). |
| `agent_tier_ceiling_multiplier` | Transaction capacity | Numeric ceiling multiplier derived from agent_profile. |
| `total_loans` | Borrower limits | Lifetime number of loans taken by this agent. |
| `is_thin_file` | Computed | 1 if agent has fewer than 3 lifetime loans (new borrower). |
| `is_active_borrower` | Computed | 1 if agent has recent credit activity (disbursement or repayment in last month). |
| `operational_activity_flag` | Transaction capacity | 1 if agent has recent transaction activity (volume > 0 or customers > 0). |
| `is_peak_season_flag` | Computed | 1 if snapshot falls in a peak month (Jan, Aug, Sep, Dec in Uganda). |
| `disbursement_val_1m` | Loan summary query | Total XtraFloat disbursement value in the last 1 month (UGX). |
| `repayment_val_1m` | Loan summary query | Total XtraFloat repayment value in the last 1 month (UGX). |
| `penalties_1m` | Loan summary query | Number of XtraFloat penalty events in the last 1 month. |
| `on_time_repayment_rate` | Borrower limits | Fraction of loans repaid on time (within 24 hours). |
| `lifetime_default_rate` | Borrower limits | Fraction of lifetime loans that defaulted. |
