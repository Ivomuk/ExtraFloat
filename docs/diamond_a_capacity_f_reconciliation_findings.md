# Diamond A x Capacity(F) Reconciliation, and the Capacity(F) vs. C3 Decomposition

**Status:** Frozen on `claude/integrated-solution-analysis-atdfe6`, against the real controlled shadow run `run_id = 2026-10-13T14:52:11.085567Z` (the Stage 4C integration baseline frozen in `docs/stage_4c_prospective_validation_schema_design.md` Section 9 / the Decision Log at that run_id).
**Next step:** Prospective outcome maturation and, eventually, a controlled-exposure (Tier B) test. No further retrospective slicing of this result is planned — both questions below were opened, answered, and closed within this session.

## 1. Objective

Two related, but distinct, retrospective questions were asked of the frozen Stage 4C integration baseline, using the independent `Capacity(F; k=0.05) x M_C3` challenger already logged in the Decision Log, without touching `k`, `Capacity(F)`, C3, or any already-frozen artifact:

1. **Diamond A reconciliation** — do agents who historically received disbursements >= 2x their engine-proposed `assigned_limit` ("Diamond A") show the independent challenger disproportionately recommending *more* exposure for them, corroborating the earlier "possible underfunding" hypothesis with a methodologically independent signal?
2. **Capacity(F) vs. C3 decomposition** — for the general population, is the challenger's generally-lower recommendation coming from `Capacity(F)` itself, or from the subsequent C3 risk adjustment?

Both are answered below. Neither establishes that acting on the challenger (raising or lowering any live limit) would be safe or economically optimal — that remains explicitly deferred to prospective monitoring and an eventual controlled-exposure test.

## 2. Diamond A x Capacity(F) Reconciliation

### 2.1 Definitions

- **Diamond A**: an agent whose `disbursed_amount` reached >= 200% of their own engine-proposed `assigned_limit` in at least one month, per `check_live_shadow_vs_category_limit.py`'s `pct_of_live_disbursed` column (= `disbursed_amount / assigned_limit`). 1,819 agents in the real `live_shadow_vs_category_limit.csv` file.
- **Non-Diamond-A disbursed (direct complement)**: agents with a real disbursement record who never reached the 200% threshold. 74,014 agents in the same file (1,819 + 74,014 = 75,833 agents with any disbursement record — exact arithmetic, verified by the reconciliation script's own printed check).
- **R_i = L_full_shadow_k0_05_base / l_current** — read directly from the Decision Log's own `disagreement_ratio_shadow_base` column, never recomputed independently.
- **R-bands**, reusing this repo's established "90-110% = agreement" convention (`check_monthly_summary_vs_assigned_limit.py`, restated independently each time): `ABOVE` (R >= 1.10, "challenger supports MORE exposure"), `AT` (0.90 <= R < 1.10, "broadly supports existing exposure"), `BELOW` (R < 0.90, "fundamentals do not support current exposure under frozen Capacity(F)+C3").

Implemented in `scripts/analyze_diamond_a_capacity_f_reconciliation.py` (commits `6c4cd02`, `dd38381`, `1f98848`, `df66d8b`).

### 2.2 Result: Diamond A vs. its direct complement

| Population | BELOW | AT | ABOVE | n (capacity_estimable=True, R defined) |
|---|---|---|---|---|
| Diamond A | 24.39% (432) | 9.60% (170) | **66.01%** (1,169) | 1,771 |
| Non-Diamond-A disbursed (direct complement) | 85.29% (62,820) | 5.09% (3,749) | **9.62%** (7,083) | 73,652 |
| OVERALL (full scored population, for continuity) | 85.62% | 4.70% | 9.68% | 131,188 |

**Lift and percentage-point difference (ABOVE band), Diamond A vs. direct complement:**

```
P(ABOVE | Diamond A)                        = 66.01%
P(ABOVE | Non-Diamond-A disbursed)           =  9.62%
Lift                                         =  6.86x
pp difference                                = +56.39
```

Restricting the comparator from the entire scored population (9.68% ABOVE) to only agents with a real disbursement history (9.62% ABOVE) barely moves the baseline — confirming the original 6.82x lift against the full population was not an artifact of comparing active borrowers against agents who may never have borrowed. The direct-complement comparison (6.86x, +56.39pp) is the cleaner, final number.

### 2.3 Severity sub-bucketing and the 58-agent 300-500% tail

Breaking Diamond A down by how far over 200% an agent's historical disbursement reached (purely descriptive bucketing, not a new business threshold):

| Severity | n | % of Diamond A | ABOVE | BELOW |
|---|---|---|---|---|
| 200-300% | 1,761 | 96.8% | 66.74% | 23.87% |
| 300-500% | 58 | 3.2% | 37.78% | 44.44% |
| 500%+ | 0 | 0% | - | - |

The dominant cohort (96.8% of Diamond A) is intact at the headline result. The small 300-500% tail shows a different, weaker pattern — investigated via a row-level decomposition (`--severity-detail-band`) rather than assumed:

- All 13 non-estimable agents in this band share `float_activity_value_1m = 0.0` with real, substantial `commission` (UGX 277K-1.3M) and are all `tier_4` — a data-quality observation (commission without float activity is atypical for a mobile-money agent), not a Capacity(F) defect.
- Among the 45 estimable agents, the BELOW/ABOVE split is explained by `risk_tier` composition (tier_4's fixed C3 floor of 0.40 dominates BELOW; tier_3's smaller haircut dominates ABOVE) combined with `float_activity_value_1m` scale (ABOVE rows have float in the tens-to-hundreds of millions; BELOW rows are mostly thousands-to-low-millions).

This is recorded as explained heterogeneity, not a reversal of the headline finding.

### 2.4 Frozen conclusion

> Diamond A agents — defined by historical disbursement reaching at least 200% of their assigned limit — are substantially more likely than other disbursed agents to have their current live limit fall below the independently derived Capacity(F)+C3 recommendation. The ABOVE rate is 66.01% for Diamond A versus 9.62% for its direct non-Diamond-A disbursed complement, a 6.86x lift (+56.39pp). Borrower-history composition does not explain this (9.68% -> 9.62% when the comparator is tightened to the same eligibility universe). This materially supports the hypothesis that the historical Diamond A phenomenon captures genuine business-capacity differences rather than merely reflecting the incumbent allocation process. It does not establish that increasing exposure is causally safe.

```
Analytical design:                       FROZEN
Diamond A definition:                    FROZEN
Capacity(F), k=0.05:                     FROZEN
C3:                                      FROZEN
Retrospective interpretation:            SUPPORTED
Claim that Diamond A should receive higher limits:  NOT YET ESTABLISHED
Causal safety of additional exposure:    REQUIRES Tier B controlled exposure
```

## 3. Capacity(F) vs. C3 Decomposition

### 3.1 Question and method

`disagreement_ratio_shadow_base` (R_full) alone cannot distinguish whether a BELOW-current result comes from `Capacity(F)` itself or from the subsequent C3 risk adjustment. Answered by cross-tabbing, at the pivot of 1.0 (a different, simpler question than the 0.90/1.10 agreement band above):

```
R_capacity = capacity_shadow_k0_05 / l_current      (raw business capacity vs. current limit)
R_full     = l_full_shadow_k0_05_base / l_current   (final challenger recommendation vs. current limit)
```

Implemented as `decomposition_crosstab()` in the same reconciliation script (commit `df66d8b`); the same decomposition was then added natively to the Stage 4C Decision Log schema itself (`r_capacity_shadow_k0_05_base`, `challenger_decomposition`; commit `802d98d`; see Section 9 of `docs/stage_4c_prospective_validation_schema_design.md`) so future runs carry it without a derived script.

### 3.2 Result

| Population | Capacity(F) < current (`CAPACITY_BELOW_CURRENT`) | Capacity >= current but C3 pushes below (`CAPACITY_ABOVE_RISK_REDUCED_BELOW_CURRENT`) | Full >= current (`FULL_ABOVE_CURRENT`) | Mechanically impossible |
|---|---|---|---|---|
| Diamond A | 13.78% | 15.36% | 70.86% | 0.00% |
| Non-Diamond-A disbursed | **85.18%** | 2.91% | 11.91% | 0.00% |
| OVERALL | 84.75% | 3.47% | 11.78% | 0.00% |

The `MECHANICALLY IMPOSSIBLE` cell (`R_capacity < 1` and `R_full >= 1`, which cannot occur while `shadow_multiplier_base <= 1`) is exactly 0 across all three populations — a clean confirmation of invariant #12 holding on real data, not just in synthetic fixtures.

### 3.3 Frozen conclusion

> For non-Diamond-A disbursed agents, the lower recommendations produced by the full challenger are overwhelmingly attributable to Capacity(F) itself rather than the C3 risk adjustment. At frozen k=0.05, 85.18% are below their current live limit before C3, compared with 88.09% after C3. C3 causes only an additional 2.91% of agents to cross from at/above current capacity to below current. This establishes that the challenger's generally lower recommendations are fundamentally capacity-driven, while leaving open whether those lower recommendations are economically preferable to the incumbent limits.

Precise wording matters here: C3 is **not responsible for putting** the 85.18% below the current limit (they were already below at raw `Capacity(F)`) — it is not that C3 is *irrelevant* to them, since it can still reduce their recommendation further.

The picture is genuinely different for Diamond A: raw business capacity is at or above the current limit for 86.22% of evaluable agents (70.86% + 15.36%), and C3 is responsible for a meaningful share (15.36 of the 29.14 points, i.e. over half) of their smaller BELOW population. This gives a clean conceptual separation, not two unrelated findings:

```
Business size asks:  "How much could this business support?"       -- Capacity(F)
C3 asks:              "How much of that capacity should we expose ourselves to?"  -- risk adjustment
```

## 4. What This Does Not Establish

- That `Capacity(F)` at `k=0.05` is more or less "correct" than the incumbent mechanism. `k=0.05` was deliberately never fitted to reproduce incumbent-assigned limits (Stage 4B); disagreement with the incumbent's scale was expected by design. A large fraction of the population being lower under the challenger shows the two systems operate on a substantially different scale — it does not by itself show which one is closer to the economically correct capacity.
- That raising Diamond A's limits, or lowering the general population's limits, toward the challenger's recommendation would be safe. Both results are observational triangulation between two retrospective signals, not a causal estimate.
- Nothing here changes `k=0.05`, `Capacity(F)`'s functional form or weights, or C3's policy parameters. No retrospective re-tuning was performed or is planned as a result of either finding.

## 5. What's Next

Both "why" questions opened in this session (why does Diamond A's historical behavior corroborate the challenger; why is the challenger generally lower for ordinary borrowers) are now closed. The open question is no longer retrospective — it is whether the independently-estimated capacity (lower for most agents, higher for Diamond A) is better supported by **subsequent** performance and economics than the incumbent allocation. That is the role of continued prospective shadow-run monitoring and, eventually, a controlled-exposure (Tier B) test — not another historical cut of this same baseline run.
