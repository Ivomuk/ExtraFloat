# Diamond A x Capacity(F) Reconciliation, and the Capacity(F) vs. C3 Decomposition

**Status:** Frozen on `claude/integrated-solution-analysis-atdfe6`, against the real controlled shadow run `run_id = 2026-10-13T14:52:11.085567Z` (the Stage 4C integration baseline frozen in `docs/stage_4c_prospective_validation_schema_design.md` Section 9 / the Decision Log at that run_id).
**Next step:** Prospective outcome maturation and, eventually, a controlled-exposure (Tier B) test. No further retrospective slicing of this result is planned — all three questions below (Sections 2, 3, 6) were opened, answered, and closed within this session.

## 1. Objective

Three related, but distinct, retrospective questions were asked of the frozen Stage 4C integration baseline, using the independent `Capacity(F; k=0.05) x M_C3` challenger already logged in the Decision Log, without touching `k`, `Capacity(F)`, C3, or any already-frozen artifact:

1. **Diamond A reconciliation** — do agents who historically received disbursements >= 2x their engine-proposed `assigned_limit` ("Diamond A") show the independent challenger disproportionately recommending *more* exposure for them, corroborating the earlier "possible underfunding" hypothesis with a methodologically independent signal?
2. **Capacity(F) vs. C3 decomposition** — for the general population, is the challenger's generally-lower recommendation coming from `Capacity(F)` itself, or from the subsequent C3 risk adjustment?
3. **Scale decomposition of Capacity(F) conservatism** (Section 6) — is the 85.18% `CAPACITY_BELOW_CURRENT` result for non-Diamond-A disbursed agents uniform across the population, or concentrated among smaller businesses?

All three are answered below. None establishes that acting on the challenger (raising or lowering any live limit) would be safe or economically optimal — that remains explicitly deferred to prospective monitoring and an eventual controlled-exposure test.

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

All three "why" questions opened in this session (why does Diamond A's historical behavior corroborate the challenger; why is the challenger generally lower for ordinary borrowers; why does that general-population conservatism have the shape it does) are now closed. The open question is no longer retrospective — it is whether the independently-estimated capacity (lower for most agents, higher for Diamond A) is better supported by **subsequent** performance and economics than the incumbent allocation. That is the role of continued prospective shadow-run monitoring and, eventually, a controlled-exposure (Tier B) test — not another historical cut of this same baseline run.

## 6. Scale Decomposition of Capacity(F) Conservatism

### 6.1 Question

Section 3 established that the general population's 85.18% `CAPACITY_BELOW_CURRENT` result is driven by raw `Capacity(F)` itself, not by the subsequent C3 adjustment. It left open *why* raw `Capacity(F)` comes in low for so much of the population: is the frozen `k=0.05` scale uniformly conservative across the whole population, or is the shortfall concentrated in a particular part of it?

### 6.2 Method

`R_capacity = capacity_shadow_k0_05 / l_current`, already defined in Section 3, examined two ways on the same frozen baseline:

1. **Magnitude distribution** — 8 named bands (`<0.25` through `>2.00`, pivoting on the same 0.90/1.10 agreement convention used in Section 2) plus percentiles, to check whether the 85.18% BELOW figure reflects near-misses (R_capacity just under 1.0) or a wide spread including severe shortfalls.
2. **Decile breakdown by business scale** — `float_activity_value_1m` and `commission` deciled independently (never combined), with median/p25/p75 `R_capacity` and the %BELOW/%AT/%ABOVE split reported per decile, to test whether `R_capacity` trends toward or above 1.0 as business scale increases.

Implemented as `summarize_r_capacity_distribution()` and `r_capacity_by_scale_decile()` in the same reconciliation script (commit `6dfc80b`). This is explicitly a diagnostic of the already-frozen `k=0.05`, not a re-tuning exercise — no result here is grounds for changing `k`, `Capacity(F)`, or C3.

### 6.3 Result

**Magnitude distribution, non-Diamond-A disbursed:** median `R_capacity` = 0.423; 22.50% of agents are below 0.25x and a further 35.68% are between 0.25x and 0.50x. The 85.18% BELOW figure is emphatically not a population sitting just under the 1.0 boundary — most of it is well below.

**Decile breakdown, non-Diamond-A disbursed, `float_activity_value_1m`:**

| Decile | median R_capacity | % BELOW 0.90 | % ABOVE 1.10 |
|---|---|---|---|
| D1 | 0.168 | 100.00 | 0.00 |
| D4 | 0.286 | 99.39 | 0.11 |
| D7 | 0.535 | 93.58 | 1.02 |
| D8 | 0.693 | 82.02 | 7.50 |
| D9 | 0.936 | 43.03 | 24.45 |
| D10 | 1.484 | 2.28 | 87.04 |

`commission` deciles independently reproduce the same shape: D10 median `R_capacity` = 1.420, 7.03% BELOW, 77.97% ABOVE.

**Cross-check within Diamond A, `float_activity_value_1m`:** D1 median `R_capacity` = 0.466, 96.63% BELOW; by D2 the median is already 1.164; from D5 onward the decile is essentially entirely ABOVE 1.10. The same scale effect holds inside Diamond A, which argues the effect is driven by business scale rather than by the Diamond A label itself.

### 6.4 Frozen conclusion

> **Capacity(F) conservatism is strongly scale-dependent rather than uniform.** Although 85.18% of evaluable non-Diamond-A disbursed agents have raw Capacity(F) below their current live limit, the disagreement is concentrated among smaller businesses. `R_capacity = Capacity(F) / l_current` increases strongly with both Float Activity and Commission when each fundamental is examined independently. In the highest Float Activity decile, median `R_capacity` is 1.484 and 87.04% of agents are above the 1.10 agreement boundary; in the highest Commission decile, median `R_capacity` is 1.420 and 77.97% are above 1.10. Conversely, the lower deciles are overwhelmingly below current limits.
>
> The result therefore does not support characterizing `k=0.05` Capacity(F) as uniformly conservative. Rather, Capacity(F) generates substantially lower capacity estimates for small/low-activity businesses, converges toward current limits as business scale increases, and frequently exceeds current limits for the largest businesses. This pattern is consistent with the incumbent mechanism containing a relatively high floor/base allocation that does not decline proportionally with business scale, although the analysis does not prove that mechanism or exclude omitted capacity factors correlated with scale — the decile analysis cannot distinguish an incumbent-floor effect from omitted capacity variables that happen to matter disproportionately for smaller agents.
>
> This also explains the Diamond A reconciliation (Section 2): Diamond A disproportionately identifies businesses for which independently measured economic scale supports current-or-higher capacity, while small Diamond A agents themselves remain below current limits. The Diamond A label is therefore not what mechanically causes the higher challenger recommendations; the underlying business fundamentals are.
>
> C3 is not the explanation for the general-population result either (Section 3): for non-Diamond-A disbursed agents, 85.18% are already below current before C3 versus 88.09% afterwards; only 2.91% occupy the specific `CAPACITY_ABOVE_RISK_REDUCED_BELOW_CURRENT` state.

```
Capacity(F) general conservatism:           EXPLAINED -- strongly scale-dependent
Global k=0.05 failure:                      NOT SUPPORTED by this diagnostic
Incumbent floor/base-allocation explanation: PLAUSIBLE / NOT PROVEN
Omitted capacity variables:                  NOT RULED OUT
C3 as primary cause:                         NOT SUPPORTED
Change k, Capacity(F), or C3:                NO
Further retrospective slicing:               STOP
```
