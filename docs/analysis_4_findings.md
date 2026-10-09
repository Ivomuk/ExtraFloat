# Analysis 4 — Business-State Capacity Frontier Findings

**Status:** Frozen at commit `a268c2c` on `claude/integrated-solution-analysis-atdfe6`.
**Next step:** Deliverable 2 — design the frozen production `Capacity(F)` mapping (not started).

## 1. Objective

Analysis 4 evaluates whether historical loan episodes provide evidence for a fundamentals-based exposure frontier that can later inform a separate `Capacity(F)` component of the credit decision engine.

The analysis does **not** estimate a causal effect of increasing limits and does **not** treat historical assigned limits as true business capacity. Instead, it asks:

> For agents in comparable measured business states, how far up the historical exposure ladder is there sufficient paired evidence before subsequent same-loan bad performance deteriorates beyond a specified tolerance?

The resulting boundaries are therefore referred to as **historically-supported exposure frontiers**, not production capacity limits.

## 2. Methodology

The analysis uses the point-in-time-aligned loan-episode dataset established in Analysis 3. Business fundamentals are measured strictly before the target loan, and subsequent performance is measured on the same loan.

Float activity and commission are evaluated independently as business-fundamental segmentation variables.

Within each fundamentals decile, the lowest sufficiently populated exposure tier is selected as the `reference_tier`. Candidate exposure tiers are compared against this fixed reference using same-agent, same-business-state paired observations.

A candidate is classified as:

- **supported** when its loan-weighted bad-rate deterioration relative to the reference tier does not exceed the selected tolerance;
- **risk_breach** when the deterioration exceeds the tolerance;
- **evidence_gap** when fewer than 10 paired agent-period units are available;
- **not_evaluated_beyond_breach** when a lower candidate tier has already produced a risk breach.

Evidence is labelled **robust** at 100 or more paired units and **exploratory** below 100 paired units.

Risk tolerances of 2, 3, 4 and 5 percentage points were evaluated as sensitivity parameters. These tolerances are not estimated risk appetite and were not expanded in response to observed results.

## 3. Float Frontier Findings

The float-based frontier shows material heterogeneity across business-scale bands.

Several bands exhibit large and well-powered deterioration at the first exposure increase from UGX 50,000 to UGX 100,000:

| Float band | Paired N | Δ bad rate, 50K→100K | Interpretation |
|---|---:|---:|---|
| D2 | 4,399 | +5.09pp | Persistent robust breach through 5pp tolerance |
| D3 | 4,712 | +5.18pp | Persistent robust breach through 5pp tolerance |
| D7 | 860 | +5.14pp | Persistent robust breach through 5pp tolerance |
| D8 | 526 | +1.19pp | Robustly supported first increase |
| D9 | 261 | +3.43pp | Tolerance-dependent |
| D10 | 104 | +1.03pp | Robustly supported first increase |

D8 provides one of the clearest examples of a climbing frontier. Relative to its 50K reference tier:

- 100K: +1.19pp, N=526, robust;
- 250K: +2.79pp, N=661, robust;
- 350K: +3.78pp, N=232, robust;
- 500K: +1.37pp, N=86, exploratory;
- 750K and 1M: insufficient paired evidence.

Consequently, at tolerances of 4–5pp, the observed evidence extends as high as 500K before reaching an evidence gap. The final step is exploratory rather than robust.

D10 supports 100K and 250K at tolerances of at least 3pp before encountering a +6.31pp deterioration at 350K. That 350K comparison has N=89 and is therefore exploratory rather than robust.

The overall float result is **not monotonic in decile rank**. In particular, D7 exhibits a persistent first-step breach while D8 supports substantially greater exposure. Business scale measured by float alone therefore does not define a simple monotonic safe-exposure function.

## 4. Commission Frontier Findings

Commission independently produces a similarly heterogeneous frontier pattern.

Persistently locked first-step bands include:

| Commission band | Paired N | Δ bad rate, 50K→100K | Interpretation |
|---|---:|---:|---|
| D4 | 3,579 | +5.80pp | Persistent robust breach |
| D5 | 2,224 | +5.75pp | Persistent robust breach |
| D10 | 100 | +8.34pp | Persistent robust breach at threshold N |

Commission D10 produces the largest observed first-step deterioration across the reviewed float and commission results. However, the paired-unit distribution is concentrated: 22% of units show positive deterioration, 64% are unchanged and 14% show negative deterioration. Its N=100 also lies exactly at the pre-specified robust-evidence threshold. It should therefore be described as **threshold-robust but distributionally concentrated**.

Other commission bands show tolerance-dependent progression.

Commission D7 has a +4.66pp first-step deterioration at 100K (N=721). It remains breached through tolerances of 2–4pp but becomes supported at 5pp, where the next candidate, 250K, breaches at +5.29pp (N=1,062).

Commission D8 has +3.54pp at 100K (N=479), supporting 100K from a 4pp tolerance. At 5pp, 250K becomes supported (+4.97pp, N=678), while 350K becomes the next breach (+6.02pp, N=281).

Commission D9 shows one of the clearer climbing patterns:

- 100K: +0.95pp, N=261;
- 250K: +3.54pp, N=304;
- 350K: +2.61pp, N=334;
- 500K: +6.80pp, N=194.

At tolerances of 4–5pp, the historically-supported frontier therefore reaches 350K before encountering a robust deterioration at 500K.

Commission D2 provides particularly strong evidence around the tolerance boundary. The 100K candidate deteriorates by +4.05pp with N=5,660 paired units. It breaches at tolerances of 2–4pp but becomes supported at 5pp. The 250K candidate similarly shows +4.60pp with N=179. No paired evidence exists at 350K or above.

These results demonstrate why the tolerance sweep should not be expanded simply because several observed differences lie close to 5pp. Doing so would effectively select the tolerance from the historical results rather than use tolerance as an externally specified risk-appetite parameter.

## 5. Float–Commission Relationship

A separate agent-period support diagnostic was conducted to determine whether float and commission represent substantially independent business-capacity dimensions.

The diagnostic covers 515,444 agent-period units with both fundamentals defined.

Results show:

- Spearman correlation of raw fundamentals: **0.879**;
- Spearman correlation of decile ranks: **0.870**;
- exact decile agreement: **37.7%**;
- within ±1 decile: **76.7%**;
- separation of at least three deciles: **8.4%**.

The joint count matrix is strongly diagonal. Float and commission therefore appear to be predominantly alternative measures of a common underlying business-scale construct rather than independent capacity dimensions.

This does not imply complete redundancy. Approximately 8.4% of agent-period observations are separated by at least three deciles, indicating a meaningful minority for which the two measures materially re-rank business state.

## 6. The Float-D7 versus Float-D8 Question

The joint distribution does not explain the substantial difference between the float-D7 and float-D8 frontiers.

For float-D7, commission membership is concentrated primarily in:

- D6: 19.7%;
- D7: 29.4%;
- D8: 23.5%;
- D9: 7.9%.

For float-D8:

- D6: 9.2%;
- D7: 20.1%;
- D8: 34.1%;
- D9: 24.6%.

Although float-D8 is shifted upward on the commission axis, the distributions overlap substantially. Float-D7 does not predominantly consist of agents classified into commission's strongly locked D4/D5 bands: only 15.2% fall in D4+D5, compared with 6.4% for float-D8.

Commission therefore does not provide a simple explanation for why float-D7 shows a persistent first-step breach while float-D8 supports materially greater exposure.

The D7/D8 discontinuity remains an **open empirical issue**. It should not be smoothed away or given a causal explanation from the current evidence.

## 7. Largest-Scale Band Caveat

Commission-D10 shows a +8.34pp first-step deterioration, while the upper float bands also eventually exhibit exposure deterioration.

However, these should not be described as independent confirmations.

Among commission-D10 agent-periods:

- 72.3% are also float-D10;
- 21.5% are float-D9.

Thus 93.8% of commission-D10 lies within the top two float deciles. The high-scale findings on the two fundamentals substantially concern the same underlying population.

The appropriate conclusion is therefore:

> High measured business scale alone does not imply unconstrained exposure capacity.

It is not appropriate to claim that float and commission independently prove this proposition.

## 8. Combined Interpretation

Analysis 4 provides four main findings.

First, **historically-supported exposure is heterogeneous**. There is no universal exposure ladder that can be applied solely from business-scale rank.

Second, **larger measured business scale does not monotonically imply greater historically-supported exposure**. Both low/middle and high fundamentals bands can exhibit material deterioration following higher exposure.

Third, **float and commission are strongly but incompletely redundant**. Their high correlation and 76.7% within-±1-decile agreement argue against immediately constructing a high-dimensional joint frontier. The 8.4% materially re-ranked tail remains relevant for future refinement.

Fourth, the results reinforce the separation between **capacity** and **risk**. Fundamentals contain useful information about the business's ability to support exposure, but historical exposure increases are themselves associated with additional repayment risk. The capacity component should therefore not replace the existing risk adjustment.

Conceptually:

```
L_candidate = Capacity(F) x RiskMultiplier
```

where `Capacity(F)` represents a fundamentals-based business-capacity estimate and the risk multiplier remains responsible for borrower credit risk.

## 9. What the Analysis Does Not Establish

The analysis does **not** establish that:

- historical assigned limits equal true business capacity;
- a supported historical tier can safely be assigned prospectively;
- increasing an agent's limit causes no additional risk;
- agents currently below a frontier are necessarily under-extended;
- float and commission should simply be averaged, maximized or minimized to produce a production capacity;
- the observed frontier should be copied directly into the live engine.

Historical exposure assignment remains endogenous. Even same-agent, same-business-state comparisons can contain unobserved time-varying information used by the historical lending process.

The frontier therefore identifies **historical support for candidate exposure**, not causal safe capacity.

## 10. Decision and Next Step

Analysis 4 descriptive frontier discovery is now considered **frozen**.

No 10x10 float x commission outcome frontier is recommended at this stage. The fundamentals are sufficiently correlated that such segmentation would substantially fragment paired support without clear evidence that the additional dimensionality is necessary.

The next stage should design the frozen production mapping:

```
F -> Capacity(F)
```

while preserving three principles established by the analysis:

1. do not train capacity directly against historical assigned limits;
2. do not assume capacity must increase monotonically with a single fundamentals decile merely because business scale increases;
3. keep the capacity estimate separate from the existing C3 credit-risk adjustment.

The historical frontier should be used as a **constraint and validation surface** for candidate capacity mappings rather than copied mechanically into production.

Any actual limit increases derived from the resulting challenger should first be evaluated in shadow mode and subsequently through a controlled prospective pilot before causal claims about safe incremental exposure are made.

### Open items retained for future work

The following are intentionally left unresolved rather than subjected to additional descriptive slicing:

- the float-D7 versus float-D8 frontier discontinuity;
- the economically meaningful 8.4% of agent-periods substantially re-ranked by float and commission;
- the appropriate method for combining highly correlated fundamentals into a stable production `Capacity(F)`;
- the prospective risk tolerance and economics required to determine whether a historically-supported exposure increase is commercially acceptable.

These questions belong to capacity-function design and prospective validation rather than further retrospective frontier mining.

## 11. Scripts and commits (reproducibility trail)

| Script | Role | Key commits |
|---|---|---|
| `scripts/derive_capacity_frontier_from_business_state.py` | Deliverable 1 -- derives the historically-supported exposure frontier (per fundamental, reference-tier walk, tolerance sweep, stability summary) | `f762323` (creation), `6905745` (stopped dumping the full classification table to stdout by default) |
| `scripts/check_float_commission_band_overlap.py` | Float x commission agent-period support-structure diagnostic (counts, both conditional distributions, Spearman/decile-agreement stats) -- no outcomes, no exposure tiers, no modeling | `d15d245` (creation), `a268c2c` (added the reverse conditional and rank-agreement diagnostics) |
| `docs/analysis_3_findings.md` | The frozen Analysis 3 findings this analysis builds on (validated episode dataset, Level 1/2/3 evidence hierarchy, Table C2) | `0de2e02` |

Real-data figures throughout this document come from a single run reported interactively during this session, not from a committed output file; the CSVs backing it (`capacity_frontier_*`, `float_commission_overlap_*`) were generated locally by the user and are not checked into the repository.
