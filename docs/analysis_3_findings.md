# Analysis 3 Findings: Episode-Grain Capacity Challenger

**Status:** Frozen at commit `45cb964` on `claude/integrated-solution-analysis-atdfe6`.
**Next step:** Analysis 4 — Capacity Frontier Design (not started).

## Purpose and dataset

Analysis 3 asked whether the existing credit engine's assigned limits understate
true business capacity for high-quality agents, using historical loan-episode
data rather than a trained model, before committing to any particular modeling
approach.

Two earlier framings were tried and explicitly rejected before this one:
quantile regression on `actual_exposure_ugx` (abandoned — exposure is ~7
discrete loan sizes, not continuous), and a monotonic
`P(Good|Fundamentals,Exposure)` classifier on the monthly-snapshot research
table (blocked at "Gate 0" — that table's outcome columns are computed over
different, later loans than the exposure column it would be trained against).

**Dataset**: `loan_episode_capacity_dataset.csv`, one row per
`disbursement_fid`, built by `scripts/build_loan_episode_capacity_dataset.py`.
Each loan is matched via an as-of join (`merge_asof`, backward, no exact
matches) to the latest prior transaction-mart snapshot for that agent. That
snapshot date (`fundamentals_snapshot_date`) is treated as a **business-state
period identifier**, not a calendar-month bucket.

**Why loan-to-loan deltas were abandoned for a business-state-period design**:
an ad hoc diagnostic (`scripts/analyze_loan_frequency_for_snapshot_cadence.py`)
found a median consecutive-loan spacing of 1 day and 91.3% of consecutive loan
pairs within the same calendar month on the real loan-training file. Chaining
30-day rolling fundamentals loan-to-loan would mostly measure noise, not real
business change, for most of this population. The redesign nests loans
*within* the slower-moving business-state period instead.

**Evidence hierarchy** (none of it causal):
- **Level 1** (`analyze_business_state_exposure_performance.py`): cross-sectional — comparable measured business states, different exposure tiers, different agents.
- **Level 2** (`analyze_business_state_exposure_variation.py`, incl. Table C2): same agent + same measured business-state anchor, different exposure tiers.
- **Level 3** (`analyze_business_state_evolution.py`): same agent, business state evolves, exposure profile evolves across periods.

Real-data scale: 5.66M loan episodes → 526,805 agent-period units, 74,729
(14%) of which experienced ≥2 distinct exposure tiers under one anchor
(Level 2); 394,733 consecutive business-state-period transitions across
114,787 agents (Level 3, k=1).

## Evidence table

| Question | Evidence | Conclusion |
|---|---|---|
| Do fundamentals distinguish risk? | Level 3 | Yes, strongly — improving fundamentals (float/commission) are consistently associated with lower subsequent bad rates, holding the other's coarse direction fixed. |
| Does higher absolute exposure carry risk? | Table C + Table C2 + Level 3 | Yes, observationally — a within-agent/same-anchor tier increase is associated with higher subsequent bad rate across every one of 21 pooled tier pairs, and the pattern survives stratifying by float scale (Table C2). |
| Does business scale modify the exposure-intensity (`L/F`) relationship? | Table B | Yes — small-float agents show deterioration as EI rises; the largest-float agents show flat-to-improving outcomes. |
| Is exposure assignment endogenous? | Level 1 vs. Table C/C2 | Almost certainly material — Level 1's cross-sectional overlap looked permissive, but once the same agent/same anchor is held fixed, the direction reverses to "higher tier = worse." The two cannot both be the whole story unless assignment itself is informative. |
| Does assigned exposure always track business growth? | REG distribution + Level 3 | Not perfectly — at k=1, 34.2% of consecutive transitions show exposure growing materially slower than float (REG<0.8) vs. 29.1% materially faster (REG>1.2), with 36.7% roughly proportional. REG trends lower at larger ordinal gaps (k=1..5), but that population is increasingly selected (agents observed across more periods), so this is not proof limits increasingly fail to catch up. |
| Are F↑,L→ agents promising capacity-gap candidates? | Level 3 | Yes — the largest, best-powered cell in the direction grid (n=138,469 transitions, 84,705 agents), with subsequent bad rate as good as or better than otherwise-similar agents whose exposure did grow. |
| Does this prove they can safely take more exposure? | — | No — there is no counterfactual for what happens to this population if actually exposed further, and Table C/C2 specifically warn against assuming that is free. |
| Can historical assigned limit be used as a capacity target? | Full evidence (Level 1 vs. Table C/C2) | No — the reversal between the cross-sectional and within-agent reads is itself evidence that historical assignment is endogenous. |
| Is causal uplift from higher limits established? | — | No — every table in this analysis is observational; selection, calendar effects, and policy changes remain live explanations throughout. |

## Synthesis

Historical evidence supports treating business capacity and credit risk as
separate dimensions of the lending decision. Across consecutive
business-state periods, improving measured business fundamentals are
consistently associated with lower subsequent bad rates, while increasing
loan exposure is consistently associated with higher subsequent bad rates
conditional on the coarse direction of fundamentals. Cross-sectional and
within-agent analyses further indicate that historical exposure assignment is
endogenous, so assigned limits should not be used as a direct capacity
target. Agents whose fundamentals improve while exposure remains unchanged
are therefore treated as capacity-gap candidates, not as demonstrated
under-extended agents. The analysis identifies where a capacity challenger
should look, but does not establish the causal effect or safe magnitude of a
limit increase. The next step is to develop a conservative Capacity(F)
frontier that remains separate from the existing C3 continuous risk
multiplier, followed ultimately by prospective validation of incremental
exposure.

## Explicit non-claims

- **No causal estimate** of the effect of raising a limit on repayment risk.
- **F↑,L→ is a capacity-gap *candidate* population, not demonstrated
  under-extension** — no counterfactual exposure was observed for it.
- **The REG decline across larger `k` is not evidence that limits
  increasingly fail to catch up** — the `k=5` population is a selected
  subset (agents observed across more periods), not a comparable sample to
  `k=1`.
- **Historical assigned limit is not a valid capacity target** — Level 1 vs.
  Table C/C2's reversal shows assignment itself carries information,
  disqualifying it as ground truth for a capacity model.
- **No classifier, monotonic model, or tau was built or is implied** by this
  analysis — that path (`P(Good|F,L)`, maximize `L`) was explicitly
  considered and rejected as mixing capacity, risk, and historical selection
  in a way that would be difficult to govern.

## Architecture implication

This analysis gives empirical backing — not just an architectural
preference — for keeping a two-stage design:

```
L_candidate = Capacity(F) × RiskMultiplier(PD, L, capacity/exposure relationship)
```

The `RiskMultiplier` side already exists in this repo, in shadow mode:
`extrafloat/engine/extrafloat_shadow_risk_multiplier.py` (commit `cd46cc5`,
"Add shadow continuous risk multiplier (C3 hybrid), computed read-only
alongside the live 4-tier policy"), with its config hook, calibration script
(`scripts/fit_shadow_risk_calibration.py`), and test coverage already in
place. The open piece motivated by Analysis 3 is the `Capacity(F)` frontier
— to be designed **without** using historical assigned limit as its target,
per the evidence above.

## Next step

**Analysis 4 — Capacity Frontier Design**: develop `Capacity(F)` independent
of historical assigned limit, to sit alongside the existing C3 continuous
`RiskMultiplier`, followed by a controlled limit-increase pilot for
prospective validation rather than extracting a final limit purely from this
observational historical data.

## Scripts and commits (reproducibility trail)

| Script | Role | Key commits |
|---|---|---|
| `scripts/build_loan_episode_capacity_dataset.py` | Builds the episode-grain dataset; as-of join to the business-state-period anchor | `ec3ddbf` (initial), `a816f91` (fixed int64 `tbl_dt` YYYYMMDD dates collapsing to 1970-01-01) |
| `scripts/check_loan_episode_dataset_integrity.py` | Integrity diagnostics; business-state coverage distributions | `ec3ddbf` (initial), `73f0976` (business-state-period coverage diagnostics) |
| `scripts/analyze_loan_frequency_for_snapshot_cadence.py` | Ad hoc diagnostic that triggered the cadence-gate pivot (91.3% same-calendar-month finding) | `3f92e2d` (creation), `cf032e6` (OOM fix) |
| `scripts/analyze_business_state_exposure_performance.py` | **Level 1** — business state × exposure → performance | `5705cdc` (as `analyze_episode_exposure_scale_overlap.py`), `265622a` (unknown-exposure reporting + tier-leak fix) |
| `scripts/analyze_business_state_exposure_variation.py` | **Level 2** — same agent/same anchor, different exposure (Tables A/B/C, plus float-stratified Table C2) | `003c56b` (creation), `8e1eaed` (Table B label-mismatch fix), `265622a` (unknown-exposure + tier restriction), `bfd808e` (vectorization), `44311fa` (Table C2 addition), `45cb964` (Table C2 tier-dtype fix) |
| `scripts/analyze_business_state_evolution.py` | **Level 3** — same agent, business state evolves, exposure evolves | `df1bf9a` (creation), `e211236` (vectorization) |
| `archive/analyze_episode_agent_transitions.py`, `archive/analyze_episode_exposure_escalation_matrix.py` | Original (loan-to-loan) Deliverables 4/5 — superseded by the cadence-gate pivot, retained for audit | `13d4bab`, `1a8a4bd` (creation); `421b220` (archived with deprecation headers citing the cadence finding) |
| `extrafloat/engine/extrafloat_shadow_risk_multiplier.py` | **C3 RiskMultiplier** (shadow mode) — the risk-side component this analysis motivates keeping separate from `Capacity(F)` | `cd46cc5` |
