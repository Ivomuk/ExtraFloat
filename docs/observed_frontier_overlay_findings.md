# Capacity(F) vs. capacity_cap vs. the Observed Exposure Frontier

**Status:** Frozen on `claude/integrated-solution-analysis-atdfe6`, against the real frozen Stage 4C baseline (`run_id = 2026-10-13T14:52:11.085567Z`) and Analysis 4's already-frozen historically-supported exposure frontier (`docs/analysis_4_findings.md`).
**Next step:** A prospective Diamond-A Capacity Challenger pilot (designed in Section 7 below; not yet implemented or run). No further retrospective slicing (e.g. of the D1/D2 undershoot) is planned.

## 1. Objective

Analysis 4 established an empirical, outcome-based historically-supported exposure frontier per float/commission band, as a constraint-and-validation surface, never a production target. `compare_capacity_f_vs_incumbent_cap_layers.py` separately established that Capacity(F) and the incumbent's own fundamentals-based `capacity_cap` disagree substantially (Capacity(F) below `capacity_cap` for 91.4% of the overall population), and that this disagreement originates at the fundamental-capacity stage, not downstream in the incumbent engine's usage/risk/smoothing/policy layers.

This analysis asks the natural follow-up: of the two fundamentals-based capacity calculations that disagree with each other, which one is closer to what historical loan outcomes actually support? Implemented in `scripts/overlay_capacity_f_vs_observed_exposure_frontier.py`, which overlays three curves per float/commission band (independently, never combined, per Analysis 4's convention): the observed frontier, Capacity(F) at the frozen k=0.05, and `capacity_cap`.

Never recomputes `capacity_shadow_k0_05`, `capacity_cap`, or the frontier itself. Purely observational. Does not touch k, Capacity(F), `capacity_cap`, or C3.

## 2. Result

### 2.1 capacity_cap exceeds the frontier by substantially larger multiples than Capacity(F) does

| Band | Frontier (tol=3pp) | Capacity(F) | ratio | capacity_cap | ratio |
|---|---|---|---|---|---|
| D5 | 50,000 | 105,756 | 2.12x EXCEEDS | 422,337 | **8.45x EXCEEDS** |
| D7 | 50,000 | 183,251 | 3.67x EXCEEDS | 441,642 | **8.83x EXCEEDS** |

Across the full tolerance sweep (2/3/4/5pp) and all 10 float bands, `capacity_cap` lands in `EXCEEDS` in roughly 37 of 40 (band x tolerance) cells, frequently by 4-12x. Capacity(F) lands in `EXCEEDS` in roughly 25 of 40 cells, with most of the remainder as `UNDERSHOOTS` concentrated at the low end of scale (D1-D2).

### 2.2 Capacity(F) is not universally closer to the frontier — the upper end reverses the pattern

| Band | Frontier (tol=4pp) | Capacity(F) | ratio | capacity_cap | ratio |
|---|---|---|---|---|---|
| D10 | 250,000 | 660,001 | **2.64x EXCEEDS** | 487,414 | 1.95x EXCEEDS |

At the extreme high-scale end, Capacity(F) becomes the *more* aggressive of the two relative to the frontier. This is not a contradiction of 2.1 — it is a genuinely different regime at the top of the scale distribution.

### 2.3 The frontier's own tolerance behavior reproduces Analysis 4's documented structure, not a coding anomaly

Float D8 moves from a 250,000 frontier at 3pp (Capacity(F) = 251,811, ratio 1.007x, `TRACKS`) to a 500,000 frontier at 4-5pp (Capacity(F) = 251,811 still, ratio 0.504x, `UNDERSHOOTS`). The frontier climbing with relaxed tolerance, and Capacity(F)'s relationship to it flipping as a result, is exactly the kind of band-specific heterogeneity (including the D7-locked / D8-climbing discontinuity) Analysis 4 Section 6 already documented and left as an open empirical issue — its reappearance here is a consistency check, not a new puzzle to chase.

### 2.4 The bottom of the distribution: already sufficiently characterized

At 2-4pp, Capacity(F) undershoots the 50,000 frontier in D1/D2 (15,400 and 38,200 respectively) while `capacity_cap` sits at 85,000 and 212,500. At 5pp the frontier itself moves (D1 becomes 100,000), making Capacity(F) only 0.154x of it; D2's frontier stays at 50,000. The practical implication is already clear enough without further slicing: **Capacity(F) should not currently replace the incumbent for small agents.**

### 2.5 Scope limitation

10,079 live agents (7.0% of the scored population) have `float_activity_value_1m` below anything observed in the historical loan-episode population that produced the frontier; 2,809 (2.0%) similarly have no commission-based frontier evidence. No frontier evidence exists for these agents at all under this analysis — not "undershoot," simply no historical anchor.

## 3. Frozen conclusion

> The historical outcome evidence does not support the proposition that the incumbent `capacity_cap` is a better absolute economic-capacity estimate simply because it is closer to actual historical borrowing or produces larger limits. Instead: both Capacity(F) and `capacity_cap` frequently exceed the historically observed supported-exposure frontier, but the incumbent `capacity_cap` generally exceeds it by substantially larger multiples. Capacity(F) is therefore the more conservative of the two fundamentals-based capacity measures relative to this historical outcome anchor, although neither is validated as a universal hard capacity boundary.

```
Observed-frontier overlay:                           SUPPORTED / FROZEN
Capacity(F) more conservative than capacity_cap
  relative to the historical frontier, mid-scale:     SUPPORTED
Capacity(F) universally closer to the frontier:       NOT SUPPORTED (reverses at the upper end, e.g. D10)
Capacity(F) validated as a hard capacity boundary:     NO
capacity_cap validated as a better absolute estimate: NO
Change k, Capacity(F), capacity_cap, or C3:           NO
Further retrospective slicing of D1/D2:               NOT PLANNED
```

## 4. Implication for Diamond A

This result does not prove Capacity(F) "correct" — it changes what the incumbent-vs-challenger comparison means:

**Incumbent engine**: badly aligned with Diamond A (Section 2 of `docs/diamond_a_capacity_f_reconciliation_findings.md`); its `capacity_cap` sits above Capacity(F) for 91.4% of the overall population; yet that larger `capacity_cap` frequently sits far above the empirical frontier (Section 2.1 above).

**Capacity(F)**: much more conservative than `capacity_cap` across ordinary agents; independently identifies materially greater capacity at the upper end / Diamond A; generally closer to the historical frontier in the middle of the distribution; but still exceeds the frontier substantially in some high-scale bands (Section 2.2 — D10 at 4pp: Capacity(F) 2.64x vs. `capacity_cap` 1.95x).

That last point matters specifically for Diamond A, since Diamond A is concentrated at the upper end of scale: Capacity(F) is not a uniformly safer choice there, and the engine's existing product ceilings remain relevant even under a challenger.

> **Capacity(F) for Diamond A should first be an uplift challenger, not an automatic production replacement.**

The accumulated position across this whole workstream: **Capacity(F) has failed the case for universal replacement, but strengthened the case for a targeted Diamond-A challenger deployment.** These are not contradictory — one fundamentals-based methodology should not be expected to serve the entire agent population equally well.

## 5. What This Does Not Establish

- That Capacity(F)'s absolute level is a safe, causally-grounded capacity boundary anywhere, including at the upper end where it tracks the frontier more closely than `capacity_cap` does. The frontier itself identifies historical support, not causal safe capacity (Analysis 4 Section 9's endogeneity caveats apply unchanged).
- That `capacity_cap` is simply "wrong" — only that its absolute level is not well-anchored to historical repayment outcomes, which is a narrower claim than a general indictment of the incumbent engine (which also includes usage/prior-exposure/risk/policy layers this analysis does not re-evaluate).
- Anything about the ~9% of the population (10,079 + 2,809 agents, not necessarily disjoint) with no historical frontier evidence at all.
- Any change to `k=0.05`, the Capacity(F) functional form, `capacity_cap`, or C3.

## 6. Scripts and commits (reproducibility trail)

| Script | Role | Key commits |
|---|---|---|
| `scripts/overlay_capacity_f_vs_observed_exposure_frontier.py` | Overlays Capacity(F)/`capacity_cap`/observed frontier per float/commission band x tolerance | `db95ca3` (creation), `aa21af4` (fixed a real-data column collision between engine output's own `commission` column and the Decision Log's) |
| `scripts/compare_capacity_f_vs_incumbent_cap_layers.py` | Established the 91.4% capacity_cap-stage disagreement this analysis follows up on | `e8814e2`, `aa21af4` |
| `scripts/derive_capacity_frontier_from_business_state.py` | Analysis 4 Deliverable 1 -- the observed frontier itself (read-only input here, never regenerated) | see `docs/analysis_4_findings.md` |

Real-data figures above come from a single run reported interactively during this session; the overlay CSVs are not checked into the repository (same convention as Analysis 4's own findings doc).

## 7. Next Step: Diamond-A Capacity Challenger Pilot (Design, Not Yet Implemented)

This is a design record, not a frozen finding — it is not implemented or run as part of this commit.

**Hypothesis**: Among prospectively identifiable Diamond-A agents for whom Capacity(F) x C3 recommends a limit above the Existing Engine, does controlled access to the additional recommended exposure produce acceptable repayment performance relative to the Existing Engine?

**Design**:
```
Eligible Diamond A -> Control:   Existing Engine
                    -> Treatment: Capacity(F) x C3

All other categories -> Existing Engine
```
Same product ceilings and regulatory controls applied after either branch — the existing product-risk envelope is **not** removed for the treatment group, specifically because Section 2.2/4 above shows Capacity(F) can itself become aggressive at the upper end (e.g. D10's 2.64x-the-frontier ratio). The experiment tests the capacity challenger inside the existing envelope, not an unconstrained 660K/1M+/multi-million raw Capacity(F) recommendation.

This is scientifically stronger than continuing to search historical bands for a "better" k: it tests the one claim this workstream has actually built evidence for (Capacity(F) identifies materially greater capacity at the upper end / Diamond A, and is closer to the historical frontier there than the incumbent, while remaining inside product guardrails) rather than re-opening the frontier's own calibration.
