# Stage 4B — Capacity Scale Selection (k) Findings

**Status:** Frozen at commit `fd78fad` on `claude/integrated-solution-analysis-atdfe6`.
**Next step:** Move `Capacity_0.05(F)` (shadow) and `Capacity_0.06(F)` / `Capacity_0.04(F)` (sensitivity scenarios) into prospective validation / shadow-mode logging alongside the existing `M_C3(PD)`. No further retrospective k-selection diagnostic is planned — the historical data have reached diminishing analytical returns on this question.

## 1. Objective

`docs/deliverable_2_capacity_f_findings.md` froze the `Capacity(F)` functional form and its 0.5/0.5 weights, but explicitly left the production scaling constant `k` unselected (Section 9: "a production value of `k` has not been chosen"). This document freezes Stage 4B's end-to-end evidence chain for selecting `k` and records the resulting decision.

As with every prior stage in this workstream, the historical data were used to **narrow a plausible region**, never to **statistically identify a unique optimum**. `k=0.05` is a policy choice made within that region, not an estimated value.

## 2. The Frozen Formula and Architecture

```
Capacity(F) = k * ( sqrt((1 + Float) * (1 + Commission)) - 1 )
L_full       = Capacity(F) x M_C3(PD)
```

with the three-layer architecture kept explicit and never collapsed:

```
Business Capacity  ->  Credit Risk  ->  Product Policy
  Capacity_k(F)          M_C3(PD)        min/max/eligibility
```

**`Capacity(F)` and `L_full` are never internally floored or capped against the product's 50,000/1,000,000 UGX boundaries.** Those boundaries are product-policy controls applied strictly *after* `L_full` is computed, never folded into the capacity or risk-adjustment formulas themselves:

- A raw capacity of 1.8m, C3-adjusted to 1.4m, with a 1m product limit, is three genuinely different, informative numbers. Capping `Capacity(F)` internally at 1m would destroy that information.
- An `L_full` of 37,000 remains 37,000 — never automatically raised to 50,000. Product policy decides downstream whether that agent is "not eligible for this product," "eligible for another product," or routed to a separately-governed fallback.

## 3. Evidence Chain

Seven independent analyses, each already frozen or verified in this session, converged on the decision in Section 4. None of them alone would justify the choice; the convergence across independently-built diagnostics is the basis for it.

| # | Analysis | Script (commit) | Finding |
|---|---|---|---|
| 1 | Unconditioned historical frontier backtest | `derive_capacity_function_and_backtest_frontier.py` (`84666b9`, see `deliverable_2_capacity_f_findings.md`) | Pooled 50% exceedance crossing ≈ k 0.0328 (Float) / 0.0299 (Commission) against the raw, unconditioned Analysis 4 frontier. |
| 2 | PD-comparable frontier reconstruction | `compute_pd_comparable_capacity_frontier.py` (`947ada2`, `0448363`) | Found substantial path-dependence in the original frontier walk (74 Float / 66 Commission cells intrinsically breach vs. 38 originally reached in each). Across the PD-similarity sweep, tighter matching (δ=0.02) moved the median crossing to ≈0.073 (Float) / 0.082 (Commission); looser matching (δ=0.10) to ≈0.050 / 0.045 — versus ≈0.033 / 0.030 unconditioned. **Not point-identified**, but every PD-comparable specification moved the plausible center upward relative to the unconditioned frontier, indicating the original ~0.03 center was partly anchored to historical risk selection rather than a pure capacity boundary. |
| 3 | Candidate-k distribution characterization | `characterize_capacity_candidate_k_distribution.py` (`4173c43`, `4657369`) | At k=0.03, 38.5% of the estimable population falls at or below the 50,000 UGX floor; even at k=0.06, 18.5% remains below it — and this is **not** a bottom-decile phenomenon: the k=0.03 below-floor population's own median Float (2.36m) and Commission (339k) are not small. Confirmed `k=0.03` was unnecessarily conservative for a broad, economically non-trivial swath of the population. |
| 4 | Raw-fundamentals tail composition diagnostic | `diagnose_capacity_score_tail_composition.py` (`6a54016`, `daa5bf7`) | Both the low tail (`both_low`, 97–99% at every k) and the high tail (`both_high`, ~100% at every k) are overwhelmingly driven by genuine joint movement of Float and Commission, not by one fundamental distorting the geometric mean. One single-month record (agent `256776254023`, 2026-01-31) was flagged as qualitatively more extreme than the rest and queued for a source-record check via the same script's `--lookup-agent-msisdn` flag; several other extreme accounts recur across 3–5 monthly snapshots each, consistent with genuine large-scale (not one-off/erroneous) accounts. |
| 5 | Frozen-C3 population consequence analysis | `combine_capacity_candidate_k_with_frozen_c3.py` (`362e872`) | `M_C3 <= 1` everywhere, so C3 only ever shrinks `L_full` relative to raw `Capacity_k(F)` — it cannot resolve the floor/ceiling trade-off, only shift the whole distribution down. Under base C3: below-floor share falls 24.51% (k=.04) -> 18.44% (k=.05) -> 14.26% (k=.06); above-ceiling share rises 1.10% -> 2.02% -> 3.15% over the same steps. |
| 6 | Dollar-weighted incremental consequence analysis | `characterize_capacity_times_c3_dollar_consequence.py` (`fd78fad`) | The .04->.05 increment (UGX 201.4bn raw, base C3) lands 83.1% in-range / 88.1% capped-usable. The equal-sized .05->.06 increment lands only 79.3% in-range / 83.4% capped-usable — a genuine, monotonic decline in marginal efficiency. Excess-above-ceiling exposure rises ~66% (50.2bn -> 83.6bn) for a 20% increase in k, while usable product-controlled exposure (`E_product_test`) rises only ~18.6% (930.7bn -> 1,104.1bn) over the same step. |
| 7 | (Reaffirmed, not re-tested) Concordance check | `derive_capacity_function_and_backtest_frontier.py` (see `deliverable_2_capacity_f_findings.md` Section 5) | No evidence the geometric-mean substitution assumption systematically distorts the discordant Float/Commission tail — reaffirmed by #4's `both_low`/`both_high` dominance, not re-run in Stage 4B. |

## 4. Decision

> **Global capacity scale selection:** `k=0.05` is selected as the initial shadow-policy scaling factor for `Capacity(F)`. The historical analyses do not statistically identify a unique optimal `k`; rather, they support a plausible range. `k=0.05` is selected as a conservative operating point within that range. Relative to `k=0.04`, it materially improves product-range coverage and deployable capacity. Increasing further to `k=0.06` continues to improve population coverage but exhibits declining marginal product efficiency, with a larger share of incremental capacity falling above the current UGX 1,000,000 product ceiling. `k=0.06` is therefore retained as an upside sensitivity rather than adopted as the initial operating point.

```
k_shadow      = 0.05   (initial shadow-policy operating point)
k_sensitivity = 0.06   (upside sensitivity scenario -- NOT rejected, retained for comparison)
k_reference   = 0.04   (downside / conservative reference sensitivity)
```

`k=0.06` is explicitly **not** found to be unsafe or misspecified — Section 3 item 6 shows it continues to deliver real net gains in both population coverage and usable exposure. The choice of `0.05` over `0.06` is a conservative risk-appetite judgment about where diminishing marginal product efficiency becomes a reason to pause, not a finding that `0.06` is wrong. Both remain live scenarios to run side-by-side.

## 5. What This Decision Does Not Establish

- `k=0.05` is not a statistically estimated optimum — the historical evidence narrows a region, it does not point-identify a value.
- Neither `Capacity(F)` nor `L_full` has been validated against actual future repayment performance at any `k`. Everything in Section 3 is retrospective and consequence-based; prospective validation (Section 7) is the explicit next step, not yet performed.
- `k=0.06` is not rejected; `k=0.04` is not eliminated as a reference point. Both remain explicitly retained sensitivity scenarios.
- The source-record check on agent `256776254023` (Section 3, item 4) is queued, not confirmed resolved as of this freeze — the finding above does not depend on its outcome, since the broader tail-composition conclusion rests on the `both_low`/`both_high` population pattern, not on this one record.
- The 50,000/1,000,000 UGX product boundaries remain hypothetical, policy-owned controls throughout every analysis in Section 3 — none of them redefined `Capacity(F)` or `L_full` internally against those boundaries, and this document does not change that.

## 6. Scripts, Commits, and Real-Data Provenance

| Script | Role | Key commit(s) |
|---|---|---|
| `scripts/compute_pd_comparable_capacity_frontier.py` | PD-comparable historical frontier re-walk; full independent re-walk per PD-similarity delta | `947ada2`, `0448363` |
| `scripts/characterize_capacity_candidate_k_distribution.py` | Candidate-k Capacity(F) output distribution characterization (overall, by decile, tier-bucket proportions, concordance) | `4173c43`, `4657369` |
| `scripts/diagnose_capacity_score_tail_composition.py` | Raw Float/Commission composition behind the low and high tails; per-agent source-record lookup | `6a54016`, `daa5bf7` |
| `scripts/combine_capacity_candidate_k_with_frozen_c3.py` | `L_full(k) = Capacity_k(F) x` frozen C3, loan-grain population consequence (tier distribution + transition matrix) | `362e872` |
| `scripts/characterize_capacity_times_c3_dollar_consequence.py` | Dollar-weighted floor/ceiling detail and incremental 0.04->0.05 / 0.05->0.06 decomposition | `fd78fad` |
| `docs/deliverable_2_capacity_f_findings.md` | The frozen functional-form/weights decision this document builds on | `84666b9` |

All five scripts were verified against hand-built synthetic fixtures (hand-computed formula cases, boundary/tie-handling cases, forbidden-columns regression guards, and end-to-end CLI smoke tests) before each commit above. Every real-data figure in this document comes from real-data runs reported interactively during this session, not from committed output files — consistent with every other real-data run in this workstream; no real data or generated CSV is checked into the repository.

## 7. Next Step

Move `Capacity_k(F)` (at `k=0.05`, with `k=0.04`/`0.06` as sensitivity scenarios) into **prospective validation**: contemporaneous shadow-mode logging of `F_t`, `Capacity_k(F_t)`, `cal_pd_t`, `M_{C3,t}`, `L_{current,t}` (the live engine's own limit), and `L_{full,t} = Capacity_k(F_t) \times M_{C3,t}`, against eventual matured real outcomes — not another retrospective slice of the same historical data. This mirrors the forward-looking design note already recorded in the Stage 4A audit (`scripts/audit_historical_pd_rescoring_feasibility.py`'s module docstring) for exactly this contingency. No code for this step has been written; it is explicitly out of scope for this document.
