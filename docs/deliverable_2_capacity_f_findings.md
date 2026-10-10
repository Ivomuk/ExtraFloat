# Deliverable 2 — Capacity(F) Findings (Stages 1–3)

**Status:** Frozen at commit `84666b9` on `claude/integrated-solution-analysis-atdfe6`.
**Next step:** Stage 4 — Capacity(F) x independent risk (C3) read-only diagnostic (design only; not started, not specified).

## 1. Objective

Deliverable 2 designs `Capacity(F)`: a frozen, fundamentals-only production mapping from business fundamentals to a UGX business-capacity estimate, built without using historical assigned limits or outcomes as a fitting target. This document freezes the conclusions from Stages 1-3 (score design, UGX mapping, and backtest against the Analysis 4 frontier) before Stage 4 (combination with the independent C3 risk multiplier) is designed or implemented.

`Capacity(F)` is explicitly **not** a repayment-risk model and does not estimate credit risk. It is a fundamentals-only economic-capacity construct, kept separate from the existing shadow risk multiplier `M_C3(PD)` by design:

```
L_full = Capacity(F) x M_C3(PD)
```

Conflating the two components, or tuning `Capacity(F)` to reproduce historical risk-adjusted behavior, would defeat the reason this separation was introduced in Analysis 3/4.

## 2. Functional Form and Weights

The candidate functional form, unchanged from the approved design and not revised by anything found in this round's backtest:

```
Capacity(F, C) = k * ( sqrt((1 + Float) * (1 + Commission)) - 1 )
```

i.e. `k` times an equally-weighted geometric mean of `(1+Float)` and `(1+Commission)`, minus 1.

Weights `w_F = w_C = 0.5` were derived, not fitted: for exactly two fundamentals standardized to unit variance, PCA's first component is provably `(1,1)/sqrt(2)` for any correlation strength between them, confirmed numerically on the real data (`derive_capacity_score_weights.py`, real-data run: Spearman correlation 0.8786, `pca_weights_standardized` = `[0.5, 0.5]`, `pca_equals_equal_weights` = `True`). Standardized two-variable PCA **confirms** this weighting rather than **estimating** it from data. The unstandardized PCA path, computed only as a labeled diagnostic, deviates (`[0.628, 0.372]` on the real data) precisely because it reflects which fundamental has more log-space variance in this snapshot -- a scale artifact, not economically meaningful information, which is why it is never used as the recommended weighting.

## 3. The k=1 Backtest: A Scale Mismatch, Not (Yet) a Functional-Form Failure

At `k=1`, `Capacity(F,C) ~ sqrt(Float * Commission)` for large fundamentals -- a business-flow/economic-magnitude quantity, not a single-loan exposure limit on the 50K-1M UGX ladder the historical frontier is measured on. The real-data `k=1` backtest showed this scale mismatch directly: at every float and commission band, at every tolerance, capacity at `k=1` exceeded the historically-supported frontier for 96-99.97% of agent-periods, with median ratios ranging roughly 4x to 260x depending on band and tolerance.

A real-data k-sensitivity sweep (pure arithmetic rescaling of the `k=1` backtest, since `R(k) = k * R(1)`) located the informative diagnostic region at roughly **k = 0.015-0.06**, with the pooled 50% exceedance point for each fundamental at:

| Fundamental | 25% crossing | 50% crossing | 75% crossing |
|---|---:|---:|---:|
| Float | k ~ 0.0175 | k ~ 0.0328 | k ~ 0.0626 |
| Commission | k ~ 0.0156 | k ~ 0.0299 | k ~ 0.0573 |

**No production `k` has been selected.** The center of this region, roughly **k ~ 0.02-0.03**, is the empirical location where the measured business-flow quantity becomes comparable in magnitude to historically-supported single-loan exposure -- it is a diagnostic finding, not a policy choice. Choosing a production `k` remains a deferred, separate decision, to be made by policy judgment informed by (never fit to) this region, not automatically by this document.

## 4. Float and Commission Independently Locate the Same Scaling Region

The float-frontier and commission-frontier backtests are independent (different historical frontier, different band assignment) yet locate their 50% pooled-exceedance crossing within ~10% of each other (k ~ 0.0328 vs. k ~ 0.0299). This cross-validates that the scale mismatch is a genuine property of the business-flow-vs.-single-loan-exposure gap itself, consistent with float and commission's high correlation (Spearman 0.879, Analysis 4 Section 5), rather than an artifact specific to one fundamental or one frontier construction.

## 5. Concordance Testing: No Evidence Against the Geometric-Mean Substitution Assumption

The concordant/discordant split (`|float_band_rank - commission_band_rank| <= 1` vs. `>= 3`) was designed specifically to detect whether the geometric mean's substitution property (a large value in one fundamental compensating for a small value in the other) systematically inflates capacity for Analysis 4's flagged 8.4% discordant tail. The real-data result does not show that:

| k | Float: concordant / discordant exceedance | Commission: concordant / discordant exceedance |
|---:|---:|---:|
| 0.01 | 9.4% / 2.1% | 15.5% / 4.7% |
| 0.03 | 47.9% / 43.5% | 50.7% / 47.9% |
| 0.05 | 66.8% / 70.9% | 71.5% / 69.9% |

Concordant exceedance leads discordant through the diagnostic region; the two cross and converge only as both approach saturation at higher k. There is no present evidence that the geometric mean systematically overstates capacity for discordant agent-periods. The directional question (whether `float >> commission` and `commission >> float` discordant cases behave differently from each other, which the current pooled discordant bucket cannot distinguish) is recorded as an open item, not as a reason to revisit the 50/50 geometric aggregation now.

## 6. Residual Frontier Heterogeneity

Independently of the global scale mismatch, substantial band-level heterogeneity remains that a single global `k` cannot remove. At the most permissive historical tolerance (5pp), float band D7's `k=1` median ratio (72.0) remains roughly 7x band D8's (9.9) at every `k` in the diagnostic region (e.g. at k=0.03: D7 ~ 2.16 vs. D8 ~ 0.30). Commission band D10 is more extreme still: a `k=1` median ratio of 256.6, unchanged across all four tolerances (its frontier never advances past the reference tier), remaining ~7.7x the frontier even at k=0.03.

Float D7 and commission D10 are the **clearest examples** of this residual heterogeneity, not established as the only ones -- a complete per-band exceedance-rate characterization across the diagnostic k region (beyond the median-ratio view used here) was not performed in this round and remains an open descriptive task, not a prerequisite for the architecture conclusions below.

## 7. Why Heterogeneity Must Not Be Absorbed Into Capacity(F)

This heterogeneity must **not** be converted into band-specific `k`'s or any other outcome-derived correction inside `Capacity(F)`. Doing so --

```
Capacity(F, C) = k(band) * ( sqrt((1+Float)*(1+Commission)) - 1 )
```

-- with `k(band)` chosen from observed historical frontier behavior, would make historical risk/assignment behavior part of the capacity formula itself: exactly the endogenous-target problem Deliverable 2 exists to avoid. Both float-D7-specific and commission-D10-specific corrections are explicitly rejected at this stage, for the same reason.

The residual heterogeneity is a genuine and important finding -- it says the historical frontier is not representable as a smooth function of fundamentals alone -- but what it implies (a missing capacity dimension, vs. credit-risk selection that belongs in `M_C3`, vs. something else) is not yet established, and is the question Stage 4 is designed to narrow.

## 8. Role of the Analysis 4 Frontier

The Analysis 4 historically-supported exposure frontier remains a **diagnostic and constraint surface** for evaluating candidate `Capacity(F)` mappings, never a fitting target. `Capacity(F)` was not adjusted, refit, or special-cased to better match the frontier at any point in this backtest. The frontier's own two definitions (`contiguous_supported_frontier_tier_ugx` and `highest_tier_with_any_supported_evidence_ugx`) were observed to coincide in essentially every band x tolerance cell of the real-data run, indicating the frontier's heterogeneity is driven by genuine supported/breach structure rather than by sequential evidence gaps.

## 9. What This Analysis Does Not Establish

This analysis does **not** establish that:

- the proposed capacity amounts, at any `k`, are safe lending limits;
- a production value of `k` has been chosen (the diagnostic region ~0.015-0.06 is not a policy recommendation);
- `Capacity(F)` combined with `M_C3` (Stage 4), production engine integration, or prospective validation have been addressed -- all remain explicitly outside this frozen conclusion;
- float D7 and commission D10 are the only bands with residual heterogeneity, only the clearest examples observed;
- the residual heterogeneity reflects a missing capacity dimension rather than credit-risk selection, or vice versa.

## 10. Decision and Next Step

Stages 1-3 of Deliverable 2 are considered **successful and frozen**, not a failed design: they established that the proposed fundamentals-only functional form has a sensible, cross-validated global scaling region, that the geometric-mean discordance assumption has not shown a structural failure, and that the remaining historical heterogeneity is real but must not be absorbed into `Capacity(F)` without importing historical risk/assignment behavior back into it.

The next stage narrows, rather than broadens, the open question. Stage 4's central question is:

> Does independently measured credit risk explain the historical-frontier heterogeneity that remains after constructing a fundamentals-only capacity measure?

This is deliberately narrower than "combine Capacity(F) and C3 to produce a number" -- it is a read-only diagnostic distinguishing two hypotheses for the problematic frontier cells (e.g. commission D10, float D7): `H1`, that the heterogeneity is substantially risk-driven (explained by low C3 multipliers / high PD among those agent-periods); vs. `H2`, that it persists even among comparable low-risk agents (high C3 multiplier / low PD), which would suggest `Capacity(F)` is missing a genuine capacity dimension rather than conflating risk. Stage 4 is design-only as of this document; no code has been written for it.

## 11. Scripts, commits, and real-data provenance (reproducibility trail)

| Script | Role | Key commit |
|---|---|---|
| `scripts/derive_capacity_score_weights.py` | Derives/verifies the 0.5/0.5 float-commission weighting from the fundamentals' own joint distribution only (never from historical limits or outcomes) | `84666b9` |
| `scripts/derive_capacity_function_and_backtest_frontier.py` | Computes the diagnostic score and UGX `Capacity(F)` via the weighted geometric mean, and backtests it against both Analysis 4 frontiers (contiguous and highest-any-supported), with the concordance split and the arithmetic-only k-sensitivity table | `84666b9` |
| `docs/analysis_4_findings.md` | The frozen Analysis 4 findings this deliverable builds on (historically-supported exposure frontier, by fundamental) | `2117c9a` |

Both scripts were verified against synthetic fixtures (hand-computed geometric-mean cases, PCA-invariance at several exact target correlations, forbidden-columns regression guards, frontier-join correctness, k-linearity, a hand-computed k-sensitivity table, and an end-to-end Script-1-into-Script-2 fixture) before commit `84666b9`.

Every real-data figure in this document (the `k=1` backtest distributions, the k-sensitivity crossing points, the concordance table, the D7/D8/commission-D10 comparisons) comes from real-data runs reported interactively during this session, not from committed output files. The weights artifact (`capacity_artifacts/capacity_score_weights.json`) and the backtest/k-sensitivity CSVs (`capacity_function_backtest_*`) were generated locally by the user and are not checked into the repository -- consistent with every other real-data run in this workstream.
