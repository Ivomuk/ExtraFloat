# ExtraFloat Agent Segmentation

Assigns MTN Mobile Money (Uganda) agents to 8 business tiers
(`capacity_tier`) from transactional KPIs, using a deterministic, versioned
capacity scorecard as the sole tiering mechanism. A lightweight, two-stage
anomaly check (`is_anomaly`) runs by default alongside it: HDBSCAN flags
agents that don't resemble any dense cluster at all (`is_global_anomaly`),
then Local Outlier Factor (LOF) re-examines only the agents HDBSCAN placed
in a cluster to catch subtler, local anomalies within it
(`is_local_anomaly`). The full GMM+HDBSCAN+UMAP diagnostics bundle
(archetype research, plus a second, deeper anomaly signal) is available as
opt-in. Neither anomaly check nor diagnostics ever decides an agent's tier.

> **Branch note.** This is `claude/segmentation-capacity-tier-primary` —
> the deterministic scorecard fully *replaces* the old ensemble-cluster
> `segment`/`tier` mechanism here (a scorecard is required by default; see
> [Running it](#running-it)). A sibling branch,
> `claude/segmentation-solution-review-ls9qpc`, instead runs the scorecard
> as an opt-in *shadow* alongside the still-intact old mechanism, for
> comparing the two on real data before committing to a replacement. This
> branch also contains the segmentation package only — the credit/float
> limit engine that consumes this package's output lives on other branches
> (e.g. `main`), not here.

## Package layout

| File | Role |
|---|---|
| `extrafloat_segmentation_features.py` | Feature engineering: cleaning → date/premium/interaction features → log1p+winsorize → correlation pruning → RobustScaler+PCA |
| `extrafloat_segmentation_scoring.py` | **Deterministic capacity scorecard** — calibration, versioned persistence, per-agent scoring/tiering, tenure safety cap. The sole source of `capacity_tier`. |
| `calibrate_scorecard.py` | CLI for the offline, human-reviewed scorecard calibration step (`calibrate_capacity_scorecard` + `save_scorecard`) |
| `extrafloat_segmentation_pipeline.py` | `flag_anomalies()` — lightweight, two-stage anomaly check (HDBSCAN global pass + LOF local refinement), **on by default**. `run_diagnostic_clustering()` — full GMM+UMAP+HDBSCAN **diagnostics, opt-in**. Neither assigns a business tier. |
| `extrafloat_segmentation_profiling.py` | Pack-based KPI profiling (research-only, opt-in) + whitelist/blacklist reference-list merge |
| `extrafloat_segmentation_validation.py` | Purity/ARI/feature-importance validation toolkit + the automated quality gate |
| `extrafloat_segmentation_drift.py` | PSI/KL feature-drift monitoring vs. a saved baseline, plus categorical PSI for `capacity_tier` proportions vs. scorecard calibration |
| `extrafloat_segmentation_viz.py` | Plotting helpers (matplotlib/seaborn, soft dependency) |
| `run_extrafloat_segmentation.py` | Orchestration entry point + CLI |
| `test_extrafloat_segmentation.py`, `test_extrafloat_segmentation_scoring.py` | Pytest suites |

## Pipeline

```
agents_df
  │
  ▼
[1] prepare_features()                 feature engineering + PCA
  │
  ▼
[1b] drift check (optional)            PSI/KL vs. saved baseline
  │
  ▼
[2] compute_agent_capacity()           REQUIRED — deterministic capacity_score,
  │                                     capacity_tier_raw, capacity_tier (frozen
  │                                     scorecard; tenure safety cap; dormant
  │                                     agents forced to lowest tier)
  │                                     → tier-proportion drift check (PSI vs.
  │                                       scorecard's calibration expectations)
  │
  ▼
[2b] flag_anomalies()                  default on (clustering.enable_anomaly_detection,
  │                                     default True) — two-stage, on the same PCA
  │                                     space (no GMM, no UMAP). Stage 1: HDBSCAN
  │                                     flags global noise (is_global_anomaly).
  │                                     Stage 2: LOF re-examines only the
  │                                     HDBSCAN-survivors for local anomalies
  │                                     (is_local_anomaly, lof_score). is_anomaly =
  │                                     OR of both stages. Degrades gracefully
  │                                     (is_anomaly=False) if hdbscan is missing —
  │                                     never blocks capacity_tier.
  │
  ▼
[3] run_diagnostic_clustering()        optional (clustering.enable_diagnostics,
  │                                     default False) — GMM + UMAP→HDBSCAN,
  │                                     diag_hdb_tier / diag_ensemble_cluster /
  │                                     diag_is_anomaly. Never touches capacity_tier.
  │
  ▼
[4] build_cluster_pack_profiles()      optional (profiling.enable_pack_profiles,
  │                                     default False) — KPI pack means/lifts
  │                                     grouped by capacity_tier, attrs-only,
  │                                     research aid
  │
  ▼
[5] merge_reference_lists()            optional whitelist/blacklist enrichment
  │                                     (agent_category)
  │
  ▼
[6] run_quality_gate()                 automated sanity/quality checks on
  │                                     capacity_tier + structured alerts
  │
  ▼
[7] output trimming                    → agent_msisdn, capacity_score,
                                          capacity_tier, ... (+ deprecated
                                          segment/tier aliases by default)
```

## Business tiers

`BUSINESS_SEGMENTS` (from `extrafloat_segmentation_scoring.py`), lowest to
highest value:

```
Below Threshold, New Bronze, Bronze, Silver, Gold, Platinum, Titanium, Diamond
```

An agent's `capacity_tier` comes from a **frozen, versioned scorecard**:
each *individual* KPI (e.g. `cash_out_value_3m`, `tenure_years`) is
normalized independently against its own *frozen* reference range fixed
at calibration time (not recomputed from the current run's population),
combined into one of three factor groups (value/activity/efficiency) via
explicit within-group KPI weights, then blended across groups (default
group weights 50/30/20) into one `capacity_score` in `[0, 1]`, and
bucketed against *frozen* cutoff thresholds. Given the same scorecard, the
same agent feature values always produce the same score and tier —
regardless of which other agents are in the run. Agents below the
scorecard's `min_tenure_years` are downgraded one tier; dormant agents
(identified by a weighted composite inactivity score) always land in
"Below Threshold".

Normalizing each KPI independently *before* combining — rather than
averaging/summing raw KPIs of different units and scales together first —
matters concretely: an earlier version of this scorecard averaged
`commission_per_value_3m`/`commission_per_value_6m` (ratios near 0.01)
directly with raw `tenure_years` (range 0–20+) inside the "efficiency"
factor, letting whichever had the larger raw scale silently dominate. Each
factor group also only keeps the *widest* available window per KPI (e.g.
`cash_out_value_3m` but not also `cash_out_value_1m`) — summing overlapping
windows double-counted the most recent month.

A scorecard is produced offline by `calibrate_capacity_scorecard`
(typically via `calibrate_scorecard.py`) and is a human-reviewed
governance artifact, not something a production run fits itself — see
[Running it](#running-it). Every scorecard produced so far in this repo is
**provisional** (`is_provisional: true`, `cutoff_version: "provisional_v0"`)
— calibrated against synthetic/tiny sample data, not reviewed against real
production data. Treat any provisional scorecard's cutoffs as a
placeholder proving the mechanism works end to end, not as calibrated
business thresholds. `run_extrafloat_segmentation` refuses to apply a
provisional scorecard by default (`scoring.allow_provisional_scorecard=False`)
— see [Reliability features](#reliability-features).

Determinism is not the same as validity. Reproducing the same number every
time says nothing about whether that number tracks real outcomes
(sustainable capacity, repayment behavior, fairness across regions/agent
types) — that validation is exactly what's blocked on real data; see
[Known limitations](#known-limitations--open-items).

## Running it

A scorecard is **required** — a run with no scorecard configured raises
`ValueError` by default (`scoring.allow_missing_scorecard=False`). A freshly
calibrated scorecard is also **provisional by default** and is likewise
refused unless explicitly allowed (`scoring.allow_provisional_scorecard=False`)
— pass `--final` to `calibrate_scorecard.py` once it's been reviewed, or
set `allow_provisional_scorecard=True` for a deliberate non-production run.

```bash
pip install -r requirements-segmentation-dev.txt

# 1. Calibrate a scorecard (offline, human-reviewed step)
python calibrate_scorecard.py --agents development_agents.csv \
    --out scorecards/capacity_scorecard_v0.json
    # add --final once reviewed, to clear is_provisional

# 2. Run the pipeline against it (add allow_provisional_scorecard=True via
#    --config/JSON, or --final above, before this will run without it)
python run_extrafloat_segmentation.py --agents agents.csv \
    --scorecard scorecards/capacity_scorecard_v0.json \
    --output segmentation_outputs/
```

Or from Python:

```python
from run_extrafloat_segmentation import run_extrafloat_segmentation, DEFAULT_SEGMENTATION_CONFIG
import pandas as pd
from copy import deepcopy

agents_df = pd.read_csv("agents.csv")
config = deepcopy(DEFAULT_SEGMENTATION_CONFIG)
config["scoring"]["scorecard_path"] = "scorecards/capacity_scorecard_v0.json"

result = run_extrafloat_segmentation(agents_df, config=config)
result[["agent_msisdn", "capacity_score", "capacity_tier", "is_anomaly"]].head()
```

`is_anomaly` is computed by default (`clustering.enable_anomaly_detection`,
default `True`) — set it to `False` to skip it. It's the OR of two stages:
`is_global_anomaly` (HDBSCAN noise) and `is_local_anomaly` (LOF, run only on
the agents HDBSCAN placed in a cluster); disable the second stage alone via
`clustering.lof_enabled = False`. To also run the full diagnostics bundle
(archetype research, plus a second, UMAP-space anomaly signal
`diag_is_anomaly`) alongside capacity scoring:
`config["clustering"]["enable_diagnostics"] = True`.

Required input columns are listed in `extrafloat_segmentation_features.REQUIRED_COLUMNS`.

## Reliability features

- **Determinism by construction**: `capacity_tier` depends only on an
  agent's own feature values and the frozen scorecard — never on which
  other agents are present in the run. This is the core property the
  scorecard mechanism exists to guarantee (see
  `test_extrafloat_segmentation_scoring.py::TestProductionScoringDeterminism`),
  replacing the old ensemble-cluster mechanism's population-relative
  quantile ranking.
- **Scorecard-required guard rail** (`scoring.scorecard_path` /
  `scoring.allow_missing_scorecard`, default `False`): a run with no usable
  scorecard raises `ValueError` before producing any output, rather than
  silently returning agents with no tier.
- **Provisional-scorecard guard rail** (`scoring.allow_provisional_scorecard`,
  default `False`): a scorecard whose `calibration_metadata.is_provisional`
  is `True` (the default from `calibrate_capacity_scorecard` until
  `--final`/`is_provisional=False` is passed) is refused with `ValueError`
  rather than silently applying unreviewed cutoffs to what looks like a
  production run.
- **Fail-closed on missing scorecard inputs** (`scoring.on_missing_column`,
  default `"raise"`): if a scorecard-declared KPI column is entirely absent
  from the input data (e.g. an upstream schema regression), scoring raises
  `ValueError` rather than silently computing that factor as 0.0 for every
  affected agent — a partial computation on a governed financial score
  would mean different agents (or different runs) get scored under
  different effective definitions with no visible error. Set to `"zero"`
  only for a deliberate degraded research/dev run, never for governed
  scoring. `calibrate_scorecard.py --allow-missing-columns` is the matching
  escape hatch for offline calibration.
- **Anomaly detection, on by default but never blocking**
  (`clustering.enable_anomaly_detection`, default `True`): `flag_anomalies`
  runs a two-stage filter (no GMM, no UMAP) on every production run, on the
  raw PCA space. Stage 1 runs HDBSCAN over all active agents and flags
  agents that don't resemble any dense cluster at all
  (`is_global_anomaly`). Stage 2 runs Local Outlier Factor
  (`sklearn.neighbors.LocalOutlierFactor`) *separately within each* HDBSCAN
  cluster — never pooled across clusters — re-examining each agent against
  its own cluster's local neighborhood density to catch subtler anomalies
  that blend in globally but stand out locally (`is_local_anomaly`, plus
  the continuous `lof_score`, more negative = more anomalous, `NaN` for
  agents LOF didn't run on). `is_anomaly` is the OR of both stages. Fitting
  one LOF model per cluster (rather than one model over every in-cluster
  agent combined) keeps "local" honest — a small or sparse cluster's
  members are judged against their own neighbors, never against a larger
  or differently-dense neighboring cluster. `clustering.lof_enabled`
  (default `True`) turns off stage 2 alone, leaving stage 1 unaffected;
  `clustering.lof_n_neighbors` (default `20`, clamped down for small
  clusters), `clustering.lof_contamination` (default `"auto"`, scikit-learn's
  fixed-offset heuristic, not data-fit), and `clustering.lof_min_cluster_population`
  (default `5`) tune it. Clusters smaller than `lof_min_cluster_population`
  are skipped for stage 2 — not silently dropped: their sizes are recorded
  in `result.attrs["anomaly_report"]["lof_skipped_cluster_sizes"]`.

  Three independent layers keep this stage from ever blocking
  `capacity_tier`, which by the time step 2b runs has already been
  computed: (1) `lof_n_neighbors`/`lof_contamination`/
  `lof_min_cluster_population` are validated before they ever reach
  scikit-learn — an invalid value disables stage 2 for the run (stage 1 is
  unaffected) and raises a `critical` alert (`source="lof_config"`) instead
  of an opaque sklearn exception; (2) each cluster's LOF fit is wrapped
  individually, so a numerical failure on one cluster doesn't stop LOF from
  running on the others — failures are counted in
  `anomaly_report["lof_clusters_failed"]` and raise a `warning` alert
  (`source="lof_runtime"`); (3) the whole step is wrapped one more time in
  orchestration as a last line of defense — if something still fails
  unexpectedly, `capacity_tier` is returned regardless, anomaly columns are
  simply absent for that run, and a `critical` alert
  (`source="anomaly_detection"`) is raised.

  Because `clustering.lof_enabled=True` only reflects the config toggle,
  not whether stage 2 actually ran (a cluster can be too small, HDBSCAN can
  find no clusters at all, or the config can be invalid),
  `anomaly_report["lof_status"]` records the real outcome —
  `"ran"` / `"ran_with_failures"` (some clusters succeeded, others errored
  — distinct from a clean `"ran"` so a consumer doesn't have to
  cross-check `lof_clusters_failed` to notice) / `"disabled"` /
  `"skipped_hdbscan_unavailable"` / `"skipped_no_active_agents"` /
  `"skipped_no_clusters"` / `"skipped_all_clusters_too_small"` /
  `"invalid_config: ..."` / `"failed"` (every cluster errored) /
  `"failed_unexpectedly"` — alongside `lof_clusters_total` /
  `lof_clusters_ran` / `lof_clusters_skipped_too_small` /
  `lof_clusters_failed`. `anomaly_report["lof_config_error"]` is validated
  and set independently of *why* stage 2 didn't run this call — e.g. it's
  still populated when `lof_status="skipped_hdbscan_unavailable"`, so an
  invalid config and a missing `hdbscan` install are both visible from one
  run (two separate alerts) instead of one hiding the other until the
  first is fixed and redeployed.

  Unlike the full diagnostics bundle below, a missing `hdbscan` install
  degrades the whole check gracefully — every one of
  `is_anomaly`/`is_global_anomaly`/`is_local_anomaly` defaults to `False`
  and `lof_score` to `NaN` for every agent, with
  `result.attrs["anomaly_report"]["hdbscan_available"]` set to `False` and
  surfaced as a `warning`-severity alert — rather than raising, since this
  is an auxiliary signal and `capacity_tier` must never depend on it being
  available. It uses a different embedding (raw PCA space) than the full
  diagnostics bundle's `diag_is_anomaly` (UMAP space), so the two can
  legitimately disagree. `result.attrs["anomaly_report"]` additionally
  breaks down `n_global_anomalies`/`n_local_anomalies` alongside the total
  `n_anomalies`/`anomaly_rate`.
- **Diagnostics dependency guard rails** (`clustering.require_hdbscan` /
  `clustering.require_umap`, default `True`): only evaluated when
  `clustering.enable_diagnostics=True`. A missing `hdbscan` or `umap-learn`
  install then raises `RuntimeError` at the start of
  `run_diagnostic_clustering` rather than silently degrading diagnostic
  quality — and can never block production capacity scoring, which does
  not import this code path at all. Set to `False` only for a deliberately
  degraded diagnostics run — the resulting
  `result.attrs["diag_degraded_mode"]` flags will be `True`.
- **Quality gate** (`extrafloat_segmentation_validation.run_quality_gate`,
  wired into orchestration step 6, config under `quality_gate`): runs
  unsupervised sanity checks (tier diversity, HDBSCAN noise share when
  diagnostics ran, "Below Threshold" share among active agents) on every
  run, plus purity/ARI checks against `agent_category` when a
  whitelist/blacklist merge supplied one. Reports via
  `result.attrs["quality_gate"]`; set `quality_gate.fail_on_breach=True` to
  raise instead of only logging `CRITICAL` once thresholds are calibrated
  against real data.
- **Tier-proportion drift** (`result.attrs["tier_drift_report"]`, always
  computed when a scorecard is applied): PSI between the live
  `capacity_tier` distribution and the scorecard's calibration
  expectations. Because cutoffs are frozen rather than live quantiles, a
  real population shift now shows up here instead of being silently
  absorbed by re-balancing — that's the point, but it means this check
  needs to be watched.
- **Diagnostic ensemble stability** (`result.attrs["diagnostic_stability_report"]`,
  opt-in via `clustering.full_stability_check=True`, requires
  `enable_diagnostics=True`, off by default because it re-runs the full
  active-agent diagnostics pipeline `stability_n_seeds` times): measures
  reproducibility of `diag_ensemble_cluster` across random seeds. Purely
  informational for diagnostics — `capacity_tier` is unaffected regardless
  of this result.
- **Drift detection**: PSI/KL divergence of input features against a saved
  baseline (`extrafloat_segmentation_drift.py`), wired into orchestration
  step 1b when `drift.baseline_path` is configured.
- **Run manifests** (`output.save_run_manifest`, default `True`, written
  alongside `agent_segments.csv` as `run_manifest.json` whenever
  `output.output_dir` is set): records the resolved config, package
  versions, git commit SHA, input/output row counts, and every report
  above (drift, tier drift, anomaly, diagnostic stability, quality gate,
  degraded mode) for a given run — so a credit-tier decision can be traced
  back to exactly what produced it, including which `scorecard_version` /
  `cutoff_version` was in effect. See `_build_run_manifest()`.
- **Structured alerts** (`result.attrs["alerts"]`, always computed): rolls
  diag-degraded-mode/anomaly-detection/quality-gate/drift/tier-drift/
  diagnostic-stability findings into a flat list of
  `{severity, source, message}` dicts. No notification channel
  (Slack/email/etc.) is wired up yet — that's a deliberate open item, see
  below — but any caller can consume this list without knowing the shape
  of each individual report. The CLI prints these; see `_collect_alerts()`.

## What data actually feeds this pipeline

The real MTN MoMo KPI-mart schema is: `agent_msisdn`, `snapshot_dt`,
`agent_profile`, `account_balance`, `average_balance`, `commission`, and
`cash_out`/`cash_in`/`payment` × `vol`/`value`/`peers`/`comm`/`cust` split
into 1m/3m/6m windows, plus `cust_1m/3m/6m` and `vol_1m/3m/6m` totals
(~57 columns), as exercised by `_make_agents_df()` in
`test_extrafloat_segmentation.py`. There is currently no real multi-month
or labeled dataset available in this repo to calibrate against — every
scorecard produced so far is calibrated against small synthetic/sample
populations and is explicitly marked provisional (see
[Business tiers](#business-tiers)).

## Integration boundary

This package's output (`capacity_score`, `capacity_tier`, `is_anomaly` and its
`is_global_anomaly`/`is_local_anomaly`/`lof_score` stage breakdown, plus
deprecated `segment`/`tier` aliases — see `output.emit_legacy_aliases`) is intended to
feed a separate downstream credit/float-limit engine that is **not present
on this branch** (see the branch note at the top of this document for
where it lives). Whether that engine should consume `capacity_tier`
directly, and how, is an open integration question that needs a decision
from whoever owns both systems — this document intentionally does not
describe the limit engine's internals.

## Known limitations / open items

A round of external review distinguished "deterministic" from "valid":
reproducing the same tier every time proves the mechanism works, not that
higher tiers track real sustainable capacity, better float utilization,
lower credit loss, or fairness across regions/agent types — call it
**engineering framework: solid; deployable business segmentation: not yet
demonstrated.** That review's structural findings (silent zero-fill on
missing inputs, permissive `validate_scorecard`, unenforced provisional
status, unit-mixing inside the "efficiency" factor, overlapping-window
double-counting) are fixed — see the fail-closed/provisional-guard bullets
above and the per-KPI normalization described under
[Business tiers](#business-tiers). What's still open:

- **Blocked on real data** — deliberately not started, not half-built:
  - The scorecard's factor-group weights, within-group KPI weights,
    normalization ranges, and tier cutoffs are hand-set/provisionally-
    calibrated defaults, not validated against real outcomes — no real
    labeled dataset exists in this repo yet.
  - **No temporal / out-of-time validation** — the single biggest missing
    evaluation. What's needed once real data exists: calibrate on one
    historical period, apply the frozen scorecard to later months, measure
    tier migration matrices and stability, check outcome monotonicity by
    tier (do higher tiers actually perform better), verify tier
    transitions correspond to genuine KPI changes rather than noise,
    backtest against any downstream limit/exposure policy, and segment
    all of the above by tenure/geography/agent profile/activity level.
    Nothing in this repo currently has more than one snapshot date to do
    any of this against.
- **Quality gate defaults are permissive by design, not yet a real gate**:
  `fail_on_breach=False`, `min_purity`/`min_ari=0.0`, a 2-tier diversity
  floor out of 8, and up to 50% of active agents allowed below threshold —
  today this is an observability report, not a deployment gate. Set
  `quality_gate.fail_on_breach=True` (and tighten the other thresholds)
  once real-data calibration gives them meaning; flipping it by default
  before that would just break every run for no calibrated reason.
- **Not attempted this pass** — no formal packaging (`src/` layout,
  `pyproject.toml`): this is still a flat script collection at the repo
  root, alongside legacy exploratory `cl_file*.txt` dumps that fed the
  original refactor.
- No notification channel (Slack/email/PagerDuty) is wired to the
  structured alerts in `result.attrs["alerts"]` — the channel choice needs
  a human decision, not a default guess.
- The tenure safety cap (`scoring.min_tenure_years`, one-tier downgrade) is
  a direct per-agent port of the old cluster-level safety filter's
  tenure check; it has not been independently re-validated as the right
  rule for a per-agent, deterministic context.
- No alert fires on a high `is_anomaly` rate itself (only on `hdbscan`
  being unavailable) — `hdbscan_min_cluster_size`/`min_samples` are tuned
  for a large production population, so small/synthetic runs will flag
  most or all active agents as global anomalies (all noise, no clusters to
  run LOF over at all); that's an artifact of population size vs. those
  defaults, not a signal worth alerting on without a human-chosen
  threshold, which doesn't exist yet.
- LOF's `contamination="auto"` (stage 2 of anomaly detection) uses
  scikit-learn's fixed literature offset rather than a value fit to this
  population, and `lof_n_neighbors=20`/`lof_min_cluster_population=5` have
  not been tuned against real cluster sizes — all three are reasonable
  defaults, not calibrated ones. (They are, at least, validated —
  `_validate_lof_config` rejects out-of-range values before they reach
  scikit-learn rather than tuning them well.)
