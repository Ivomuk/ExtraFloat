# ExtraFloat Agent Segmentation

Clusters MTN Mobile Money (Uganda) agents into 8 business tiers based on
transactional KPIs, using a two-stage KMeans → GMM+HDBSCAN ensemble. This
document covers the segmentation package only — see the note on
[integration boundary](#integration-boundary) below for how it relates to
the rest of the repository.

## Package layout

| File | Role |
|---|---|
| `extrafloat_segmentation_features.py` | Feature engineering: cleaning → date/premium/interaction features → log1p+winsorize → correlation pruning → RobustScaler+PCA |
| `extrafloat_segmentation_pipeline.py` | Core clustering algorithm (see below) |
| `extrafloat_segmentation_profiling.py` | Pack-based KPI profiling, quantile tiering, sub-labels, whitelist/blacklist merge |
| `extrafloat_segmentation_validation.py` | Purity/ARI/feature-importance validation toolkit + the automated quality gate |
| `extrafloat_segmentation_drift.py` | PSI/KL feature-drift monitoring vs. a saved baseline |
| `extrafloat_segmentation_viz.py` | Plotting helpers (matplotlib/seaborn, soft dependency) |
| `run_extrafloat_segmentation.py` | Orchestration entry point + CLI |
| `test_extrafloat_segmentation.py` | Pytest suite |

## Pipeline

```
agents_df
  │
  ▼
[1] prepare_features()            feature engineering + PCA
  │
  ▼
[1b] drift check (optional)       PSI/KL vs. saved baseline
  │
  ▼
[2] run_clustering_pipeline()     KMeans(k=6, all agents)
  │                                → composite-score dormant detection
  │                                → KMeans round 2 (active agents)
  │                                → GMM (BIC search) + UMAP→HDBSCAN (active agents)
  │                                → ensemble label "GMM_{id}__{tier}"
  │                                → composite score → 8-tier BUSINESS_SEGMENTS
  │
  ▼
[2b] full segment stability (optional, expensive — off by default)
  │
  ▼
[3] build_cluster_pack_profiles() KPI pack means/lifts per cluster
  │
  ▼
[4] build_cluster_tiers()         Platinum/Gold/Silver/Bronze + safety-filter downgrades
  │
  ▼
[5] merge_reference_lists()       optional whitelist/blacklist enrichment (agent_category)
  │
  ▼
[5b] run_quality_gate()           automated sanity/quality checks
  │
  ▼
[6] output trimming               → agent_msisdn, segment, tier, ...
```

## Business segments

`BUSINESS_SEGMENTS` (from `extrafloat_segmentation_pipeline.py`), lowest to
highest value:

```
Below Threshold, New Bronze, Bronze, Silver, Gold, Platinum, Titanium, Diamond
```

An agent's `segment` is assigned by ranking ensemble clusters on a composite
score (50% value / 30% activity / 20% efficiency by default — configurable
via `clustering.composite_weights`) and quantile-mapping the ranked clusters
onto these 8 tiers. Dormant agents (identified by a weighted composite
inactivity score) always land in "Below Threshold" regardless of the ranking.

These weights and tier boundaries are hand-set defaults, not (yet) empirically
validated against ground truth by default — `clustering.optimize_composite_weights`
can flip on an ARI-based grid search against an `agent_category` column when
available, but that is opt-in, not the default.

## Running it

```bash
pip install -r requirements-segmentation-dev.txt
python run_extrafloat_segmentation.py --agents agents.csv --output segmentation_outputs/
```

Or from Python:

```python
from run_extrafloat_segmentation import run_extrafloat_segmentation, DEFAULT_SEGMENTATION_CONFIG
import pandas as pd

agents_df = pd.read_csv("agents.csv")
result = run_extrafloat_segmentation(agents_df, config=DEFAULT_SEGMENTATION_CONFIG)
result[["agent_msisdn", "segment", "hdb_tier", "ensemble_cluster"]].head()
```

Required input columns are listed in `extrafloat_segmentation_features.REQUIRED_COLUMNS`.

## Reliability features

- **Dependency guard rails** (`clustering.require_hdbscan` /
  `clustering.require_umap`, default `True`): a missing `hdbscan` or
  `umap-learn` install raises `RuntimeError` at the start of
  `run_clustering_pipeline` rather than silently degrading cluster quality.
  Set to `False` only for a deliberately degraded local/dev run — the
  resulting `result.attrs["degraded_mode"]` flags will be `True`.
- **Quality gate** (`extrafloat_segmentation_validation.run_quality_gate`,
  wired into orchestration step 5b, config under `quality_gate`): runs
  unsupervised sanity checks (segment diversity, HDBSCAN noise share,
  "Below Threshold" share among active agents) on every run, plus
  purity/ARI checks against `agent_category` when a whitelist/blacklist
  merge supplied one. Reports via `result.attrs["quality_gate"]`; set
  `quality_gate.fail_on_breach=True` to raise instead of only logging
  `CRITICAL` once thresholds are calibrated against real data.
- **Stability reporting**: `result.attrs["stability_report"]` (always
  computed) measures Round-1 KMeans reproducibility across seeds.
  `result.attrs["final_stability_report"]` (opt-in via
  `clustering.full_stability_check=True`, off by default because it re-runs
  the full active-agent pipeline `stability_n_seeds` times) measures
  reproducibility of the actual `segment` an agent is assigned — the number
  that matters downstream.
- **Drift detection**: PSI/KL divergence of input features against a saved
  baseline (`extrafloat_segmentation_drift.py`), wired into orchestration
  step 1b when `drift.baseline_path` is configured.
- **Run manifests** (`output.save_run_manifest`, default `True`, written
  alongside `agent_segments.csv` as `run_manifest.json` whenever
  `output.output_dir` is set): records the resolved config, package
  versions, git commit SHA, input/output row counts, and every report
  above (stability, drift, quality gate, degraded mode) for a given run —
  so a credit-tier decision can be traced back to exactly what produced
  it. See `_build_run_manifest()`.
- **Structured alerts** (`result.attrs["alerts"]`, always computed): rolls
  degraded-mode/quality-gate/drift/final-stability findings into a flat
  list of `{severity, source, message}` dicts. No notification channel
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
or labeled dataset available in this repo to calibrate against, which is
why the two remaining rigor items below are deliberately un-started rather
than half-built against nothing.

## Integration boundary

This package's output (`segment`, `tier`, `hdb_tier`, `ensemble_cluster`
columns) is intended to feed a separate downstream credit/float-limit
engine in this repository. **The current wiring between the two has not
been verified as part of this review** — the limit engine derives its own
tier multipliers from a differently-named/valued mechanism, and whether
that is deliberately independent or should be consuming this package's
`segment` column directly is an open question that needs a decision from
whoever owns both systems before further integration work is scoped. This
document intentionally does not describe the limit engine's internals.

## Known limitations / open items

- **Blocked on real data** — deliberately not started, not half-built:
  - Business segment boundaries and composite-score weights are hand-set
    defaults; `clustering.optimize_composite_weights` runs an ARI-based
    grid search against `agent_category` ground truth, but flipping it on
    by default needs a real labeled dataset to validate the result isn't
    worse than the hand-set weights — none exists in this repo yet.
  - No temporal / out-of-time validation (month-over-month tier-churn
    checking) — needs real multi-month snapshots to build and calibrate
    against; nothing in this repo currently has more than one snapshot date.
- **Not attempted this pass** — no formal packaging (`src/` layout,
  `pyproject.toml`): this is still a flat script collection at the repo
  root, alongside legacy exploratory `cl_file*.txt`/`file*.txt` dumps that
  fed the original refactor and a concurrently in-flight branch
  (`pd-model-review`) touching the same file layout, which makes a
  repo-wide restructure a poor fit for an unreviewed drive-by change.
- No notification channel (Slack/email/PagerDuty) is wired to the
  structured alerts in `result.attrs["alerts"]` — the channel choice needs
  a human decision, not a default guess.
