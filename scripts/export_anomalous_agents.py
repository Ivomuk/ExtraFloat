"""
Dumps the agents flagged is_anomaly=True from a segmentation output
(agent_segments.csv) to a standalone CSV for manual analysis -- joined back
against the raw agent mart file so the KPI columns that might explain *why*
an agent got flagged are actually visible, since agent_segments.csv itself
only carries capacity-scoring/anomaly columns, not raw KPIs.

Sorted so the most informative rows come first: global anomalies (HDBSCAN
noise -- doesn't fit any dense cluster at all) before local anomalies
(fit a cluster but stood out within it), and within local anomalies, most
negative lof_score first (more negative = more anomalous per
sklearn.neighbors.LocalOutlierFactor's convention).

Usage:
    python scripts\\export_anomalous_agents.py ^
        --segments segmentation_outputs\\agent_segments.csv ^
        --agents data\\mfs_daily_agent_mart_20260731_retail_filtered.csv ^
        --output segmentation_outputs\\anomalous_agents.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segments", required=True, metavar="PATH",
                   help="Path to agent_segments.csv from a segmentation run.")
    p.add_argument("--agents", required=True, metavar="PATH",
                   help="Path to the raw agent mart CSV used for that run (for KPI context).")
    p.add_argument("--output", required=True, metavar="PATH")
    args = p.parse_args(argv)

    segments_path = Path(args.segments)
    agents_path = Path(args.agents)
    if not segments_path.exists():
        sys.exit(f"ERROR: file not found: {segments_path}")
    if not agents_path.exists():
        sys.exit(f"ERROR: file not found: {agents_path}")

    segments_df = pd.read_csv(segments_path)
    print(f"Loaded {len(segments_df):,} rows from {segments_path}")

    if "is_anomaly" not in segments_df.columns:
        sys.exit(
            "ERROR: 'is_anomaly' column not found -- was this run with "
            "clustering.enable_anomaly_detection=True (the default)?"
        )

    agents_df = pd.read_csv(agents_path)
    print(f"Loaded {len(agents_df):,} rows from {agents_path}")

    merge_keys = [c for c in ("agent_msisdn", "pos_msisdn") if c in segments_df.columns and c in agents_df.columns]
    if not merge_keys:
        sys.exit(
            "ERROR: no shared key columns (agent_msisdn/pos_msisdn) found "
            "between --segments and --agents -- can't join KPI context."
        )
    print(f"Joining on: {merge_keys}")

    # Avoid duplicate non-key columns from the raw mart clobbering the
    # segmentation output's own columns (e.g. both might carry a `segment`
    # column with different meanings).
    kpi_cols = [c for c in agents_df.columns if c in merge_keys or c not in segments_df.columns]
    merged = segments_df.merge(agents_df[kpi_cols], on=merge_keys, how="left")

    anomalous = merged.loc[merged["is_anomaly"].fillna(False).astype(bool)].copy()

    # is_anomaly is always False for dormant agents (anomaly detection only
    # runs on active ones), so the rate over ALL rows understates the real
    # rate among agents it could actually flag. Recompute the active mask
    # the same way production does, so the printed rate matches what
    # check_diagnostic_stages_output.py reports rather than the diluted
    # whole-population figure.
    from segmentation.extrafloat_segmentation_features import prepare_features
    from segmentation.extrafloat_segmentation_pipeline import _get_clustering_config, _identify_dormant_mask

    try:
        feat_df, _, _, _ = prepare_features(agents_df)
        cfg = _get_clustering_config(None)
        active_mask = ~_identify_dormant_mask(feat_df, cfg)
        n_active = int(active_mask.sum())
    except Exception as exc:  # noqa: BLE001 -- this is a reporting nicety, never fatal
        print(f"(couldn't recompute active population for a rate breakdown: {exc})")
        n_active = None

    print(f"\n{len(anomalous):,} agents flagged is_anomaly=True")
    print(f"  {len(anomalous) / len(merged):.1%} of all {len(merged):,} rows (dormant agents included, diluted)")
    if n_active:
        print(f"  {len(anomalous) / n_active:.1%} of {n_active:,} ACTIVE agents (the real, undiluted rate)")

    # Global anomalies (didn't fit any cluster) first, then local anomalies
    # sorted by how anomalous LOF found them (most negative lof_score first).
    anomalous["_sort_global_first"] = ~anomalous["is_global_anomaly"].fillna(False).astype(bool)
    anomalous = anomalous.sort_values(
        by=["_sort_global_first", "lof_score"], ascending=[True, True], na_position="last"
    ).drop(columns="_sort_global_first")

    print("\nBreakdown:")
    print(f"  global anomalies (HDBSCAN noise): {int(anomalous['is_global_anomaly'].fillna(False).sum()):,}")
    print(f"  local anomalies (LOF within cluster): {int(anomalous['is_local_anomaly'].fillna(False).sum()):,}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anomalous.to_csv(out_path, index=False)
    print(f"\nWrote {len(anomalous):,} rows to {out_path}")


if __name__ == "__main__":
    main()
