@echo off
echo === Running agent segmentation WITH diagnostic clustering (research/exploratory) ===
echo GMM/HDBSCAN clustering + two-stage anomaly detection (HDBSCAN + LOF).
echo This NEVER affects capacity_tier -- diagnostics-only, per
echo extrafloat_segmentation_pipeline.py's own design (see SEGMENTATION_README.md).
echo Requires hdbscan + umap-learn installed (confirmed present on this machine).
echo --keep-intermediate keeps diag_cluster_id_gmm/diag_cluster_hdb_raw in the
echo output CSV so you can inspect the raw cluster assignments, not just the
echo derived diag_hdb_tier/diag_ensemble_cluster/diag_is_anomaly columns.

python -m segmentation.run_extrafloat_segmentation --agents data\mfs_daily_agent_mart_20260731_retail_filtered.csv --scorecard scorecards\capacity_scorecard_v1.json --config segmentation_diagnostics_test.json --keep-intermediate --output segmentation_outputs_diagnostics

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: segmentation with diagnostics failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Complete. See segmentation_outputs_diagnostics\agent_segments.csv --
echo new columns: diag_cluster_id_gmm, diag_cluster_hdb_raw, diag_hdb_tier,
echo diag_ensemble_cluster, diag_is_anomaly (research-only, never fed into
echo capacity_tier or the credit limit engine).
pause
