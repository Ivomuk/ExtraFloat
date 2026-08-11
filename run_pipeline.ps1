# XtraFloat Pipeline — Full Three-Phase Run
# ============================================
# Phase 0: Calibrate capacity scorecard
# Phase 1: Train PD model
# Phase 2: Score agents (segmentation -> PD -> credit limit engine)
#
# Usage:
#   .\run_pipeline.ps1                  # run all three phases
#   .\run_pipeline.ps1 -Phase run       # Phase 2 only (requires trained artifacts)
#   .\run_pipeline.ps1 -Phase train     # Phase 1 only
#   .\run_pipeline.ps1 -Phase calibrate # Phase 0 only

param(
    [ValidateSet("all", "calibrate", "train", "run")]
    [string]$Phase = "all"
)

# ── Input files (edit these paths to match your exports) ─────────────────────
$TransactionFile  = "data\agent_profile_snapshot.csv"
$LoanFile         = "data\loan_summary.csv"
$BorrowerFile     = "data\borrower_credit.csv"
$RepaymentFile    = "data\repayments.csv"
$TrainFile        = "data\agent_snapshot_train.csv"
$ValFile          = "data\agent_snapshot_val.csv"

# ── Output paths ──────────────────────────────────────────────────────────────
$ScorecardPath    = "scorecards\capacity_scorecard_v1.json"
$ArtifactsDir     = "pd_model\artifacts"
$Output           = "output\credit_risk_output.csv"

# ── Training dates ────────────────────────────────────────────────────────────
$TrainSnapshotDate = "20250831"
$ValSnapshotDate   = "20251130"
$TrainCutoff       = "2025-08-31"
$Champion          = "xgb"

# ── Helpers ───────────────────────────────────────────────────────────────────
function Run-Phase($label, $cmd) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " $label" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: $label failed (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "DONE: $label" -ForegroundColor Green
}

# ── Phase 0: Calibrate scorecard ─────────────────────────────────────────────
if ($Phase -eq "all" -or $Phase -eq "calibrate") {
    New-Item -ItemType Directory -Force -Path (Split-Path $ScorecardPath) | Out-Null
    Run-Phase "Phase 0: Calibrate capacity scorecard" `
        "python calibrate_scorecard.py --agents `"$TransactionFile`" --out `"$ScorecardPath`" --force"
}

# ── Phase 1: Train PD model ───────────────────────────────────────────────────
if ($Phase -eq "all" -or $Phase -eq "train") {
    Run-Phase "Phase 1: Train PD model" `
        "python -m pd_model.run_pipeline --train-file `"$TrainFile`" --val-file `"$ValFile`" --repayment-file `"$RepaymentFile`" --train-snapshot-date $TrainSnapshotDate --val-snapshot-date $ValSnapshotDate --train-cutoff $TrainCutoff --output-dir `"$ArtifactsDir`" --champion $Champion"
}

# ── Phase 2: Score agents ─────────────────────────────────────────────────────
if ($Phase -eq "all" -or $Phase -eq "run") {
    New-Item -ItemType Directory -Force -Path (Split-Path $Output) | Out-Null
    Run-Phase "Phase 2: Score agents" `
        "python run_credit_risk_pipeline.py --transaction-file `"$TransactionFile`" --loan-file `"$LoanFile`" --borrower-file `"$BorrowerFile`" --repayment-file `"$RepaymentFile`" --artifacts-dir `"$ArtifactsDir`" --scorecard-path `"$ScorecardPath`" --output `"$Output`""
}

Write-Host ""
Write-Host "Pipeline complete. Output: $Output" -ForegroundColor Green
