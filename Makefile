# CreditRisk pipeline — three-phase workflow
#
# Phase 0 (calibrate): Build the capacity scorecard from a labelled agent snapshot.
# Phase 1 (train):     Build PD model artifacts from historical agent snapshots.
# Phase 2 (run):       Score agents: segmentation → PD model → credit limit engine.
#
# Quick start:
#   make install
#   make calibrate   TRANSACTION_FILE=data/agent_profile_snapshot.csv
#   make train       TRAIN=data/agent_snapshot_train.csv VAL=data/agent_snapshot_val.csv
#   make run

.PHONY: install calibrate train run test test-pipeline check-artifacts help

# ── Configurable inputs (override on the command line) ──────────────────────
TRAIN            ?= data/agent_snapshot_train.csv
VAL              ?= data/agent_snapshot_val.csv
REPAYMENT        ?= data/repayments.csv
TRAIN_SNAPSHOT   ?= 20250831
VAL_SNAPSHOT     ?= 20251130
TRAIN_CUTOFF     ?= 2025-08-31
ARTIFACTS_DIR    ?= pd_model/artifacts
CHAMPION         ?= xgb

TRANSACTION_FILE ?= data/agent_profile_snapshot.csv
LOAN_FILE        ?= data/loan_summary.csv
BORROWER_FILE    ?= data/borrower_credit.csv
SCORECARD_PATH   ?= scorecards/capacity_scorecard_v1.json
OUTPUT           ?= output/credit_risk_output.csv

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev,monitor]"

# ── Phase 0: Calibrate the capacity scorecard ────────────────────────────────
calibrate:
	python calibrate_scorecard.py \
		--input-file    $(TRANSACTION_FILE) \
		--output-path   $(SCORECARD_PATH)

# ── Phase 1: Train the PD model ─────────────────────────────────────────────
train:
	python -m pd_model.run_pipeline \
		--train-file            $(TRAIN) \
		--val-file              $(VAL) \
		--repayment-file        $(REPAYMENT) \
		--train-snapshot-date   $(TRAIN_SNAPSHOT) \
		--val-snapshot-date     $(VAL_SNAPSHOT) \
		--train-cutoff          $(TRAIN_CUTOFF) \
		--output-dir            $(ARTIFACTS_DIR) \
		--champion              $(CHAMPION)

# ── Phase 2: Run the credit risk pipeline ───────────────────────────────────
run: check-artifacts
	python run_credit_risk_pipeline.py \
		--transaction-file  $(TRANSACTION_FILE) \
		--loan-file         $(LOAN_FILE) \
		--borrower-file     $(BORROWER_FILE) \
		--repayment-file    $(REPAYMENT) \
		--artifacts-dir     $(ARTIFACTS_DIR) \
		--scorecard-path    $(SCORECARD_PATH) \
		--champion          $(CHAMPION) \
		--output            $(OUTPUT)

# ── Tests ────────────────────────────────────────────────────────────────────
test:
	python -m pytest tests/ pd_model/tests/ -v

test-pipeline:
	python -m pytest tests/pipeline/ -v

# ── Utilities ────────────────────────────────────────────────────────────────
check-artifacts:
	@python -c "\
from pathlib import Path; \
from run_credit_risk_pipeline import _check_artifacts; \
_check_artifacts(Path('$(ARTIFACTS_DIR)')); \
print('All artifacts present in $(ARTIFACTS_DIR).')"

help:
	@echo ""
	@echo "CreditRisk pipeline — available targets:"
	@echo ""
	@echo "  make install          Install package and dev dependencies"
	@echo "  make calibrate        Phase 0 — calibrate capacity scorecard"
	@echo "  make train            Phase 1 — train PD model (produces artifacts)"
	@echo "  make run              Phase 2 — segmentation → PD → credit limit engine"
	@echo "  make test             Run all tests"
	@echo "  make test-pipeline    Run pipeline integration tests only"
	@echo "  make check-artifacts  Verify PD model artifacts exist"
	@echo ""
	@echo "Key variables (override with make <target> VAR=value):"
	@echo "  TRAIN            Training snapshot CSV     (default: $(TRAIN))"
	@echo "  VAL              Validation snapshot CSV    (default: $(VAL))"
	@echo "  ARTIFACTS_DIR    Model artifacts dir        (default: $(ARTIFACTS_DIR))"
	@echo "  TRANSACTION_FILE Agent profile snapshot     (default: $(TRANSACTION_FILE))"
	@echo "  SCORECARD_PATH   Capacity scorecard JSON    (default: $(SCORECARD_PATH))"
	@echo "  REPAYMENT        Repayment history CSV      (default: $(REPAYMENT))"
	@echo "  OUTPUT           Pipeline output CSV        (default: $(OUTPUT))"
	@echo ""
