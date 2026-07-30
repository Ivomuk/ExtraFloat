# CreditRisk pipeline — two-phase workflow
#
# Phase 1 (train): Build PD model artifacts from historical agent snapshots.
# Phase 2 (run):   Score agents and assign credit limits.
#
# Quick start:
#   make install
#   make train   TRAIN=data/agent_snapshot_train.csv VAL=data/agent_snapshot_val.csv
#   make run

.PHONY: install train run test test-pipeline check-artifacts help

# ── Configurable inputs (override on the command line) ──────────────────────
TRAIN            ?= data/agent_snapshot_train.csv
VAL              ?= data/agent_snapshot_val.csv
REPAYMENT        ?= data/repayments.csv
TRAIN_SNAPSHOT   ?= 20250831
VAL_SNAPSHOT     ?= 20251130
TRAIN_CUTOFF     ?= 2025-08-31
ARTIFACTS_DIR    ?= pd_model/artifacts
CHAMPION         ?= xgb

PD_MODEL_FILE    ?= data/agent_profile_snapshot.csv
TRANSACTION_FILE ?= data/transaction_capacity.csv
LOAN_FILE        ?= data/loan_summary.csv
BORROWER_FILE    ?= data/borrower_credit.csv
OUTPUT           ?= output/credit_risk_output.csv

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev,monitor]"

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
		--pd-model-file     $(PD_MODEL_FILE) \
		--transaction-file  $(TRANSACTION_FILE) \
		--loan-file         $(LOAN_FILE) \
		--borrower-file     $(BORROWER_FILE) \
		--artifacts-dir     $(ARTIFACTS_DIR) \
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
	@echo "  make train            Train the PD model (produces artifacts)"
	@echo "  make run              Run the end-to-end credit risk pipeline"
	@echo "  make test             Run all tests"
	@echo "  make test-pipeline    Run pipeline integration tests only"
	@echo "  make check-artifacts  Verify PD model artifacts exist"
	@echo ""
	@echo "Key variables (override with make <target> VAR=value):"
	@echo "  TRAIN            Training snapshot CSV  (default: $(TRAIN))"
	@echo "  VAL              Validation snapshot CSV (default: $(VAL))"
	@echo "  ARTIFACTS_DIR    Model artifacts dir     (default: $(ARTIFACTS_DIR))"
	@echo "  PD_MODEL_FILE    Agent profile snapshot  (default: $(PD_MODEL_FILE))"
	@echo "  OUTPUT           Pipeline output CSV     (default: $(OUTPUT))"
	@echo ""
