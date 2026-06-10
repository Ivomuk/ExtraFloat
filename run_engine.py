"""
ExtraFloat limit engine — CLI runner.

Usage:
    python run_engine.py \
        --transaction  data/transaction_capacity_features_sample.txt \
        --borrower     data/borrower_credit_limit_expected_final.txt \
        --loan-summary data/loan_summary.csv \
        --output       data/engine_output.csv

All arguments are optional; the defaults shown above are used when omitted.
--loan-summary is the only file that may not exist yet — the engine zero-fills
loan features and warns if the file is missing or empty.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from extrafloat.io.extrafloat_data_loaders import (
    load_transaction_capacity_features,
    load_borrower_limit_features,
    load_loan_summary_recent_features,
)
from extrafloat.engine.extrafloat_limit_engine_features import (
    build_extrafloat_limit_engine_features,
)
from extrafloat.engine.run_extrafloat_limit_engine import run_extrafloat_limit_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger("run_engine")

DEFAULT_TRANSACTION  = "data/transaction_capacity_features_sample.txt"
DEFAULT_BORROWER     = "data/borrower_credit_limit_expected_final.txt"
DEFAULT_LOAN_SUMMARY = "data/loan_summary.csv"
DEFAULT_OUTPUT       = "data/engine_output.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Run the ExtraFloat limit engine.")
    p.add_argument("--transaction",  default=DEFAULT_TRANSACTION)
    p.add_argument("--borrower",     default=DEFAULT_BORROWER)
    p.add_argument("--loan-summary", default=DEFAULT_LOAN_SUMMARY, dest="loan_summary")
    p.add_argument("--output",       default=DEFAULT_OUTPUT)
    return p.parse_args()


_LOAN_SUMMARY_EMPTY_COLS = [
    "msisdn", "snapshot_dt", "last_disbursement_date", "last_repayment_date",
    "disbursement_vol_1m", "disbursement_val_1m",
    "repayment_vol_1m", "repayment_val_1m", "penalties_1m",
    "disbursement_val_3m", "repayment_val_3m", "penalties_3m",
]


def _load_loan_summary(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        logger.warning(
            "loan-summary file not found (%s) — engine will run without loan "
            "features (all loan columns defaulting to 0).",
            path,
        )
        return pd.DataFrame(columns=_LOAN_SUMMARY_EMPTY_COLS)
    return load_loan_summary_recent_features(p)


def main():
    args = parse_args()

    logger.info("Loading transaction capacity features from %s", args.transaction)
    transaction_df = load_transaction_capacity_features(args.transaction)

    logger.info("Loading borrower limit features from %s", args.borrower)
    borrower_df = load_borrower_limit_features(args.borrower)

    logger.info("Loading loan summary features from %s", args.loan_summary)
    loan_summary_df = _load_loan_summary(args.loan_summary)

    logger.info("Building engine features (%d borrower rows, %d transaction rows, %d loan rows)",
                len(borrower_df), len(transaction_df), len(loan_summary_df))
    features_df = build_extrafloat_limit_engine_features(
        borrower_df, transaction_df, loan_summary_df
    )

    logger.info("Running limit engine on %d rows", len(features_df))
    result_df = run_extrafloat_limit_engine(features_df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    logger.info("Output written to %s  (%d rows, %d columns)", out_path, len(result_df), len(result_df.columns))


if __name__ == "__main__":
    sys.exit(main())
