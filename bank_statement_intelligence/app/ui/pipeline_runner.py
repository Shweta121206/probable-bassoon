from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.extraction.balance_extractor import extract_balance_history
from app.extraction.pdf_extractor import extract_pdf_text
from app.llm.transaction_extractor import TransactionExtractor
from app.metadata import extract_metadata
from app.reports import create_bank_statement_workbook
from app.retrieval.retrieval_classifier import RetrievalClassifier
from app.validation import TransactionValidator

logger = logging.getLogger(__name__)
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FOLDER = Path(os.getenv("INPUT_FOLDER", str(PROJECT_ROOT / "test_inputs")))
OUTPUT_FOLDER = Path(os.getenv("OUTPUT_FOLDER", str(PROJECT_ROOT / "output")))
ENABLE_GEMINI_VERIFICATION = os.getenv("ENABLE_GEMINI_VERIFICATION", "true").strip().lower() in {"1", "true", "yes", "on"}


def get_config() -> dict[str, Any]:
    return {
        "enable_gemini_verification": ENABLE_GEMINI_VERIFICATION,
        "input_folder": str(INPUT_FOLDER),
        "output_folder": str(OUTPUT_FOLDER),
    }


def run_pipeline(pdf_path: str | os.PathLike[str]) -> dict[str, Any]:
    source_path = Path(pdf_path)
    if not source_path.exists():
        raise FileNotFoundError(f"PDF not found: {source_path}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    INPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    statement_text = extract_pdf_text(str(source_path))
    metadata = extract_metadata(statement_text)

    extractor = TransactionExtractor()
    raw_transactions = extractor.extract_transactions(statement_text)
    validator = TransactionValidator()
    validated_transactions = validator.validate_transactions(raw_transactions, metadata.statement_year or 2000)

    transactions_json = [transaction.to_dict() for transaction in validated_transactions]
    transactions_output = OUTPUT_FOLDER / "transactions.json"
    transactions_output.write_text(json.dumps(transactions_json, indent=2), encoding="utf-8")

    historical_examples_path = PROJECT_ROOT / "config" / "historical_examples.xlsx"
    classifier = RetrievalClassifier(historical_examples_path=historical_examples_path)
    classified_transactions = classifier.classify_transactions(transactions_json)
    classified_output = OUTPUT_FOLDER / "classified_transactions.json"
    classified_output.write_text(json.dumps(classified_transactions, indent=2), encoding="utf-8")

    balance_history = extract_balance_history(statement_text)
    excel_path = OUTPUT_FOLDER / "streamlit_report.xlsx"
    create_bank_statement_workbook(
        transactions=classified_transactions,
        metadata=metadata,
        balance_history=balance_history,
        output_path=str(excel_path),
    )

    summary = {
        "total_transactions": len(classified_transactions),
        "deposits": sum(1 for item in classified_transactions if float(item.get("credit") or 0) > 0),
        "withdrawals": sum(1 for item in classified_transactions if float(item.get("debit") or 0) > 0),
        "total_credits": sum(float(item.get("credit") or 0) for item in classified_transactions),
        "total_debits": sum(float(item.get("debit") or 0) for item in classified_transactions),
        "average_confidence": round(
            sum(int(item.get("confidence", 0)) for item in classified_transactions) / len(classified_transactions),
            2,
        ) if classified_transactions else 0,
        "high_confidence_count": sum(1 for item in classified_transactions if int(item.get("confidence", 0)) >= 90),
        "medium_confidence_count": sum(1 for item in classified_transactions if 70 <= int(item.get("confidence", 0)) < 90),
        "low_confidence_count": sum(1 for item in classified_transactions if int(item.get("confidence", 0)) < 70),
    }

    return {
        "metadata": {
            "bank_name": metadata.bank_name,
            "account_holder": metadata.account_holder,
            "account_number": metadata.account_number,
            "statement_period": f"{metadata.statement_period_start or ''} - {metadata.statement_period_end or ''}".strip(" -"),
            "statement_year": metadata.statement_year,
        },
        "transactions": classified_transactions,
        "summary": summary,
        "excel_path": str(excel_path),
        "warnings": [],
        "errors": [],
    }
