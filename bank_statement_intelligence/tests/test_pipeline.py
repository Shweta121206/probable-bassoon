from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.categorization import ChartOfAccountsRetriever
from app.classification.gemini_classifier import GeminiClassifier
from app.classification.transaction_classifier import TransactionClassifier
from app.extraction.balance_extractor import extract_balance_history
from app.extraction.pdf_extractor import extract_pdf_text
from app.llm import TransactionExtractor
from app.metadata import extract_metadata
from app.reports import create_bank_statement_workbook
from app.validation import TransactionValidator

try:
    from app.memory import MemoryManager
    memory_dependencies_available = True
except ImportError:
    MemoryManager = None  # type: ignore[assignment]
    memory_dependencies_available = False


def _safe_print(message: str) -> None:
    print(message)
    sys.stdout.flush()


def _log_time(start: float, step: str) -> None:
    elapsed = time.perf_counter() - start
    _safe_print(f"{step} completed in {elapsed:.2f}s")


def _assert_date_format(transaction: dict[str, Any]) -> None:
    date_value = transaction.get("date")
    if not isinstance(date_value, str) or not len(date_value.split("/")) == 3:
        raise AssertionError(f"Transaction date not normalized: {date_value}")

    month, day, year = date_value.split("/")
    if len(month) != 2 or len(day) != 2 or len(year) != 2:
        raise AssertionError(f"Transaction date not normalized to MM/DD/YY: {date_value}")


def _transaction_summary(transaction: dict[str, Any]) -> str:
    return (
        f"{transaction.get('date', '')} | "
        f"{transaction.get('description', '')} | "
        f"{transaction.get('classification_name', '')} | "
        f"{transaction.get('classification_type', '')} | "
        f"{transaction.get('classification_confidence', '')}"
    )


def test_end_to_end_pipeline() -> None:
    logging.basicConfig(level=logging.INFO)

    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "test_inputs" / "December 2025 Bank Statement.pdf"
    transactions_output_path = project_root / "output" / "transactions.json"
    classified_output_path = project_root / "output" / "classified_transactions.json"
    excel_output_path = project_root / "output" / "December_2025_Classified.xlsx"

    if not pdf_path.exists():
        raise FileNotFoundError(f"Required input PDF not found: {pdf_path}")

    start = time.perf_counter()
    _safe_print("STEP 1: Extract PDF text")
    statement_text = extract_pdf_text(str(pdf_path))
    _safe_print("PDF loaded successfully.")
    _safe_print(f"Number of pages: {statement_text.count('--- Page')}")
    _safe_print(f"Characters extracted: {len(statement_text)}")
    _log_time(start, "STEP 1")

    start = time.perf_counter()
    _safe_print("STEP 2: Extract metadata")
    metadata = extract_metadata(statement_text)
    _safe_print(f"Bank Name: {metadata.bank_name or 'UNKNOWN'}")
    _safe_print(f"Account Holder: {metadata.account_holder or 'UNKNOWN'}")
    _safe_print(f"Account Number: {metadata.account_number or 'UNKNOWN'}")
    _safe_print(f"Statement Period: {metadata.statement_period_start or 'UNKNOWN'} to {metadata.statement_period_end or 'UNKNOWN'}")
    _safe_print(f"Statement Year: {metadata.statement_year or 'UNKNOWN'}")
    _log_time(start, "STEP 2")

    start = time.perf_counter()
    _safe_print("STEP 3: Extract transactions")
    transaction_extractor = TransactionExtractor()
    raw_transactions = transaction_extractor.extract_transactions(statement_text)
    total_transactions = len(raw_transactions)
    total_deposits = sum(1 for transaction in raw_transactions if transaction.amount > 0)
    total_withdrawals = sum(1 for transaction in raw_transactions if transaction.amount < 0)
    _safe_print(f"Total Transactions: {total_transactions}")
    _safe_print(f"Total Deposits: {total_deposits}")
    _safe_print(f"Total Withdrawals: {total_withdrawals}")
    _log_time(start, "STEP 3")

    start = time.perf_counter()
    _safe_print("STEP 4: Normalize dates")
    if metadata.statement_year is None:
        raise ValueError("Statement year could not be determined for normalization")

    validator = TransactionValidator()
    validated_transactions = validator.validate_transactions(raw_transactions, metadata.statement_year)
    for transaction in validated_transactions:
        _assert_date_format(transaction.to_dict())
    _safe_print("All transactions contain MM/DD/YY dates")
    _log_time(start, "STEP 4")

    start = time.perf_counter()
    _safe_print("STEP 5: Save transactions.json")
    transactions_output_path.parent.mkdir(parents=True, exist_ok=True)
    transactions_json = [transaction.to_dict() for transaction in validated_transactions]
    transactions_output_path.write_text(json.dumps(transactions_json, indent=2), encoding="utf-8")
    _safe_print(f"Saved transactions.json to {transactions_output_path}")
    _log_time(start, "STEP 5")

    start = time.perf_counter()
    _safe_print("STEP 6: Load Chart of Accounts")
    chart_retriever = ChartOfAccountsRetriever()
    _safe_print(f"Total Accounts Loaded: {len(chart_retriever._accounts)}")
    _log_time(start, "STEP 6")

    start = time.perf_counter()
    _safe_print("STEP 7: Load Historical Examples")
    from app.classification.example_loader import ExampleLoader

    example_loader = ExampleLoader()
    _safe_print(f"Historical Examples Loaded: {len(example_loader.get_examples())}")
    _log_time(start, "STEP 7")

    start = time.perf_counter()
    _safe_print("STEP 8: Build Memory Engine and retrieve similar transactions")
    if not memory_dependencies_available or MemoryManager is None:
        _safe_print("Memory engine dependencies are unavailable; skipping memory retrieval.")
    elif not example_loader.get_examples():
        _safe_print("No historical examples available; skipping memory retrieval.")
    else:
        memory_manager = MemoryManager()
        similar_transactions = memory_manager.get_similar_examples(validated_transactions[0].description, top_k=5)
        if not similar_transactions:
            _safe_print("No similar historical transactions were found.")
        else:
            _safe_print("Top 5 Similar Transactions for first extracted transaction:")
            for similar in similar_transactions[:5]:
                _safe_print(f"- {similar.description} -> {similar.classification_name} / {similar.classification_type}")
    _log_time(start, "STEP 8")

    start = time.perf_counter()
    _safe_print("STEP 9: Classify all transactions using Gemini")
    classifier = TransactionClassifier()
    classified_transactions = classifier.classify_transactions([transaction.to_dict() for transaction in validated_transactions])
    for transaction in classified_transactions:
        _safe_print(
            f"{transaction['date']} | {transaction['description']} | "
            f"{transaction['classification_name']} | {transaction['classification_type']} | {transaction['classification_confidence']}"
        )
    _log_time(start, "STEP 9")

    start = time.perf_counter()
    _safe_print("STEP 10: Save classified_transactions.json")
    classified_output_path.write_text(json.dumps(classified_transactions, indent=2), encoding="utf-8")
    _safe_print(f"Saved classified_transactions.json to {classified_output_path}")
    _log_time(start, "STEP 10")

    start = time.perf_counter()
    _safe_print("STEP 11: Generate Excel output")
    create_bank_statement_workbook(
        transactions=classified_transactions,
        metadata=metadata,
        balance_history=extract_balance_history(statement_text),
        output_path=str(excel_output_path),
    )
    _safe_print(f"Output: {excel_output_path}")
    _log_time(start, "STEP 11")

    total_time = time.perf_counter() - start
    average_confidence = (
        sum(transaction.get("classification_confidence", 0) for transaction in classified_transactions) /
        len(classified_transactions)
        if classified_transactions
        else 0
    )
    highest_confidence = max((transaction.get("classification_confidence", 0) for transaction in classified_transactions), default=0)
    lowest_confidence = min((transaction.get("classification_confidence", 0) for transaction in classified_transactions), default=0)

    _safe_print("STEP 12: Summary")
    _safe_print(f"Transactions Processed: {len(classified_transactions)}")
    _safe_print(f"Average Confidence: {average_confidence:.2f}")
    _safe_print(f"Highest Confidence: {highest_confidence}")
    _safe_print(f"Lowest Confidence: {lowest_confidence}")
    _safe_print(f"Total Processing Time: {total_time:.2f}s")


if __name__ == "__main__":
    try:
        test_end_to_end_pipeline()
        _safe_print("\n\033[92mSUCCESS: End-to-end pipeline completed successfully.\033[0m")
    except Exception as exc:
        _safe_print(f"ERROR: {exc}")
        raise
