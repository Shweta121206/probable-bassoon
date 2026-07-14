"""Phase 5 test for transaction validation and balance mapping."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.extraction.balance_extractor import extract_balance_history
from app.extraction.pdf_extractor import extract_pdf_text
from app.llm import TransactionExtractor
from app.metadata import extract_metadata
from app.models import Transaction
from app.validation import TransactionValidator

logger = logging.getLogger(__name__)

PDF_FILENAME = "December 2025 Bank Statement.pdf"


def main() -> None:
    """Run the full Phase 5 validation workflow."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    project_root = Path(__file__).resolve().parent
    pdf_path = project_root / PDF_FILENAME
    output_path = project_root / "output" / "final_transactions.json"

    try:
        statement_text = extract_pdf_text(str(pdf_path))
        metadata = extract_metadata(statement_text)
        if metadata.statement_year is None:
            raise ValueError("statement year is required for transaction validation")

        extracted_transactions = TransactionExtractor().extract_transactions(statement_text)
        balance_history = extract_balance_history(statement_text)

        validator = TransactionValidator()
        valid_transactions = validator.validate_transactions(
            extracted_transactions,
            metadata.statement_year,
        )
        final_transactions = validator.merge_balance_history(
            valid_transactions,
            balance_history,
        )

        final_records = [_to_final_dict(transaction) for transaction in final_transactions]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(final_records, indent=2), encoding="utf-8")

        _print_summary(
            total_extracted=len(extracted_transactions),
            valid_count=len(final_transactions),
            invalid_count=len(extracted_transactions) - len(valid_transactions),
            transactions=final_transactions,
        )
        _print_sample(final_records)
        print(f"Saved final dataset: {output_path}")

        if validator.errors:
            print()
            print("Validation errors:")
            for error in validator.errors[:20]:
                print(f"- {error}")
    except Exception as exc:
        logger.exception("Phase 5 validation failed")
        print("Phase 5 validation failed.")
        print(f"Error: {exc}")


def _print_summary(
    total_extracted: int,
    valid_count: int,
    invalid_count: int,
    transactions: list[Transaction],
) -> None:
    """Print the Phase 5 validation summary."""
    with_balance = sum(transaction.balance is not None for transaction in transactions)
    without_balance = len(transactions) - with_balance

    print("=================================")
    print("VALIDATION SUMMARY")
    print("=================================")
    print()
    print(f"Total Transactions Extracted: {total_extracted}")
    print()
    print(f"Valid Transactions: {valid_count}")
    print()
    print(f"Invalid Transactions Removed: {invalid_count}")
    print()
    print(f"Transactions With Balance: {with_balance}")
    print()
    print(f"Transactions Without Balance: {without_balance}")
    print()


def _print_sample(records: list[dict[str, str | float | None]]) -> None:
    """Print the first 30 final records."""
    print("=================================")
    print("SAMPLE OUTPUT")
    print("=================================")
    print()

    for record in records[:30]:
        print(record)


def _to_final_dict(transaction: Transaction) -> dict[str, str | float | None]:
    """Convert a validated Transaction to final JSON output."""
    return {
        "date": transaction.date,
        "description": transaction.description,
        "amount": transaction.amount,
        "balance": transaction.balance,
        "transaction_type": transaction.transaction_type,
    }


if __name__ == "__main__":
    main()
