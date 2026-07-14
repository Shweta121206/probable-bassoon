"""Phase 4 test for LLM-based transaction extraction."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.extraction.pdf_extractor import extract_pdf_text
from app.llm import TransactionExtractor
from app.models import Transaction

logger = logging.getLogger(__name__)

PDF_FILENAME = "December 2025 Bank Statement.pdf"


def main() -> None:
    """Extract transactions from the real statement PDF and save JSON output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    project_root = Path(__file__).resolve().parent
    pdf_path = project_root / PDF_FILENAME
    output_path = project_root / "output" / "transactions.json"

    try:
        statement_text = extract_pdf_text(str(pdf_path))
        extractor = TransactionExtractor()
        transactions = extractor.extract_transactions(statement_text)
        transaction_json = extractor.last_raw_json or json.dumps(
            [_to_phase4_dict(transaction) for transaction in transactions],
            indent=2,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(transaction_json, encoding="utf-8")

        print(f"Number of transactions found: {len(transactions)}")
        print("First 100 transactions:")
        for transaction in transactions[:100]:
            print(transaction)

        print()
        print("Extraction statistics:")
        print(f"Total statement text length: {len(statement_text)}")
        print(f"Transactions with balance: {sum(t.balance is not None for t in transactions)}")
        print(f"Transactions saved to: {output_path}")
    except Exception as exc:
        logger.exception("Phase 4 transaction extraction failed")
        print("Phase 4 transaction extraction failed.")
        print(f"Error: {exc}")


def _to_phase4_dict(transaction: Transaction) -> dict[str, str | float | None]:
    """Convert a Transaction to the Phase 4 JSON schema."""
    return {
        "date": transaction.date,
        "description": transaction.description,
        "amount": transaction.amount,
        "balance": transaction.balance,
        "transaction_type": transaction.transaction_type,
    }


if __name__ == "__main__":
    main()
