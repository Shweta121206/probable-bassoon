"""Phase 6 test for creating the final Excel report from validated transactions."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.categorization import ChartOfAccountsRetriever
from app.extraction.balance_extractor import extract_balance_history
from app.extraction.pdf_extractor import extract_pdf_text
from app.llm import TransactionExtractor
from app.metadata import extract_metadata
from app.reports import create_bank_statement_workbook
from app.validation import TransactionValidator

logger = logging.getLogger(__name__)

PDF_FILENAME = "Janaury 2026 Bank Statement.pdf"
INPUT_FILENAME = "validated_transactions.json"
OUTPUT_FILENAME = "BankStatementOutput.xlsx"


def main() -> None:
    """Generate the Excel workbook from the statement PDF and validated transactions."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    project_root = Path(__file__).resolve().parent
    pdf_path = project_root / PDF_FILENAME
    input_path = project_root / "output" / INPUT_FILENAME
    output_path = project_root / OUTPUT_FILENAME

    if not input_path.exists():
        fallback_paths = [
            project_root / "output" / "final_transactions.json",
            project_root / "output" / "transactions.json",
        ]
        for fallback_path in fallback_paths:
            if fallback_path.exists():
                input_path = fallback_path
                break

    if not input_path.exists():
        raise FileNotFoundError(
            f"No transaction input found. Expected {input_path} or a fallback file."
        )

    statement_text = extract_pdf_text(str(pdf_path))
    metadata = extract_metadata(statement_text)
    balance_history = extract_balance_history(statement_text)

    if metadata.statement_year is None:
        raise ValueError("statement year is required for transaction validation")

    extracted_transactions = TransactionExtractor().extract_transactions(statement_text)
    validator = TransactionValidator()
    validated_transactions = validator.validate_transactions(
        extracted_transactions,
        metadata.statement_year,
    )
    final_transactions = validator.merge_balance_history(
        validated_transactions,
        balance_history,
    )

    account_path = project_root / "ChartOfAccounts - TEMPLATE.xlsx"
    retriever = ChartOfAccountsRetriever(account_path)
    transaction_records = [
        _to_final_dict(transaction, retriever)
        for transaction in final_transactions
    ]

    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps(transaction_records, indent=2), encoding="utf-8")

    saved_path = create_bank_statement_workbook(
        transactions=transaction_records,
        metadata=metadata,
        balance_history=balance_history,
        output_path=str(output_path),
    )

    print(f"Generated workbook: {saved_path}")
    print(f"Transactions loaded: {len(transaction_records)}")
    print(f"Balance history entries: {len(balance_history)}")


def _to_final_dict(transaction: object, retriever: ChartOfAccountsRetriever) -> dict[str, object]:
    """Convert a validated transaction to the workbook-friendly record format."""
    description = getattr(transaction, "description", "")
    matching_accounts = retriever.retrieve(str(description), top_n=1)
    suggested_account = matching_accounts[0]["account_name"] if matching_accounts else ""
    suggested_account_type = matching_accounts[0]["account_type"] if matching_accounts else ""

    return {
        "date": getattr(transaction, "date", ""),
        "description": description,
        "amount": getattr(transaction, "amount", 0),
        "balance": getattr(transaction, "balance", None),
        "transaction_type": getattr(transaction, "transaction_type", ""),
        "suggested_account": suggested_account,
        "suggested_account_type": suggested_account_type,
    }


if __name__ == "__main__":
    main()
