"""Phase 3 smoke test for real PDF extraction and metadata extraction."""

from __future__ import annotations

import logging
from pathlib import Path

from app.extraction.pdf_extractor import extract_pdf_text
from app.metadata import StatementMetadata, extract_metadata

logger = logging.getLogger(__name__)

PDF_FILENAME = "December 2025 Bank Statement.pdf"


def main() -> None:
    """Extract text from the sample PDF and print detected metadata."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    pdf_path = Path(__file__).resolve().parent / PDF_FILENAME

    try:
        logger.info("Starting Phase 3 test with PDF: %s", pdf_path)
        extracted_text = extract_pdf_text(str(pdf_path))

        print(f"Total text length: {len(extracted_text)}")
        print("First 3000 characters:")
        print(extracted_text[:3000])
        print()

        metadata = extract_metadata(extracted_text)
        print_metadata(metadata)

        logger.info("Phase 3 test completed successfully")
    except Exception as exc:
        logger.exception("Phase 3 test failed")
        print("Phase 3 test failed.")
        print(f"Error: {exc}")


def print_metadata(metadata: StatementMetadata) -> None:
    """Print statement metadata in a readable format."""
    print("====================================")
    print("BANK STATEMENT METADATA")
    print("====================================")
    print()
    print("Bank Name:")
    print(metadata.bank_name or "Not found")
    print()
    print("Account Holder:")
    print(metadata.account_holder or "Not found")
    print()
    print("Account Number:")
    print(metadata.account_number or "Not found")
    print()
    print("Statement Start:")
    print(metadata.statement_period_start or "Not found")
    print()
    print("Statement End:")
    print(metadata.statement_period_end or "Not found")
    print()
    print("Statement Year:")
    print(metadata.statement_year if metadata.statement_year is not None else "Not found")


if __name__ == "__main__":
    main()
