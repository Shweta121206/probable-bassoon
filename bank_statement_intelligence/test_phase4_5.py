"""Phase 4.5 test for deterministic balance extraction."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.extraction.balance_extractor import extract_balance_history
from app.extraction.pdf_extractor import extract_pdf_text

logger = logging.getLogger(__name__)

PDF_FILENAME = "December 2025 Bank Statement.pdf"


def main() -> None:
    """Extract balance history from the real statement PDF and save JSON output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    project_root = Path(__file__).resolve().parent
    pdf_path = project_root / PDF_FILENAME
    output_path = project_root / "output" / "balances.json"

    try:
        statement_text = extract_pdf_text(str(pdf_path))
        balances = extract_balance_history(statement_text)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(balances, indent=2), encoding="utf-8")

        print(f"Number of balances extracted: {len(balances)}")
        print("Balance entries:")
        for balance in balances:
            print(balance)

        print(f"Balances saved to: {output_path}")
    except Exception as exc:
        logger.exception("Phase 4.5 balance extraction failed")
        print("Phase 4.5 balance extraction failed.")
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
