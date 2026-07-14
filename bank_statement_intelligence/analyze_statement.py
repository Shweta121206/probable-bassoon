"""Extract and analyze a sample bank statement PDF."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from app.analysis.statement_analyzer import (
    analyze_statement_text,
    print_statement_statistics,
)
from app.extraction.pdf_extractor import extract_pdf_text


def main() -> None:
    """Load a PDF, analyze extracted text, and print readable statistics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample.pdf")
    extracted_text = extract_pdf_text(str(pdf_path))
    stats = analyze_statement_text(extracted_text)
    print_statement_statistics(stats)


if __name__ == "__main__":
    main()
