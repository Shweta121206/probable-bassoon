"""Smoke test script for PDF text extraction."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from app.extraction.pdf_extractor import extract_pdf_text


def main() -> None:
    """Load a sample PDF and print basic extraction output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample.pdf")
    extracted_text = extract_pdf_text(str(pdf_path))

    print(f"Total character count: {len(extracted_text)}")
    print("First 2000 characters:")
    print(extracted_text[:2000])


if __name__ == "__main__":
    main()
