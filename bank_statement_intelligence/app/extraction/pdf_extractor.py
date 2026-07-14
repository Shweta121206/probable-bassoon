"""Utilities for extracting text from PDF files."""

from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from every page of a PDF file.

    Pages are read in order and joined with clear page separators. This function
    performs text extraction only; it does not run OCR, AI analysis, or
    categorization.

    Args:
        pdf_path: Path to the PDF file to extract text from.

    Returns:
        A single string containing the extracted text from all pages.

    Raises:
        FileNotFoundError: If the PDF path does not exist.
        ValueError: If the provided path is not a file.
        pdfplumber exceptions: Propagated when the PDF cannot be opened or read.
    """
    path = Path(pdf_path)
    logger.info("Starting PDF text extraction: %s", path)

    if not path.exists():
        logger.error("PDF file does not exist: %s", path)
        raise FileNotFoundError(f"PDF file does not exist: {path}")

    if not path.is_file():
        logger.error("PDF path is not a file: %s", path)
        raise ValueError(f"PDF path is not a file: {path}")

    extracted_pages: list[str] = []

    with pdfplumber.open(path) as pdf:
        logger.info("Opened PDF with %d page(s): %s", len(pdf.pages), path)

        for page_number, page in enumerate(pdf.pages, start=1):
            logger.debug("Extracting text from page %d", page_number)
            page_text = page.extract_text() or ""
            separator = f"\n\n--- Page {page_number} ---\n\n"
            extracted_pages.append(f"{separator}{page_text}")
            logger.debug(
                "Extracted %d character(s) from page %d",
                len(page_text),
                page_number,
            )

    text = "".join(extracted_pages).strip()
    logger.info("Completed PDF text extraction with %d character(s)", len(text))
    return text
