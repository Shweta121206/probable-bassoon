"""Regex-based statement text analysis."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PAGE_SEPARATOR_PATTERN = re.compile(r"---\s*Page\s+\d+\s*---", re.IGNORECASE)
DATE_PATTERN = re.compile(
    r"\b(?:"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|"
    r"\d{4}[/-]\d{1,2}[/-]\d{1,2}"
    r"|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}"
    r")\b",
    re.IGNORECASE,
)
AMOUNT_PATTERN = re.compile(
    r"(?<!\w)(?:[$€£₹]\s*)?-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?(?!\w)"
)
SECTION_HEADER_PATTERN = re.compile(
    r"^\s*(?:"
    r"[A-Z][A-Z0-9 &/().,-]{3,}"
    r"|"
    r"(?:Account|Statement|Transaction|Transactions|Deposits|Withdrawals|"
    r"Payments|Purchases|Fees|Summary|Balance|Details|Activity)\b[^\n:]{0,60}:?"
    r")\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class StatementStatistics:
    """Basic statistics found in extracted statement text."""

    page_count: int
    total_characters: int
    total_lines: int
    date_like_patterns: int
    amount_like_patterns: int
    section_headers: list[str]


def analyze_statement_text(statement_text: str) -> StatementStatistics:
    """Analyze extracted statement text using regular expressions.

    Args:
        statement_text: Text extracted from a bank statement.

    Returns:
        Basic statement statistics, including counts and detected headers.
    """
    logger.info("Starting statement text analysis")

    page_matches = PAGE_SEPARATOR_PATTERN.findall(statement_text)
    date_matches = DATE_PATTERN.findall(statement_text)
    amount_matches = AMOUNT_PATTERN.findall(statement_text)
    section_headers = _detect_section_headers(statement_text)

    stats = StatementStatistics(
        page_count=len(page_matches) if page_matches else 1 if statement_text else 0,
        total_characters=len(statement_text),
        total_lines=len(statement_text.splitlines()),
        date_like_patterns=len(date_matches),
        amount_like_patterns=len(amount_matches),
        section_headers=section_headers,
    )

    logger.info(
        "Completed statement analysis: %d page(s), %d character(s), %d line(s)",
        stats.page_count,
        stats.total_characters,
        stats.total_lines,
    )
    return stats


def print_statement_statistics(stats: StatementStatistics) -> None:
    """Print statement statistics in a readable format.

    Args:
        stats: Statistics returned by analyze_statement_text.
    """
    print("Statement Analysis")
    print("==================")
    print(f"Page count: {stats.page_count}")
    print(f"Total characters: {stats.total_characters}")
    print(f"Total lines: {stats.total_lines}")
    print(f"Date-like patterns: {stats.date_like_patterns}")
    print(f"Amount-like patterns: {stats.amount_like_patterns}")
    print("Detected section headers:")

    if not stats.section_headers:
        print("  None detected")
        return

    for header in stats.section_headers:
        print(f"  - {header}")


def _detect_section_headers(statement_text: str) -> list[str]:
    """Find unique section-like headers while preserving their first-seen order."""
    headers: list[str] = []
    seen: set[str] = set()

    for match in SECTION_HEADER_PATTERN.finditer(statement_text):
        header = " ".join(match.group(0).strip().split())
        if not header:
            continue

        normalized = header.lower()
        if normalized in seen:
            continue

        headers.append(header)
        seen.add(normalized)

    logger.debug("Detected %d section header(s)", len(headers))
    return headers
