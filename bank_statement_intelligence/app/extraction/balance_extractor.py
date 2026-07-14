"""Deterministic balance history extraction from statement text."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime

from app.metadata import extract_metadata

logger = logging.getLogger(__name__)

DATE_MMDD_PATTERN = r"\d{1,2}/\d{1,2}"
DATE_FULL_PATTERN = r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
MONEY_PATTERN = r"-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})"
BALANCE_SECTION_PATTERN = re.compile(
    r"(Daily ledger balances|Running balance|Available balance).*?"
    r"(?=(?:\n--- Page|\n[A-Z][A-Za-z ]+\nDate |\nPage \d+ of \d+|\Z))",
    re.IGNORECASE | re.DOTALL,
)
BALANCE_PAIR_PATTERN = re.compile(
    rf"(?P<date>{DATE_MMDD_PATTERN}|{DATE_FULL_PATTERN})\s+"
    rf"(?P<balance>{MONEY_PATTERN})"
)
BALANCE_COLUMN_ROW_PATTERN = re.compile(
    rf"(?P<date>{DATE_FULL_PATTERN})\s+.+?\s+(?P<balance>{MONEY_PATTERN})\s*$",
    re.MULTILINE,
)
BALANCE_COLUMN_SECTION_PATTERN = re.compile(
    r"^.*\bDate\b.*\bBalance\b.*$(?P<body>.*?)(?=^\S.*\bDate\b|^--- Page|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class BalanceEntry:
    """Represents an account balance on a statement date."""

    date: str
    balance: float

    def to_dict(self) -> dict[str, str | float]:
        """Convert the balance entry to a dictionary."""
        return {"date": self.date, "balance": self.balance}


def extract_balance_history(text: str) -> list[dict[str, str | float]]:
    """Extract balance history from statement text without LLM usage.

    Args:
        text: Extracted statement text.

    Returns:
        Balance entries as dictionaries with date and balance.
    """
    logger.info("Starting deterministic balance history extraction")
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    statement_year = _detect_statement_year(text)
    entries: list[BalanceEntry] = []

    for section in _balance_sections(text):
        entries.extend(_extract_balance_pairs(section, statement_year))

    entries.extend(_extract_balance_column_entries(text, statement_year))
    entries = _deduplicate_entries(entries)
    entries = _validate_entries(entries)

    logger.info("Extracted %d balance entrie(s)", len(entries))
    return [entry.to_dict() for entry in entries]


def _detect_statement_year(text: str) -> int | None:
    """Detect statement year from metadata, returning None if unavailable."""
    try:
        return extract_metadata(text).statement_year
    except Exception as exc:
        logger.warning("Could not detect statement year for balances: %s", exc)
        return None


def _balance_sections(text: str) -> list[str]:
    """Return text sections likely to contain balance history."""
    sections = [match.group(0) for match in BALANCE_SECTION_PATTERN.finditer(text)]
    logger.info("Detected %d balance section(s)", len(sections))
    return sections


def _extract_balance_pairs(section: str, statement_year: int | None) -> list[BalanceEntry]:
    """Extract date/balance pairs from a balance section."""
    entries: list[BalanceEntry] = []

    for match in BALANCE_PAIR_PATTERN.finditer(section):
        date_text = _normalize_date(match.group("date"), statement_year)
        balance = _parse_money(match.group("balance"))
        entries.append(BalanceEntry(date=date_text, balance=balance))

    logger.debug("Extracted %d balance pair(s) from section", len(entries))
    return entries


def _extract_balance_column_entries(
    text: str,
    statement_year: int | None,
) -> list[BalanceEntry]:
    """Extract rows from explicit balance-column tables."""
    entries: list[BalanceEntry] = []
    for section_match in BALANCE_COLUMN_SECTION_PATTERN.finditer(text):
        section = section_match.group("body")
        for row_match in BALANCE_COLUMN_ROW_PATTERN.finditer(section):
            line = row_match.group(0)
            if _looks_like_transaction_amount_only(line):
                continue

            entries.append(
                BalanceEntry(
                    date=_normalize_date(row_match.group("date"), statement_year),
                    balance=_parse_money(row_match.group("balance")),
                )
            )

    logger.debug("Extracted %d balance-column entrie(s)", len(entries))
    return entries


def _looks_like_transaction_amount_only(line: str) -> bool:
    """Return whether a line appears to be a transaction without a balance column."""
    lower_line = line.lower()
    excluded_terms = ("total", "deposit", "withdrawal", "fee", "payment", "transfer")
    return any(term in lower_line for term in excluded_terms)


def _normalize_date(value: str, statement_year: int | None) -> str:
    """Normalize supported date text to MM/DD/YYYY."""
    cleaned_value = value.strip().replace("-", "/")
    parts = cleaned_value.split("/")

    if len(parts) == 2:
        if statement_year is None:
            raise ValueError(f"cannot normalize date without year: {value}")
        month, day = (int(part) for part in parts)
        return date(statement_year, month, day).strftime("%m/%d/%Y")

    if len(parts) == 3:
        month, day, year = (int(part) for part in parts)
        if year < 100:
            year += 2000
        return date(year, month, day).strftime("%m/%d/%Y")

    raise ValueError(f"unsupported date format: {value}")


def _parse_money(value: str) -> float:
    """Parse a money-like value into a float."""
    return float(value.replace("$", "").replace(",", ""))


def _deduplicate_entries(entries: list[BalanceEntry]) -> list[BalanceEntry]:
    """Remove duplicate date/balance pairs while preserving order."""
    deduplicated_entries: list[BalanceEntry] = []
    seen: set[tuple[str, float]] = set()

    for entry in entries:
        key = (entry.date, entry.balance)
        if key in seen:
            continue
        deduplicated_entries.append(entry)
        seen.add(key)

    return deduplicated_entries


def _validate_entries(entries: list[BalanceEntry]) -> list[BalanceEntry]:
    """Validate balance entries and sort by date."""
    validated_entries: list[BalanceEntry] = []

    for entry in entries:
        datetime.strptime(entry.date, "%m/%d/%Y")
        if not isinstance(entry.balance, float):
            raise TypeError("balance must be a float")
        validated_entries.append(entry)

    return sorted(
        validated_entries,
        key=lambda entry: datetime.strptime(entry.date, "%m/%d/%Y"),
    )
