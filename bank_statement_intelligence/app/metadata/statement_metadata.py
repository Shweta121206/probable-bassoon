"""Deterministic metadata extraction for bank statement text."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger(__name__)

BANK_NAMES = (
    "Bank of America",
    "Chase",
    "Wells Fargo",
    "Citibank",
    "HDFC Bank",
    "ICICI Bank",
    "Axis Bank",
    "State Bank of India",
)

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

NUMERIC_DATE_PATTERN = r"\d{1,2}[/-]\d{1,2}[/-]\d{4}"
MONTH_DATE_PATTERN = (
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},?\s+\d{4}"
)
DATE_PATTERN = re.compile(
    rf"\b(?:{NUMERIC_DATE_PATTERN}|{MONTH_DATE_PATTERN})\b",
    re.IGNORECASE,
)
PERIOD_PATTERN = re.compile(
    rf"(?P<start>{NUMERIC_DATE_PATTERN}|{MONTH_DATE_PATTERN})"
    rf"\s*(?:-|to|through)\s*"
    rf"(?P<end>{NUMERIC_DATE_PATTERN}|{MONTH_DATE_PATTERN})",
    re.IGNORECASE,
)
ACCOUNT_HOLDER_PATTERNS = (
    re.compile(
        r"^\s*(?:Primary\s+Account\s+Holder|Customer\s+Name|Account\s+Holder)"
        r"\s*:\s*(?P<name>[A-Za-z][A-Za-z .,'-]{1,80})\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
)
ACCOUNT_NUMBER_PATTERNS = (
    re.compile(
        r"\b(?:Account\s+Number|Account\s+#|Acct\s+#|Checking\s+Account|"
        r"Savings\s+Account)\s*:\s*(?P<number>[A-Za-z0-9*Xx -]+)",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class StatementMetadata:
    """Metadata detected from extracted bank statement text."""

    bank_name: str | None
    account_holder: str | None
    account_number: str | None
    statement_period_start: str | None
    statement_period_end: str | None
    statement_year: int | None


def extract_metadata(text: str) -> StatementMetadata:
    """Extract bank statement metadata using regex and deterministic parsing.

    Args:
        text: Extracted bank statement text.

    Returns:
        StatementMetadata with unavailable values set to None.
    """
    logger.info("Starting statement metadata extraction")
    _validate_text_input(text)

    statement_period = _extract_statement_period(text)
    statement_year: int | None = None

    if statement_period:
        start_text, end_text = statement_period
        statement_year = _parse_date(end_text).year
        logger.info("Using statement period year: %d", statement_year)
    else:
        start_text = None
        end_text = None
        try:
            statement_year = extract_statement_year(text)
            logger.info("Using fallback statement year: %d", statement_year)
        except ValueError:
            logger.info("Statement year could not be determined")

    metadata = StatementMetadata(
        bank_name=_extract_bank_name(text),
        account_holder=_extract_account_holder(text),
        account_number=_extract_account_number(text),
        statement_period_start=start_text,
        statement_period_end=end_text,
        statement_year=statement_year,
    )

    validate_metadata(metadata)
    logger.info("Completed statement metadata extraction")
    return metadata


def extract_statement_year(text: str) -> int:
    """Return the most likely statement year from statement text.

    Statement-period date ranges are preferred over other dates. If no period is
    found, the most common year among recognized dates is returned.

    Args:
        text: Extracted bank statement text.

    Returns:
        The most likely statement year.

    Raises:
        ValueError: If no year can be determined.
    """
    logger.info("Extracting statement year")
    _validate_text_input(text)

    statement_period = _extract_statement_period(text)
    if statement_period:
        year = _parse_date(statement_period[1]).year
        logger.info("Statement year found from statement period: %d", year)
        return year

    years = [_parse_date(match.group(0)).year for match in DATE_PATTERN.finditer(text)]
    if not years:
        logger.error("No date-like values found while extracting statement year")
        raise ValueError("statement year cannot be determined")

    year = max(sorted(set(years)), key=years.count)
    logger.info("Statement year found from document dates: %d", year)
    return year


def validate_metadata(metadata: StatementMetadata) -> bool:
    """Validate extracted metadata without crashing on missing values.

    Args:
        metadata: Metadata extracted from statement text.

    Returns:
        True when present period values are internally consistent, otherwise
        False.
    """
    logger.info("Validating statement metadata")

    if not metadata.statement_period_start or not metadata.statement_period_end:
        logger.info("Skipping period validation because period is incomplete")
        return True

    try:
        start_date = _parse_date(metadata.statement_period_start)
        end_date = _parse_date(metadata.statement_period_end)
    except ValueError:
        logger.warning("Statement period contains an invalid date")
        return False

    if start_date > end_date:
        logger.warning("Statement period start is after statement period end")
        return False

    if metadata.statement_year is not None and metadata.statement_year != end_date.year:
        logger.warning("Statement year does not match statement period end year")
        return False

    logger.info("Statement metadata validation passed")
    return True


def _extract_bank_name(text: str) -> str | None:
    """Detect a known bank name from the statement header or first page."""
    first_page_text = _first_page_text(text)
    for bank_name in BANK_NAMES:
        if re.search(rf"\b{re.escape(bank_name)}\b", first_page_text, re.IGNORECASE):
            logger.info("Detected bank name: %s", bank_name)
            return bank_name

    logger.info("No known bank name detected")
    return None


def _extract_account_holder(text: str) -> str | None:
    """Extract the account holder name from labeled fields."""
    first_page_text = _first_page_text(text)
    for pattern in ACCOUNT_HOLDER_PATTERNS:
        match = pattern.search(first_page_text)
        if match:
            account_holder = " ".join(match.group("name").strip().split())
            logger.info("Detected account holder")
            return account_holder

    logger.info("No account holder detected")
    return None


def _extract_account_number(text: str) -> str | None:
    """Extract an account number exactly as displayed after a supported label."""
    first_page_text = _first_page_text(text)
    for pattern in ACCOUNT_NUMBER_PATTERNS:
        match = pattern.search(first_page_text)
        if match:
            account_number = match.group("number").strip()
            logger.info("Detected account number")
            return account_number

    logger.info("No account number detected")
    return None


def _extract_statement_period(text: str) -> tuple[str, str] | None:
    """Extract and normalize a statement period date range."""
    period_search_text = _first_page_text(text)
    match = PERIOD_PATTERN.search(period_search_text)
    if not match:
        logger.info("No statement period detected")
        return None

    start_date = _parse_date(match.group("start"))
    end_date = _parse_date(match.group("end"))
    logger.info("Detected statement period: %s to %s", start_date, end_date)
    return _format_date(start_date), _format_date(end_date)


def _parse_date(value: str) -> date:
    """Parse a supported statement date into a date object."""
    cleaned_value = value.strip().replace(",", "")

    numeric_match = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", cleaned_value)
    if numeric_match:
        month, day, year = (int(part) for part in numeric_match.groups())
        return date(year, month, day)

    month_match = re.fullmatch(
        r"([A-Za-z]+)\s+(\d{1,2})\s+(\d{4})",
        cleaned_value,
        re.IGNORECASE,
    )
    if month_match:
        month_name, day_text, year_text = month_match.groups()
        month = MONTHS.get(month_name.lower())
        if month is None:
            raise ValueError(f"unsupported month name: {month_name}")

        return date(int(year_text), month, int(day_text))

    raise ValueError(f"unsupported date format: {value}")


def _format_date(value: date) -> str:
    """Format a date as MM/DD/YYYY."""
    return value.strftime("%m/%d/%Y")


def _first_page_text(text: str) -> str:
    """Return statement header and first-page text for metadata searches."""
    page_two_match = re.search(r"---\s*Page\s+2\s*---", text, re.IGNORECASE)
    if page_two_match:
        return text[: page_two_match.start()]

    return text[:4000]


def _validate_text_input(text: str) -> None:
    """Validate text input for extraction functions."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
