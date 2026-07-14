"""Validation and normalization for extracted transaction data."""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

from app.models import Transaction

logger = logging.getLogger(__name__)


class TransactionValidator:
    """Validate, normalize, deduplicate, and enrich transaction records."""

    def __init__(self) -> None:
        """Initialize the validator with an empty error collection."""
        self.errors: list[str] = []

    def normalize_date(self, date_str: str, statement_year: int) -> str:
        """Normalize transaction dates to MM/DD/YY using statement_year.

        Supported input formats are MM/DD, M/D, MM/DD/YY, and MM/DD/YYYY.
        Years are resolved only from the provided statement_year.

        Args:
            date_str: Date string extracted from a transaction row.
            statement_year: Statement year from metadata extraction.

        Returns:
            The normalized date as MM/DD/YY.

        Raises:
            TypeError: If date_str is not a string.
            ValueError: If the date is missing or malformed.
        """
        if not isinstance(date_str, str):
            raise TypeError("date_str must be a string")

        cleaned_date = date_str.strip().replace("-", "/")
        if not cleaned_date:
            raise ValueError("date is missing")

        parts = cleaned_date.split("/")
        if len(parts) not in (2, 3):
            raise ValueError(f"unsupported date format: {date_str}")

        month = int(parts[0])
        day = int(parts[1])
        resolved_year = statement_year

        normalized = date(resolved_year, month, day)
        return normalized.strftime("%m/%d/%y")

    def validate_transactions(
        self,
        transactions: list[Any],
        statement_year: int,
    ) -> list[Transaction]:
        """Validate and normalize extracted transactions.

        Invalid rows are skipped and recorded in self.errors. Duplicates are
        removed using date, description, and amount.

        Args:
            transactions: Raw transaction dicts or Transaction objects.
            statement_year: Statement year from metadata extraction.

        Returns:
            Valid, normalized, deduplicated Transaction objects.
        """
        logger.info("Starting transaction validation for %d row(s)", len(transactions))
        self.errors.clear()
        valid_transactions: list[Transaction] = []
        seen_keys: set[tuple[str, str, float]] = set()

        for index, transaction in enumerate(transactions, start=1):
            try:
                data = self._to_dict(transaction)
                normalized_date = self.normalize_date(data.get("date", ""), statement_year)
                description = self._normalize_description(data.get("description"))
                amount = self._normalize_amount(data.get("amount"))
                balance = self._normalize_optional_amount(data.get("balance"))
                transaction_type = str(data.get("transaction_type") or "").strip()
                category = str(data.get("category") or "Uncategorized").strip()

                dedupe_key = (normalized_date, description.casefold(), amount)
                if dedupe_key in seen_keys:
                    logger.info("Skipping duplicate transaction at row %d", index)
                    continue

                valid_transactions.append(
                    Transaction(
                        date=normalized_date,
                        description=description,
                        amount=amount,
                        category=category or "Uncategorized",
                        balance=balance,
                        transaction_type=transaction_type,
                    )
                )
                seen_keys.add(dedupe_key)
            except Exception as exc:
                error = f"row {index}: {exc}"
                self.errors.append(error)
                logger.warning("Invalid transaction removed: %s", error)

        logger.info(
            "Transaction validation complete: %d valid, %d invalid",
            len(valid_transactions),
            len(self.errors),
        )
        return valid_transactions

    def merge_balance_history(
        self,
        transactions: list[Any],
        balance_history: list[Any],
    ) -> list[Transaction]:
        """Populate missing transaction balances from balance history by date.

        Existing transaction balances are preserved. When multiple transactions
        share a date, the same balance-history value is applied to each missing
        balance.

        Args:
            transactions: Validated transactions.
            balance_history: Balance dictionaries containing date and balance.

        Returns:
            Transactions with missing balances filled where possible.
        """
        logger.info("Starting balance mapping for %d transaction(s)", len(transactions))
        balance_by_date = self._build_balance_map(balance_history)
        merged_transactions: list[Transaction] = []

        for index, transaction in enumerate(transactions, start=1):
            try:
                data = self._to_dict(transaction)
                transaction_obj = Transaction.from_dict(data)

                if transaction_obj.balance is None:
                    balance = balance_by_date.get(transaction_obj.date)
                    if balance is not None:
                        transaction_obj.balance = balance
                        logger.debug(
                            "Mapped balance %.2f to transaction row %d",
                            balance,
                            index,
                        )

                merged_transactions.append(transaction_obj)
            except Exception as exc:
                error = f"balance merge row {index}: {exc}"
                self.errors.append(error)
                logger.warning("Could not merge balance for transaction: %s", error)

        logger.info("Balance mapping complete")
        return merged_transactions

    def _build_balance_map(self, balance_history: list[Any]) -> dict[str, float]:
        """Build a date-to-balance map from raw balance history records."""
        balance_by_date: dict[str, float] = {}

        for index, balance_entry in enumerate(balance_history, start=1):
            try:
                if not isinstance(balance_entry, dict):
                    raise TypeError("balance entry must be a dictionary")

                normalized_date = self._normalize_balance_date(balance_entry.get("date"))
                balance = self._normalize_amount(balance_entry.get("balance"))
                balance_by_date.setdefault(normalized_date, balance)
            except Exception as exc:
                error = f"balance row {index}: {exc}"
                self.errors.append(error)
                logger.warning("Invalid balance entry ignored: %s", error)

        return balance_by_date

    def _normalize_balance_date(self, value: Any) -> str:
        """Normalize balance-history dates to MM/DD/YY."""
        if not isinstance(value, str):
            raise TypeError("balance date must be a string")

        cleaned_date = value.strip().replace("-", "/")
        parts = cleaned_date.split("/")
        if len(parts) == 2:
            raise ValueError("balance date requires a year before validation")
        if len(parts) != 3:
            raise ValueError(f"unsupported balance date format: {value}")

        month, day, year = (int(part) for part in parts)
        if year >= 100:
            year %= 100

        return f"{month:02d}/{day:02d}/{year:02d}"

    @staticmethod
    def _to_dict(transaction: Any) -> dict[str, Any]:
        """Convert a Transaction or mapping-like value to a plain dictionary."""
        if isinstance(transaction, Transaction):
            return transaction.to_dict()

        if isinstance(transaction, dict):
            return transaction

        raise TypeError("transaction must be a dictionary or Transaction")

    @staticmethod
    def _normalize_description(value: Any) -> str:
        """Validate and normalize a transaction description."""
        if not isinstance(value, str):
            raise TypeError("description missing or not a string")

        description = " ".join(value.strip().split())
        if not description:
            raise ValueError("description missing")

        return description

    @staticmethod
    def _normalize_amount(value: Any) -> float:
        """Validate and normalize amount-like values to float."""
        if value is None:
            raise ValueError("amount missing")

        if isinstance(value, bool):
            raise TypeError("amount not numeric")

        if isinstance(value, int | float):
            return float(value)

        if not isinstance(value, str):
            raise TypeError("amount not numeric")

        cleaned_amount = value.strip()
        if not cleaned_amount:
            raise ValueError("amount missing")

        is_parenthesized_negative = bool(
            re.fullmatch(r"\(\s*\$?[\d,]+(?:\.\d+)?\s*\)", cleaned_amount)
        )
        cleaned_amount = (
            cleaned_amount.replace("$", "")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .strip()
        )

        amount = float(cleaned_amount)
        return -abs(amount) if is_parenthesized_negative else amount

    @classmethod
    def _normalize_optional_amount(cls, value: Any) -> float | None:
        """Normalize optional balance values to float or None."""
        if value is None or value == "":
            return None

        return cls._normalize_amount(value)
