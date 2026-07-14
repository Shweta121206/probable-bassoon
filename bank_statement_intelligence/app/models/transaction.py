"""Transaction domain model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Transaction:
    """Represents a single bank statement transaction."""

    date: str
    description: str
    amount: float
    category: str = "Uncategorized"
    balance: float | None = None
    transaction_type: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize transaction fields after initialization."""
        self.date = self.validate_date(self.date)
        self.description = self.validate_description(self.description)
        self.amount = self.validate_amount(self.amount)
        self.balance = self.validate_balance(self.balance)
        self.transaction_type = self.validate_transaction_type(self.transaction_type)
        self.category = self.validate_category(self.category)

    @staticmethod
    def validate_date(date: str) -> str:
        """Validate a transaction date value.

        Args:
            date: Date text extracted from a statement.

        Returns:
            The stripped date string.

        Raises:
            TypeError: If date is not a string.
            ValueError: If date is empty.
        """
        if not isinstance(date, str):
            raise TypeError("date must be a string")

        cleaned_date = date.strip()
        if not cleaned_date:
            raise ValueError("date cannot be empty")

        return cleaned_date

    @staticmethod
    def validate_description(description: str) -> str:
        """Validate a transaction description value.

        Args:
            description: Description text extracted from a statement.

        Returns:
            The stripped description string.

        Raises:
            TypeError: If description is not a string.
            ValueError: If description is empty.
        """
        if not isinstance(description, str):
            raise TypeError("description must be a string")

        cleaned_description = description.strip()
        if not cleaned_description:
            raise ValueError("description cannot be empty")

        return cleaned_description

    @staticmethod
    def validate_amount(amount: float) -> float:
        """Validate a transaction amount value.

        Args:
            amount: Numeric transaction amount.

        Returns:
            The amount as a float.

        Raises:
            TypeError: If amount is not numeric.
        """
        if isinstance(amount, bool) or not isinstance(amount, int | float):
            raise TypeError("amount must be a number")

        return float(amount)

    @staticmethod
    def validate_balance(balance: float | None) -> float | None:
        """Validate an optional transaction balance value.

        Args:
            balance: Optional account balance after the transaction.

        Returns:
            The balance as a float, or None.

        Raises:
            TypeError: If balance is not numeric or None.
        """
        if balance is None:
            return None

        if isinstance(balance, bool) or not isinstance(balance, int | float):
            raise TypeError("balance must be a number or None")

        return float(balance)

    @staticmethod
    def validate_transaction_type(transaction_type: str) -> str:
        """Validate a transaction type value.

        Args:
            transaction_type: Transaction type label.

        Returns:
            The stripped transaction type string.

        Raises:
            TypeError: If transaction_type is not a string.
        """
        if not isinstance(transaction_type, str):
            raise TypeError("transaction_type must be a string")

        return transaction_type.strip()

    @staticmethod
    def validate_category(category: str) -> str:
        """Validate a transaction category value.

        Args:
            category: Category label for the transaction.

        Returns:
            The stripped category string.

        Raises:
            TypeError: If category is not a string.
            ValueError: If category is empty.
        """
        if not isinstance(category, str):
            raise TypeError("category must be a string")

        cleaned_category = category.strip()
        if not cleaned_category:
            raise ValueError("category cannot be empty")

        return cleaned_category

    def is_valid(self) -> bool:
        """Return whether the transaction passes field validation."""
        try:
            self.validate_date(self.date)
            self.validate_description(self.description)
            self.validate_amount(self.amount)
            self.validate_balance(self.balance)
            self.validate_transaction_type(self.transaction_type)
            self.validate_category(self.category)
        except (TypeError, ValueError):
            return False

        return True

    def to_dict(self) -> dict[str, str | float | None]:
        """Convert the transaction to a dictionary."""
        return {
            "date": self.date,
            "description": self.description,
            "amount": self.amount,
            "balance": self.balance,
            "transaction_type": self.transaction_type,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transaction:
        """Create a transaction from a dictionary.

        Args:
            data: Dictionary containing transaction field values.

        Returns:
            A validated Transaction instance.

        Raises:
            TypeError: If data is not a dictionary.
            KeyError: If a required field is missing.
        """
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")

        return cls(
            date=data["date"],
            description=data["description"],
            amount=data["amount"],
            balance=data.get("balance"),
            transaction_type=data.get("transaction_type", ""),
            category=data.get("category", "Uncategorized"),
        )

    def __str__(self) -> str:
        """Return a readable transaction summary."""
        balance_text = "None" if self.balance is None else f"{self.balance:.2f}"
        return (
            f"{self.date} | {self.description} | "
            f"{self.amount:.2f} | balance={balance_text} | "
            f"{self.transaction_type or 'Unspecified'} | {self.category}"
        )
