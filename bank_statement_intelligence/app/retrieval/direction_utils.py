from __future__ import annotations

from typing import Any


class DirectionUtils:
    """Infer transaction direction from amounts and transaction type."""

    @staticmethod
    def detect_direction(debit: Any, credit: Any, transaction_type: Any = None) -> str:
        debit_value = DirectionUtils._to_optional_float(debit)
        credit_value = DirectionUtils._to_optional_float(credit)
        transaction_type_text = str(transaction_type or "").strip().lower()

        if debit_value is not None and credit_value is None:
            return "Withdrawal"
        if credit_value is not None and debit_value is None:
            return "Deposit"
        if debit_value is not None and credit_value is not None:
            if abs(debit_value - credit_value) < 0.01:
                return "Unknown"
            if transaction_type_text in {"transfer", "transferout", "transferin"}:
                return "Transfer"
            if transaction_type_text in {"deposit", "credit"}:
                return "Deposit"
            if transaction_type_text in {"withdrawal", "debit", "expense"}:
                return "Withdrawal"
            return "Unknown"

        if transaction_type_text in {"deposit", "credit", "income"}:
            return "Deposit"
        if transaction_type_text in {"withdrawal", "debit", "expense"}:
            return "Withdrawal"
        if transaction_type_text in {"transfer", "transferout", "transferin"}:
            return "Transfer"

        return "Unknown"

    @staticmethod
    def _to_optional_float(value: Any) -> float | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
