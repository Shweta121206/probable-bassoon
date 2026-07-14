"""Tests for transaction validation and normalization."""

from __future__ import annotations

import unittest

from app.validation import TransactionValidator


class TransactionValidatorTests(unittest.TestCase):
    """Unit tests for TransactionValidator."""

    def setUp(self) -> None:
        self.validator = TransactionValidator()

    def test_normalize_mm_dd_date(self) -> None:
        self.assertEqual(self.validator.normalize_date("12/5", 2025), "12/05/25")
        self.assertEqual(self.validator.normalize_date("12/05", 2025), "12/05/25")

    def test_normalize_mm_dd_yy_date(self) -> None:
        self.assertEqual(self.validator.normalize_date("12/05/25", 2025), "12/05/25")

    def test_normalize_mm_dd_yyyy_date(self) -> None:
        self.assertEqual(self.validator.normalize_date("12/05/2025", 2025), "12/05/25")

    def test_duplicate_removal_keeps_first(self) -> None:
        transactions = [
            {"date": "12/05", "description": "ATM WITHDRAWAL", "amount": "-20.00"},
            {"date": "12/5", "description": "ATM WITHDRAWAL", "amount": -20.0},
        ]

        validated = self.validator.validate_transactions(transactions, 2025)

        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0].amount, -20.0)

    def test_missing_amount_removed(self) -> None:
        transactions = [
            {"date": "12/05", "description": "ATM WITHDRAWAL"},
            {"date": "12/06", "description": "PAYROLL DEPOSIT", "amount": 1200},
        ]

        validated = self.validator.validate_transactions(transactions, 2025)

        self.assertEqual(len(validated), 1)
        self.assertEqual(len(self.validator.errors), 1)

    def test_negative_parentheses_amount(self) -> None:
        transactions = [
            {"date": "12/05", "description": "SERVICE CHARGE", "amount": "(120.50)"}
        ]

        validated = self.validator.validate_transactions(transactions, 2025)

        self.assertEqual(validated[0].amount, -120.50)

    def test_balance_mapping(self) -> None:
        transactions = self.validator.validate_transactions(
            [{"date": "12/05", "description": "SERVICE CHARGE", "amount": -16.0}],
            2025,
        )
        balances = [{"date": "12/05/2025", "balance": 7410.91}]

        merged = self.validator.merge_balance_history(transactions, balances)

        self.assertEqual(merged[0].balance, 7410.91)

    def test_multiple_transactions_same_day_share_balance(self) -> None:
        transactions = self.validator.validate_transactions(
            [
                {"date": "12/05", "description": "SERVICE CHARGE", "amount": -16.0},
                {"date": "12/05", "description": "ATM WITHDRAWAL", "amount": -20.0},
            ],
            2025,
        )
        balances = [{"date": "12/05/2025", "balance": 7410.91}]

        merged = self.validator.merge_balance_history(transactions, balances)

        self.assertEqual(merged[0].balance, 7410.91)
        self.assertEqual(merged[1].balance, 7410.91)


if __name__ == "__main__":
    unittest.main()
