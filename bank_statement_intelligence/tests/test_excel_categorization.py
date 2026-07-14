"""Regression test for writing transaction categorization into the Excel sheet."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook

from app.metadata import StatementMetadata
from app.reports import create_bank_statement_workbook


class ExcelCategorizationTests(unittest.TestCase):
    """Ensure the generated workbook includes suggested account columns and values."""

    def test_excel_sheet_contains_suggested_accounts(self) -> None:
        metadata = StatementMetadata(
            bank_name="Bank of America",
            account_holder="Jane Doe",
            account_number="123456789",
            statement_period_start="01/01/2026",
            statement_period_end="01/31/2026",
            statement_year=2026,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "categorized.xlsx"
            create_bank_statement_workbook(
                transactions=[
                    {
                        "date": "01/15/2026",
                        "description": "utility bill payment",
                        "amount": 120.0,
                        "balance": 1000.0,
                        "transaction_type": "Debit",
                        "suggested_account": "Utilities",
                        "suggested_account_type": "Expense",
                    }
                ],
                metadata=metadata,
                balance_history=[{"date": "01/15/2026", "balance": 1000.0}],
                output_path=str(output_path),
            )

            workbook = load_workbook(output_path)
            transactions_sheet = workbook["Transactions"]

            self.assertEqual(transactions_sheet["I1"].value, "Suggested Account")
            self.assertEqual(transactions_sheet["J1"].value, "Suggested Account Type")
            self.assertEqual(transactions_sheet["I2"].value, "Utilities")
            self.assertEqual(transactions_sheet["J2"].value, "Expense")


if __name__ == "__main__":
    unittest.main()
