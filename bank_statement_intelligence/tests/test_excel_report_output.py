"""Regression tests for bank statement Excel export."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook

from app.metadata import StatementMetadata
from app.reports import create_bank_statement_workbook


class ExcelReportOutputTests(unittest.TestCase):
    """Ensure the workbook includes statement metadata on every transaction row."""

    def test_transaction_rows_include_statement_metadata(self) -> None:
        metadata = StatementMetadata(
            bank_name="Bank of America",
            account_holder="Jane Doe",
            account_number="123456789",
            statement_period_start="12/01/2025",
            statement_period_end="12/31/2025",
            statement_year=2025,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "report.xlsx"
            create_bank_statement_workbook(
                transactions=[
                    {
                        "date": "12/01/2025",
                        "description": "Deposit",
                        "amount": 100.0,
                        "balance": 1100.0,
                        "transaction_type": "Credit",
                    }
                ],
                metadata=metadata,
                balance_history=[{"date": "12/01/2025", "balance": 1100.0}],
                output_path=str(output_path),
            )

            workbook = load_workbook(output_path)
            transactions_sheet = workbook["Transactions"]

            self.assertEqual(transactions_sheet["A2"].value, "Bank of America")
            self.assertEqual(transactions_sheet["B2"].value, "Jane Doe")
            self.assertEqual(transactions_sheet["C2"].value, "123456789")


if __name__ == "__main__":
    unittest.main()
