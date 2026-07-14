"""Tests for the chart-of-account retrieval engine."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from app.categorization.retriever import ChartOfAccountsRetriever


class ChartOfAccountsRetrieverTests(unittest.TestCase):
    """Verify account retrieval and caching behavior."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.account_path = Path(self.temp_dir.name) / "chart_of_accounts.xlsx"
        self._write_accounts_workbook()

    def _write_accounts_workbook(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "ChartOfAccounts - TEMPLATE"
        sheet.append(["*Code", "*Name", "*Type"])
        sheet.append([1000, "Cash", "Asset"])
        sheet.append([2000, "Accounts Receivable", "Asset"])
        sheet.append([4000, "Utilities Expense", "Expense"])
        sheet.append([5000, "Sales Revenue", "Revenue"])
        workbook.save(self.account_path)

    def test_returns_top_candidates_for_single_description(self) -> None:
        retriever = ChartOfAccountsRetriever(self.account_path)
        results = retriever.retrieve("utility bill payment", top_n=3)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["account_name"], "Utilities Expense")
        self.assertGreater(results[0]["similarity"], 0.0)

    def test_batch_processing_returns_results_per_description(self) -> None:
        retriever = ChartOfAccountsRetriever(self.account_path)
        results = retriever.retrieve_batch(["utility bill payment", "cash deposit"], top_n=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0]["account_name"], "Utilities Expense")
        self.assertEqual(results[1][0]["account_name"], "Cash")

    def test_cache_is_reused_for_repeated_calls(self) -> None:
        retriever = ChartOfAccountsRetriever(self.account_path)
        first_results = retriever.retrieve("sales revenue", top_n=1)
        second_results = retriever.retrieve("sales revenue", top_n=1)

        self.assertEqual(first_results[0]["account_code"], second_results[0]["account_code"])
        self.assertEqual(first_results[0]["account_name"], second_results[0]["account_name"])


if __name__ == "__main__":
    unittest.main()
