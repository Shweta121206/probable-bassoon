"""Tests for deterministic bank statement metadata extraction."""

from __future__ import annotations

import unittest

from app.metadata import (
    StatementMetadata,
    extract_metadata,
    extract_statement_year,
    validate_metadata,
)


class StatementMetadataTests(unittest.TestCase):
    """Unit tests for bank statement metadata extraction."""

    def test_extract_statement_year_from_numeric_date_range(self) -> None:
        text = "Statement Period: 09/01/2025 - 09/30/2025"

        self.assertEqual(extract_statement_year(text), 2025)

    def test_extract_statement_year_from_month_date_range(self) -> None:
        text = "September 1 2025 to September 30 2025"

        self.assertEqual(extract_statement_year(text), 2025)

    def test_extract_statement_year_prefers_period_with_multiple_dates(self) -> None:
        text = """
        Previous notice date: 12/31/2024
        Statement Period:
        09/01/2025 - 09/30/2025
        Payment due date: 10/15/2025
        """

        self.assertEqual(extract_statement_year(text), 2025)

    def test_extract_statement_year_raises_for_missing_dates(self) -> None:
        with self.assertRaises(ValueError):
            extract_statement_year("No statement dates are present here.")

    def test_extract_metadata_for_bank_of_america(self) -> None:
        text = """
        Bank of America
        Primary Account Holder: John Smith
        Account Number: 123456789
        Statement Period: 09/01/2025 - 09/30/2025
        """

        metadata = extract_metadata(text)

        self.assertEqual(metadata.bank_name, "Bank of America")
        self.assertEqual(metadata.account_holder, "John Smith")
        self.assertEqual(metadata.account_number, "123456789")
        self.assertEqual(metadata.statement_period_start, "09/01/2025")
        self.assertEqual(metadata.statement_period_end, "09/30/2025")
        self.assertEqual(metadata.statement_year, 2025)

    def test_extract_metadata_for_chase(self) -> None:
        text = """
        Chase
        Customer Name: Jane Doe
        Account #: XXXX4321
        09-01-2025 to 09-30-2025
        """

        metadata = extract_metadata(text)

        self.assertEqual(metadata.bank_name, "Chase")
        self.assertEqual(metadata.account_holder, "Jane Doe")
        self.assertEqual(metadata.account_number, "XXXX4321")
        self.assertEqual(metadata.statement_period_start, "09/01/2025")
        self.assertEqual(metadata.statement_period_end, "09/30/2025")
        self.assertEqual(metadata.statement_year, 2025)

    def test_extract_metadata_for_hdfc(self) -> None:
        text = """
        HDFC Bank
        Account Holder: Priya Sharma
        Savings Account: ****9876
        September 1 2025 through September 30 2025
        """

        metadata = extract_metadata(text)

        self.assertEqual(metadata.bank_name, "HDFC Bank")
        self.assertEqual(metadata.account_holder, "Priya Sharma")
        self.assertEqual(metadata.account_number, "****9876")
        self.assertEqual(metadata.statement_period_start, "09/01/2025")
        self.assertEqual(metadata.statement_period_end, "09/30/2025")
        self.assertEqual(metadata.statement_year, 2025)

    def test_extract_metadata_supports_masked_account_numbers(self) -> None:
        text = """
        Wells Fargo
        Acct #: ****4321
        09/01/2025 - 09/30/2025
        """

        metadata = extract_metadata(text)

        self.assertEqual(metadata.account_number, "****4321")

    def test_extract_metadata_preserves_spaced_account_numbers(self) -> None:
        text = """
        Bank of America
        Account number: 0003 3976 6400
        December 1 2025 to December 31 2025
        """

        metadata = extract_metadata(text)

        self.assertEqual(metadata.account_number, "0003 3976 6400")

    def test_extract_metadata_returns_none_for_missing_account_holder(self) -> None:
        text = """
        Citibank
        Account Number: 123456789
        09/01/2025 - 09/30/2025
        """

        metadata = extract_metadata(text)

        self.assertIsNone(metadata.account_holder)

    def test_extract_metadata_returns_none_for_missing_statement_period(self) -> None:
        text = """
        Axis Bank
        Account Holder: Ravi Kumar
        Checking Account: XXXX1234
        Notice generated on 10/05/2025
        """

        metadata = extract_metadata(text)

        self.assertIsNone(metadata.statement_period_start)
        self.assertIsNone(metadata.statement_period_end)
        self.assertEqual(metadata.statement_year, 2025)

    def test_validate_metadata_returns_false_for_reversed_period(self) -> None:
        metadata = StatementMetadata(
            bank_name=None,
            account_holder=None,
            account_number=None,
            statement_period_start="09/30/2025",
            statement_period_end="09/01/2025",
            statement_year=2025,
        )

        self.assertFalse(validate_metadata(metadata))


if __name__ == "__main__":
    unittest.main()
