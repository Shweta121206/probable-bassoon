"""Smoke test script for the Excel transaction report generator."""

from __future__ import annotations

import logging

from app.reports import create_transactions_workbook


def main() -> None:
    """Create a sample transaction workbook."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    transactions = [
        {
            "date": "12/01/25",
            "description": "MOBILE DEPOSIT",
            "amount": 2500,
        }
    ]

    output_path = create_transactions_workbook(
        transactions=transactions,
        output_path="transactions_report.xlsx",
    )
    print(f"Created Excel report: {output_path}")


if __name__ == "__main__":
    main()
