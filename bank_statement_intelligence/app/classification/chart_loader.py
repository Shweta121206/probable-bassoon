from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

CHART_FILE_PATH = Path("config/chart_of_accounts.xlsx")


@dataclass(frozen=True)
class Account:
    code: str
    name: str
    type: str


class ChartLoader:
    """Load the chart of accounts from a worksheet."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else CHART_FILE_PATH
        self.accounts: list[Account] = []
        self._load_chart()

    def _load_chart(self) -> None:
        if not self.path.exists():
            logger.error("Chart of accounts file not found at %s", self.path)
            self.accounts = []
            return

        workbook = pd.read_excel(self.path, engine="openpyxl")
        normalized_columns = {
            self._normalize_header(str(column)): column for column in workbook.columns
        }

        code_key = self._find_header(normalized_columns, ["code", "account code", "*code"])
        name_key = self._find_header(normalized_columns, ["name", "account name", "*name"])
        type_key = self._find_header(normalized_columns, ["type", "account type", "*type"])

        if code_key is None or name_key is None or type_key is None:
            logger.error("Chart of accounts workbook is missing required columns: %s", workbook.columns.tolist())
            self.accounts = []
            return

        for _, row in workbook.iterrows():
            account_code = row.get(code_key)
            account_name = row.get(name_key)
            account_type = row.get(type_key)
            if account_name is None:
                continue

            self.accounts.append(
                Account(
                    code=str(account_code).strip() if account_code is not None else "",
                    name=str(account_name).strip(),
                    type=str(account_type).strip() if account_type is not None else "",
                )
            )

        logger.info("Loaded %d chart accounts from %s", len(self.accounts), self.path)

    @staticmethod
    def _normalize_header(header: str) -> str:
        return header.strip().lower().replace(" ", "")

    @staticmethod
    def _find_header(normalized_columns: dict[str, Any], candidates: list[str]) -> str | None:
        for candidate in candidates:
            normalized_candidate = candidate.strip().lower().replace(" ", "")
            if normalized_candidate in normalized_columns:
                return normalized_columns[normalized_candidate]
        return None

    def get_accounts(self) -> list[Account]:
        return list(self.accounts)
