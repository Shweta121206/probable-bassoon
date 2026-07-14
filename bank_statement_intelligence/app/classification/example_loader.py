from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

HISTORICAL_EXAMPLES_PATH = Path("config/historical_examples.xlsx")


@dataclass(frozen=True)
class HistoricalExample:
    description: str
    debit: float | None
    credit: float | None
    name: str
    item_split_account: str | None
    transaction_type: str
    classification_name: str
    classification_type: str


@dataclass(frozen=True)
class HistoricalExampleSimilarity:
    example: HistoricalExample
    similarity: float


class ExampleLoader:
    """Load historical example transactions from a worksheet."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else HISTORICAL_EXAMPLES_PATH
        self.examples: list[HistoricalExample] = []
        self._load_examples()

    def _load_examples(self) -> None:
        if not self.path.exists():
            logger.error("Historical examples file not found at %s", self.path)
            self.examples = []
            return

        try:
            workbook = pd.read_excel(self.path, engine="openpyxl", sheet_name=None, header=None)
        except Exception as exc:
            logger.error("Unable to read historical examples workbook %s: %s", self.path, exc)
            self.examples = []
            return

        sheet_data = self._find_example_sheet(workbook)
        if sheet_data is None or sheet_data.empty:
            logger.error("Historical examples workbook did not contain a valid transaction sheet: %s", self.path)
            self.examples = []
            return

        normalized_columns = {
            self._normalize_header(str(column)): column for column in sheet_data.columns
        }

        description_key = self._find_header(normalized_columns, ["description"])
        debit_key = self._find_header(normalized_columns, ["debit"])
        credit_key = self._find_header(normalized_columns, ["credit"])
        name_key = self._find_header(normalized_columns, ["name", "classification name"])
        item_split_key = self._find_header(normalized_columns, ["item_split_account", "item split account"])
        transaction_type_key = self._find_header(normalized_columns, ["transaction_type", "transaction type"])
        classification_type_key = transaction_type_key

        if description_key is None or classification_type_key is None:
            logger.error("Historical examples workbook is missing required columns: %s", list(sheet_data.columns))
            self.examples = []
            return

        for _, row in sheet_data.iterrows():
            description = str(row.get(description_key, "")).strip()
            if not description:
                continue

            name_value = self._to_optional_str(row.get(name_key)) if name_key else None
            item_split_value = self._to_optional_str(row.get(item_split_key)) if item_split_key else None
            classification_name_value = item_split_value or name_value
            if not classification_name_value:
                continue

            self.examples.append(
                HistoricalExample(
                    description=description,
                    debit=self._to_optional_float(row.get(debit_key)) if debit_key else None,
                    credit=self._to_optional_float(row.get(credit_key)) if credit_key else None,
                    name=name_value or "",
                    item_split_account=item_split_value,
                    transaction_type=str(row.get(transaction_type_key, "")).strip() if transaction_type_key else "",
                    classification_name=classification_name_value,
                    classification_type=str(row.get(classification_type_key, "")).strip(),
                )
            )

        logger.info("Loaded %d historical examples from %s", len(self.examples), self.path)

    def _find_example_sheet(self, workbook: dict[str, "pd.DataFrame"]) -> "pd.DataFrame" | None:
        for sheet_name, sheet_df in workbook.items():
            header_row = self._detect_header_row(sheet_df)
            if header_row is None:
                continue

            try:
                candidate_df = pd.read_excel(self.path, engine="openpyxl", sheet_name=sheet_name, header=header_row)
            except Exception:
                continue

            normalized_columns = {
                self._normalize_header(str(column)): column for column in candidate_df.columns
            }
            if self._find_header(normalized_columns, ["description"]) is not None and (
                self._find_header(normalized_columns, ["item_split_account", "item split account"]) is not None
                or self._find_header(normalized_columns, ["name"])
            ):
                return candidate_df

        return None

    def _detect_header_row(self, sheet_df: "pd.DataFrame") -> int | None:
        candidate_header_keys = {
            "description",
            "transactiontype",
            "transaction_type",
            "name",
            "item_split_account",
            "item split account",
        }

        for row_index in range(min(10, len(sheet_df))):
            row = sheet_df.iloc[row_index].fillna("").astype(str).tolist()
            normalized = {
                self._normalize_header(str(cell))
                for cell in row
                if str(cell).strip()
            }

            if "description" in normalized and (
                "transactiontype" in normalized
                or "transaction_type" in normalized
            ):
                return row_index

            if "description" in normalized and (
                "name" in normalized
                or "item_split_account" in normalized
                or "item split account" in normalized
            ):
                return row_index

        return None

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

    @staticmethod
    def _to_optional_float(value: Any) -> float | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned if cleaned else None

    def get_examples(self) -> list[HistoricalExample]:
        return list(self.examples)
