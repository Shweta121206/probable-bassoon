"""LLM-backed financial transaction extraction."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.llm.ollama_client import OllamaClient
from app.models import Transaction

logger = logging.getLogger(__name__)

TRANSACTION_SECTION_PATTERN = re.compile(
    r"^\s*(?P<header>"
    r"Deposits and other credits|"
    r"Withdrawals and other debits(?:\s*-\s*continued)?|"
    r"Service fees|"
    r"Account activity|"
    r"Transactions?"
    r")"
    r".*?"
    r"(?=(?:\n(?:Total |Daily ledger balances|Page \d+ of \d+|--- Page)|\Z))",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
DATED_LINE_PATTERN = re.compile(r"^\s*\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", re.MULTILINE)
INLINE_DATE_PATTERN = re.compile(r"^\s*\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
SECTION_HEADER_PATTERN = re.compile(
    r"^\s*(?:"
    r"Deposits and other credits|"
    r"Withdrawals and other debits(?:\s*-\s*continued)?|"
    r"Service fees|"
    r"Account activity|"
    r"Transactions?|"
    r"Date Description Amount|"
    r"Total .*|"
    r"Page \d+ of \d+|"
    r"--- Page|"
    r"continued on the next page)\b",  # 'continued' may appear after a line break
    re.IGNORECASE,
)


class TransactionExtractor:
    """Extract statement transactions using Ollama and validate JSON output."""

    def __init__(self, client: OllamaClient | None = None) -> None:
        """Initialize the extractor.

        Args:
            client: Optional Ollama client. Defaults to llama3.2:3b locally.
        """
        self.client = client or OllamaClient(model="llama3.2:3b", timeout_seconds=180)
        self.last_raw_json: str | None = None

    def extract_transactions(self, raw_statement_text: str) -> list[Transaction]:
        """Extract transactions from raw statement text.

        Args:
            raw_statement_text: Full text extracted from a bank statement.

        Returns:
            Validated Transaction objects.

        Raises:
            TypeError: If raw_statement_text is not a string.
            ValueError: If the model output is malformed or invalid.
            RuntimeError: If Ollama generation fails.
        """
        if not isinstance(raw_statement_text, str):
            raise TypeError("raw_statement_text must be a string")

        extraction_text = self._prepare_extraction_text(raw_statement_text)
        logger.info(
            "Prepared transaction extraction text: %d -> %d character(s)",
            len(raw_statement_text),
            len(extraction_text),
        )

        transactions = self._parse_transactions_from_text(extraction_text)
        if transactions:
            logger.info(
                "Extracted %d transaction(s) from statement text without Ollama",
                len(transactions),
            )
            self.last_raw_json = json.dumps(
                [self._transaction_to_schema_dict(transaction) for transaction in transactions],
                indent=2,
            )
            return transactions

        prompt = self._build_prompt(extraction_text)
        logger.info("Requesting transaction extraction from Ollama")
        model_output = self.client.generate(prompt)
        parsed_json = self._parse_json_output(model_output)
        transactions = self._validate_transactions(parsed_json)
        self.last_raw_json = json.dumps(
            [self._transaction_to_schema_dict(transaction) for transaction in transactions],
            indent=2,
        )
        logger.info("Extracted %d transaction(s)", len(transactions))
        return transactions

    @staticmethod
    def _build_prompt(raw_statement_text: str) -> str:
        """Build the extraction prompt for Ollama."""
        return f"""
You are a financial transaction extraction engine.

A transaction is any dated financial event that changes account value.

Examples:

PAYROLL DEPOSIT
ONLINE TRANSFER
ACH CREDIT
ACH DEBIT
SERVICE CHARGE
MONTHLY FEE
ATM WITHDRAWAL
INTEREST PAYMENT
CARD PURCHASE

Transactions may appear under any section.

Ignore:

Balances
Daily Ledger Balances
Opening Balance
Closing Balance
Available Balance
Running Balance
Totals
Subtotals
Headers
Page Numbers
Account Information
Statement Information

Extract:

date
description
amount

If balance appears in the transaction row:

extract balance.

Otherwise:

balance = null

Return JSON only.

No markdown.

No explanation.

Use this exact JSON schema:
[
  {{
    "date": "",
    "description": "",
    "amount": 0.0,
    "balance": null,
    "transaction_type": ""
  }}
]

Statement text:
{raw_statement_text}
""".strip()

    @staticmethod
    def _prepare_extraction_text(raw_statement_text: str) -> str:
        """Keep only transaction-like statement sections before prompting Ollama."""
        sections = [
            match.group(0).strip()
            for match in TRANSACTION_SECTION_PATTERN.finditer(raw_statement_text)
            if DATED_LINE_PATTERN.search(match.group(0))
        ]

        if sections:
            return "\n\n".join(sections)

        logger.warning(
            "No transaction sections found; falling back to dated statement lines"
        )
        return TransactionExtractor._extract_dated_line_context(raw_statement_text)

    @staticmethod
    def _extract_dated_line_context(raw_statement_text: str) -> str:
        """Extract dated lines and likely continuation lines from statement text."""
        lines = raw_statement_text.splitlines()
        selected_lines: list[str] = []
        current_line: str | None = None

        for line in lines:
            stripped_line = line.rstrip()
            if DATED_LINE_PATTERN.match(stripped_line):
                if current_line is not None:
                    selected_lines.append(current_line.strip())

                current_line = stripped_line
                continue

            if current_line is not None and stripped_line and not SECTION_HEADER_PATTERN.match(stripped_line):
                current_line += " " + stripped_line.strip()
                continue

            if current_line is not None:
                selected_lines.append(current_line.strip())
                current_line = None

        if current_line is not None:
            selected_lines.append(current_line.strip())

        if not selected_lines:
            return raw_statement_text

        return "\n".join(selected_lines)

    @staticmethod
    def _parse_transactions_from_text(raw_statement_text: str) -> list[Transaction]:
        """Attempt to parse dated transactions directly from the statement text."""
        lines = raw_statement_text.splitlines()
        transactions: list[Transaction] = []
        current_line: str | None = None

        for line in lines:
            stripped_line = line.strip()
            if DATED_LINE_PATTERN.match(stripped_line):
                if current_line is not None:
                    transaction = TransactionExtractor._parse_transaction_line(current_line)
                    if transaction is not None:
                        transactions.append(transaction)
                current_line = stripped_line
                continue

            if current_line is not None and stripped_line and not SECTION_HEADER_PATTERN.match(stripped_line):
                current_line += " " + stripped_line
                continue

            if current_line is not None:
                transaction = TransactionExtractor._parse_transaction_line(current_line)
                if transaction is not None:
                    transactions.append(transaction)
                current_line = None

        if current_line is not None:
            transaction = TransactionExtractor._parse_transaction_line(current_line)
            if transaction is not None:
                transactions.append(transaction)

        return transactions

    @staticmethod
    def _parse_transaction_line(line: str) -> Transaction | None:
        """Parse a single transaction line into a Transaction if it has a valid amount."""
        match = DATED_LINE_PATTERN.match(line.strip())
        if not match:
            return None

        date_part = match.group(0)
        body = line.strip()[match.end() :].strip()
        if not body:
            return None

        amount_matches = list(
            re.finditer(r"-?\d{1,3}(?:,\d{3})*(?:\.\d{2})", body)
        )
        if not amount_matches:
            return None

        last_amount = amount_matches[-1]
        amount_text = last_amount.group(0)
        description = (body[: last_amount.start()] + body[last_amount.end() :]).strip()

        if not description:
            return None

        try:
            amount = TransactionExtractor._coerce_number(amount_text, "amount")
        except (TypeError, ValueError):
            return None

        try:
            return Transaction(
                date=date_part.strip(),
                description=description,
                amount=amount,
                balance=None,
                transaction_type="",
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_json_output(model_output: str) -> Any:
        """Parse JSON from model output and reject malformed responses."""
        cleaned_output = model_output.strip()
        if cleaned_output.startswith("```"):
            cleaned_output = re.sub(r"^```(?:json)?\s*", "", cleaned_output)
            cleaned_output = re.sub(r"\s*```$", "", cleaned_output)

        try:
            return json.loads(cleaned_output)
        except json.JSONDecodeError as exc:
            logger.exception("Transaction extractor returned malformed JSON")
            raise ValueError("transaction extractor returned malformed JSON") from exc

    @staticmethod
    def _validate_transactions(parsed_json: Any) -> list[Transaction]:
        """Validate parsed JSON and convert it to Transaction objects."""
        if isinstance(parsed_json, dict) and "transactions" in parsed_json:
            parsed_json = parsed_json["transactions"]

        if not isinstance(parsed_json, list):
            raise ValueError("transaction extractor JSON must be a list")

        transactions: list[Transaction] = []
        for index, item in enumerate(parsed_json, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"transaction {index} must be an object")

            required_fields = {"date", "description", "amount"}
            missing_fields = required_fields - set(item)
            if missing_fields:
                raise ValueError(
                    f"transaction {index} missing fields: {sorted(missing_fields)}"
                )

            try:
                balance = item.get("balance", None)
                if balance == "":
                    balance = None

                transaction_type = item.get("transaction_type", "")
                if transaction_type is None:
                    transaction_type = ""

                transaction = Transaction(
                    date=item["date"],
                    description=item["description"],
                    amount=TransactionExtractor._coerce_number(item["amount"], "amount"),
                    balance=TransactionExtractor._coerce_optional_number(
                        balance,
                        "balance",
                    ),
                    transaction_type=TransactionExtractor._coerce_string(
                        transaction_type,
                        "transaction_type",
                    ),
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"transaction {index} is invalid: {exc}") from exc

            transactions.append(transaction)

        return transactions

    @staticmethod
    def _coerce_number(value: Any, field_name: str) -> float:
        """Coerce a JSON number or money-like string to float."""
        if isinstance(value, bool):
            raise TypeError(f"{field_name} must be a number")

        if isinstance(value, int | float):
            return float(value)

        if isinstance(value, str):
            cleaned_value = value.strip().replace("$", "").replace(",", "")
            if not cleaned_value:
                raise ValueError(f"{field_name} cannot be empty")
            return float(cleaned_value)

        raise TypeError(f"{field_name} must be a number")

    @staticmethod
    def _coerce_optional_number(value: Any, field_name: str) -> float | None:
        """Coerce an optional JSON number or money-like string to float."""
        if value is None:
            return None

        return TransactionExtractor._coerce_number(value, field_name)

    @staticmethod
    def _coerce_string(value: Any, field_name: str) -> str:
        """Coerce a JSON value to a string, allowing empty text."""
        if value is None:
            return ""

        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")

        return value.strip()

    @staticmethod
    def _transaction_to_schema_dict(
        transaction: Transaction,
    ) -> dict[str, str | float | None]:
        """Convert a Transaction to the Phase 4 JSON schema."""
        return {
            "date": transaction.date,
            "description": transaction.description,
            "amount": transaction.amount,
            "balance": transaction.balance,
            "transaction_type": transaction.transaction_type,
        }
