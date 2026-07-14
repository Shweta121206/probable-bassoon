from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.classification.chart_loader import Account
from app.classification.example_loader import HistoricalExampleSimilarity


@dataclass(frozen=True)
class PromptContext:
    transaction_date: str
    description: str
    debit: float | None
    credit: float | None
    balance: float | None
    transaction_type: str


def _format_account_list(accounts: Iterable[Account]) -> str:
    lines = ["Chart of Accounts:\n", "Code | Name | Type"]
    for account in accounts:
        lines.append(f"{account.code} | {account.name} | {account.type}")
    return "\n".join(lines)


def _format_historical_examples(examples: Iterable[HistoricalExampleSimilarity]) -> str:
    lines = ["Previous Similar Transactions:"]
    for example in examples:
        lines.append(f"Similarity {example.similarity:.2f}")
        lines.append(example.example.description)
        lines.append("↓")
        lines.append(example.example.classification_name)
    return "\n".join(lines)


def build_classification_prompt(
    transaction: PromptContext,
    similar_examples: list[HistoricalExampleSimilarity],
    accounts: list[Account],
) -> str:
    prompt_parts: list[str] = [
        "SYSTEM:\n",
        "You are a senior accounting professional.",
        "You classify bank transactions for a bookkeeping system.",
        "You NEVER invent accounts.",
        "You MUST choose exactly ONE account from the provided chart of accounts.",
        "You reason carefully before answering.",
        "\nINPUT:\n",
        f"Date: {transaction.transaction_date}",
        f"Description: {transaction.description}",
        f"Debit: {transaction.debit}",
        f"Credit: {transaction.credit}",
        f"Balance: {transaction.balance}",
        f"Transaction Type: {transaction.transaction_type}",
        "\n\n",
        _format_historical_examples(similar_examples),
        "\n\n",
        _format_account_list(accounts),
        "\n\nINSTRUCTIONS:\n",
        "Reason like an accountant.",
        "Look at merchant names, keywords, transaction type, debit or credit, historical examples.",
        "Do NOT invent categories.",
        "Choose ONLY one.",
        "\n\nOUTPUT:\n",
        "Return STRICT JSON ONLY",
        "{",
        '"classification_code":"<code>",',
        '"classification_name":"<name>",',
        '"classification_type":"<type>",',
        '"confidence":<number>,',
        '"reason":"<reason>"',
        "}",
    ]

    return "\n".join(prompt_parts)
