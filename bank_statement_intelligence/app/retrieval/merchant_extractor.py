from __future__ import annotations

import re
from typing import Any


class MerchantExtractor:
    """Extract and normalize merchant names from transaction text."""

    @staticmethod
    def extract_merchant(value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""

        text = text.replace("&", " and ")
        text = re.sub(r"\s+", " ", text)

        prefixes = [
            "pos purchase",
            "pos",
            "purchase",
            "payment",
            "payment to",
            "card purchase",
            "ach",
            "checkcard",
            "online",
            "mobile",
            "deposit",
            "withdrawal",
            "transfer",
            "cash",
            "atm",
            "sq",
        ]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break

        if "bkofamerica" in text or "bank of america" in text:
            return "bank of america"
        if "chase" in text:
            return "chase"
        if "walmart" in text:
            return "walmart"

        text = re.sub(r"^(?:sq|\*)+\s*", "", text)
        text = text.replace("*", " ")
        text = text.replace("-", " ")
        text = re.sub(r"\b(?:inc|llc|corp|co|company|bank|bofa|bkofamerica)\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return ""

        if text.startswith("amazon"):
            return "amazon"
        if "starbucks" in text:
            return "starbucks"
        if "bkofamerica" in text or "bank of america" in text:
            return "bank of america"
        if "chase" in text:
            return "chase"
        if "walmart" in text:
            return "walmart"

        return text
