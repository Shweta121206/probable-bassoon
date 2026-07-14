"""Accounting AI classification package."""

from app.classification.chart_loader import Account, ChartLoader
from app.classification.example_loader import HistoricalExample, HistoricalExampleSimilarity, ExampleLoader
from app.classification.prompt_builder import build_classification_prompt

try:
    from app.classification.transaction_classifier import TransactionClassifier
except ImportError:
    TransactionClassifier = None  # type: ignore[assignment]

try:
    from app.classification.gemini_classifier import GeminiClassifier, ClassificationResult
except ImportError:
    GeminiClassifier = None  # type: ignore[assignment]
    ClassificationResult = None  # type: ignore[assignment]

__all__ = [
    "Account",
    "ChartLoader",
    "HistoricalExample",
    "HistoricalExampleSimilarity",
    "ExampleLoader",
    "GeminiClassifier",
    "ClassificationResult",
    "build_classification_prompt",
    "TransactionClassifier",
]
