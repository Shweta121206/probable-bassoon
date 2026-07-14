from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.classification.example_loader import ExampleLoader
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    precision: float
    recall: float
    top_1_accuracy: float
    top_3_accuracy: float
    confusion_matrix: dict[str, dict[str, int]]


class EvaluationTool:
    """Evaluate retrieval classifier output against historical examples."""

    def __init__(self, historical_examples_path: str | Path | None = None) -> None:
        self.historical_examples_path = Path(historical_examples_path) if historical_examples_path is not None else Path("config/historical_examples.xlsx")
        self.examples = ExampleLoader(path=self.historical_examples_path).get_examples()
        self.truth_map = self._build_truth_map(self.examples)

    def _build_truth_map(self, examples: list[dict[str, Any]]) -> dict[str, list[str]]:
        truth_map: dict[str, list[str]] = {}
        for example in examples:
            description = str(example.description).strip().lower()
            if not description:
                continue
            truth_map.setdefault(description, []).append(example.classification_name)
        return truth_map

    def evaluate(self, classified_path: str | Path | None = None, output_path: str | Path | None = None) -> Path:
        classified_path = Path(classified_path) if classified_path is not None else Path("output/classified_transactions.json")
        output_path = Path(output_path) if output_path is not None else Path("output/evaluation_report.xlsx")

        with classified_path.open("r", encoding="utf-8") as handle:
            classified = json.load(handle)

        report = self._calculate_metrics(classified)
        self._save_report(report, output_path)
        return output_path

    def _calculate_metrics(self, classified: list[dict[str, Any]]) -> EvaluationReport:
        if not classified:
            return EvaluationReport(0.0, 0.0, 0.0, 0.0, 0.0, {})

        predictions: list[str] = []
        truths: list[str] = []
        evaluated_count = 0

        for item in classified:
            prediction = str(item.get("classification_name", "")).strip()
            description = str(item.get("description", "")).strip().lower()
            truth_candidates = self.truth_map.get(description, [])
            truth = truth_candidates[0] if truth_candidates else ""
            if truth:
                evaluated_count += 1
            predictions.append(prediction)
            truths.append(truth)

        matched = sum(
            1 for prediction, truth in zip(predictions, truths)
            if truth and prediction == truth
        )
        accuracy = matched / max(evaluated_count, 1)
        precision = self._precision(predictions, truths)
        recall = self._recall(predictions, truths)
        top_1 = accuracy
        top_3 = self._top_k_accuracy(classified, 3)

        confusion = self._build_confusion_matrix(predictions, truths)

        return EvaluationReport(
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            top_1_accuracy=round(top_1, 4),
            top_3_accuracy=round(top_3, 4),
            confusion_matrix=confusion,
        )

    def _precision(self, predictions: list[str], truths: list[str]) -> float:
        true_positives = sum(1 for prediction, truth in zip(predictions, truths) if prediction == truth and truth)
        predicted_positives = sum(1 for prediction in predictions if prediction)
        return true_positives / max(predicted_positives, 1)

    def _recall(self, predictions: list[str], truths: list[str]) -> float:
        true_positives = sum(1 for prediction, truth in zip(predictions, truths) if prediction == truth and truth)
        actual_positives = sum(1 for truth in truths if truth)
        return true_positives / max(actual_positives, 1)

    def _top_k_accuracy(self, classified: list[dict[str, Any]], k: int) -> float:
        if not classified:
            return 0.0

        top_k_matches = 0
        for item in classified:
            candidates = item.get("top_candidates", [])
            if not candidates:
                continue
            truth = item.get("classification_name", "")
            if any(candidate.get("classification_name", "") == truth for candidate in candidates[:k]):
                top_k_matches += 1
        return top_k_matches / len(classified)

    def _build_confusion_matrix(self, predictions: list[str], truths: list[str]) -> dict[str, dict[str, int]]:
        matrix: dict[str, dict[str, int]] = {}
        labels = sorted(set(truths + predictions))
        for truth in labels:
            matrix[truth] = {label: 0 for label in labels}
        for prediction, truth in zip(predictions, truths):
            matrix.setdefault(truth, {}).setdefault(prediction, 0)
            matrix[truth][prediction] += 1
        return matrix

    def _save_report(self, report: EvaluationReport, output_path: Path) -> None:
        workbook = pd.ExcelWriter(output_path, engine="openpyxl")
        summary = {
            "Metric": ["Accuracy", "Precision", "Recall", "Top-1 Accuracy", "Top-3 Accuracy"],
            "Value": [
                report.accuracy,
                report.precision,
                report.recall,
                report.top_1_accuracy,
                report.top_3_accuracy,
            ],
        }
        pd.DataFrame(summary).to_excel(workbook, index=False, sheet_name="Summary")

        matrix_rows: list[dict[str, Any]] = []
        for truth, row in report.confusion_matrix.items():
            row_data = {"Truth": truth}
            row_data.update(row)
            matrix_rows.append(row_data)

        if matrix_rows:
            pd.DataFrame(matrix_rows).to_excel(workbook, index=False, sheet_name="ConfusionMatrix")

        workbook.close()
