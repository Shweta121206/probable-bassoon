from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from app.retrieval.retrieval_classifier import RetrievalClassifier
from app.retrieval.evaluation import EvaluationTool
from app.classification.example_loader import ExampleLoader


class RetrievalClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.example_path = Path(self.temp_dir.name) / "historical_examples.xlsx"
        self.output_path = Path(self.temp_dir.name) / "output.json"
        self._create_historical_examples()

    def _create_historical_examples(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Sheet1"
        sheet.append([
            "Transaction date",
            "Transaction type",
            "Num",
            "Name",
            "Description",
            "Full name",
            "Item split account",
            "Debit",
            "Credit",
            "Balance",
        ])
        sheet.append(["01/01/2025", "Expense", "", "Office Depot", "Office supplies purchase", "Company Checking", "Office Supplies", 150.0, None, 9850.0])
        sheet.append(["01/02/2025", "Expense", "", "Delta Airlines", "Travel ticket purchase", "Company Checking", "Travel", 300.0, None, 9550.0])
        sheet.append(["01/03/2025", "Expense", "", "Starbucks", "Coffee meeting", "Company Checking", "Office Refreshments", 25.0, None, 9525.0])
        workbook.save(self.example_path)

    def test_retrieval_classifier_returns_predictions_for_transactions(self) -> None:
        classifier = RetrievalClassifier(historical_examples_path=self.example_path)
        transactions = [
            {"description": "Office supply order at Office Depot", "transaction_type": "Expense", "debit": 150.0, "credit": None},
            {"description": "Airline ticket purchase", "transaction_type": "Expense", "debit": 300.0, "credit": None},
        ]

        records = classifier.classify_transactions(transactions)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["method"], "Retrieval")
        self.assertIn(records[0]["status"], {"Certain", "Very High", "High", "Medium", "Low"})
        self.assertEqual(records[1]["method"], "Retrieval")
        self.assertIn(records[1]["top_candidates"][0]["classification_name"], {"Office Supplies", "Travel"})
        self.assertIsInstance(records[0]["confidence"], int)
        self.assertGreaterEqual(records[0]["confidence"], 0)
        self.assertLessEqual(records[0]["confidence"], 100)

    def test_save_classified_transactions_writes_json(self) -> None:
        classifier = RetrievalClassifier(historical_examples_path=self.example_path)
        records = classifier.classify_transactions([
            {"description": "Office supply order", "transaction_type": "Expense", "debit": 150.0, "credit": None},
        ])

        output_file = classifier.save_classified_transactions(records, self.output_path)
        self.assertTrue(output_file.exists())
        data = json.loads(output_file.read_text(encoding="utf-8"))
        self.assertEqual(data[0]["method"], "Retrieval")


class EvaluationToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.example_path = Path(self.temp_dir.name) / "historical_examples.xlsx"
        self.classified_path = Path(self.temp_dir.name) / "classified_transactions.json"
        self.report_path = Path(self.temp_dir.name) / "evaluation_report.xlsx"
        self._create_historical_examples()
        self._create_classified_output()

    def _create_historical_examples(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Sheet1"
        sheet.append([
            "Transaction date",
            "Transaction type",
            "Num",
            "Name",
            "Description",
            "Full name",
            "Item split account",
            "Debit",
            "Credit",
            "Balance",
        ])
        sheet.append(["01/01/2025", "Expense", "", "Office Depot", "Office supplies purchase", "Company Checking", "Office Supplies", 150.0, None, 9850.0])
        workbook.save(self.example_path)

    def _create_classified_output(self) -> None:
        records = [
            {
                "description": "Office supplies purchase",
                "classification_name": "Office Supplies",
                "top_candidates": [
                    {"classification_name": "Office Supplies"},
                ],
            }
        ]
        self.classified_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def test_evaluate_generates_report_file(self) -> None:
        evaluator = EvaluationTool(historical_examples_path=self.example_path)
        report_file = evaluator.evaluate(self.classified_path, self.report_path)

        self.assertTrue(report_file.exists())
        self.assertEqual(report_file, self.report_path)


if __name__ == "__main__":
    unittest.main()
