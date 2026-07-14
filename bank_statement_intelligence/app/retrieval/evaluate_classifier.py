from __future__ import annotations

from pathlib import Path

from app.retrieval.evaluation import EvaluationTool


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    classified_path = project_root / "output" / "classified_transactions.json"
    report_path = project_root / "output" / "evaluation_report.xlsx"

    evaluator = EvaluationTool()
    output_file = evaluator.evaluate(classified_path=classified_path, output_path=report_path)
    print(f"Evaluation report written to: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
