from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ui.components import (
    render_download_button,
    render_logs,
    render_metadata_card,
    render_results_table,
    render_sidebar,
    render_summary_cards,
    render_upload_section,
)
from app.ui.pipeline_runner import run_pipeline


def summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the dashboard summary without depending on a component export."""
    def number(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return float(str(value).replace("$", "").replace(",", "").strip())
        except (TypeError, ValueError):
            return None

    debits = credits = 0.0
    deposits = withdrawals = 0
    confidence_values: list[float] = []
    for row in transactions:
        debit = number(row.get("debit", row.get("Debit")))
        credit = number(row.get("credit", row.get("Credit")))
        if debit is None and credit is None:
            amount = number(row.get("amount", row.get("Amount")))
            debit, credit = (abs(amount), 0.0) if amount is not None and amount < 0 else (0.0, amount or 0.0)
        debit, credit = debit or 0.0, credit or 0.0
        debits += debit
        credits += credit
        withdrawals += debit > 0
        deposits += credit > 0
        confidence = number(row.get("confidence", row.get("Confidence", row.get("classification_confidence"))))
        if confidence is not None:
            confidence_values.append(confidence)

    return {
        "total_transactions": len(transactions),
        "deposits": deposits,
        "withdrawals": withdrawals,
        "total_credits": round(credits, 2),
        "total_debits": round(debits, 2),
        "average_confidence": round(sum(confidence_values) / len(confidence_values), 2) if confidence_values else 0.0,
        "high_confidence_count": sum(value >= 90 for value in confidence_values),
        "medium_confidence_count": sum(70 <= value < 90 for value in confidence_values),
        "low_confidence_count": sum(value < 70 for value in confidence_values),
    }


@st.cache_data(show_spinner=False)
def _load_config() -> dict[str, Any]:
    return {
        "enable_gemini_verification": os.getenv("ENABLE_GEMINI_VERIFICATION", "true").strip().lower() in {"1", "true", "yes", "on"},
        "input_folder": os.getenv("INPUT_FOLDER", "test_inputs"),
        "output_folder": os.getenv("OUTPUT_FOLDER", "output"),
    }

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Bank Statement Intelligence", page_icon="🏦", layout="wide")
st.title("Bank Statement Intelligence")

settings = render_sidebar(version="1.0.0")
config = _load_config()

if "run_state" not in st.session_state:
    st.session_state.run_state = {"status": "idle", "result": None}

uploaded_file, upload_info = render_upload_section()

if uploaded_file is not None:
    if st.button("Process Statement", type="primary"):
        start_time = time.perf_counter()
        st.session_state.run_state["status"] = "running"
        st.session_state.run_state["result"] = None

        progress_placeholder = st.empty()
        progress_placeholder.info("Processing statement. Please wait...")
        progress_bar = st.progress(0)

        try:
            for index in range(1, 8):
                progress_bar.progress(index / 7)
                time.sleep(0.08)

            result = run_pipeline(uploaded_file)
            st.session_state.run_state["result"] = result
            st.session_state.run_state["status"] = "completed"
        except Exception as exc:
            logger.exception("Streamlit pipeline failed")
            st.session_state.run_state["status"] = "failed"
            st.session_state.run_state["result"] = None
            st.error(f"Unable to process the statement right now. {exc}")
        finally:
            progress_bar.progress(1.0)
            progress_placeholder.empty()

        runtime = time.perf_counter() - start_time
        st.session_state.run_state["runtime_seconds"] = runtime

if st.session_state.run_state["status"] == "running":
    st.info("The pipeline is currently processing your statement.")

result = st.session_state.run_state.get("result")
if result is not None:
    metadata = result.get("metadata", {})
    transactions = result.get("transactions", [])
    summary = summarize_transactions(transactions)
    excel_path = result.get("excel_path")

    render_metadata_card(metadata)
    render_summary_cards(summary)
    render_results_table(transactions)
    render_download_button(excel_path)

    logs = {
        "current_phase": "Completed",
        "runtime_seconds": st.session_state.run_state.get("runtime_seconds", 0.0),
        "warnings": result.get("warnings", []),
        "errors": result.get("errors", []),
    }
    render_logs(logs)
else:
    st.info("Upload a PDF and start the workflow to inspect statement details and results.")
