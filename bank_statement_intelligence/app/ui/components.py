from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


def _coerce_numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("$", "").replace(",", "")
        if not cleaned or cleaned.lower() in {"none", "nan", "null"}:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _get_field_value(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    if not isinstance(row, dict):
        return None

    for name in names:
        if name in row:
            return row.get(name)

    normalized_keys = {str(key).strip().lower(): key for key in row.keys() if isinstance(key, str)}
    for name in names:
        key = normalized_keys.get(name.lower())
        if key is not None:
            return row.get(key)

    return None


def _derive_debit_credit(row: dict[str, Any]) -> tuple[float, float]:
    debit_value = _coerce_numeric(_get_field_value(row, ("debit", "Debit")))
    credit_value = _coerce_numeric(_get_field_value(row, ("credit", "Credit")))

    if debit_value is not None or credit_value is not None:
        return (debit_value or 0.0), (credit_value or 0.0)

    amount_value = _coerce_numeric(_get_field_value(row, ("amount", "Amount")))
    if amount_value is None:
        return 0.0, 0.0

    if amount_value > 0:
        return 0.0, amount_value
    return abs(amount_value), 0.0


def summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    total_transactions = len(transactions)
    deposits = 0
    withdrawals = 0
    total_credits = 0.0
    total_debits = 0.0
    confidence_values: list[float] = []

    for transaction in transactions:
        debit, credit = _derive_debit_credit(transaction)
        if credit > 0:
            deposits += 1
        if debit > 0:
            withdrawals += 1
        total_credits += credit
        total_debits += debit

        confidence_value = _coerce_numeric(
            _get_field_value(transaction, ("confidence", "Confidence", "classification_confidence"))
        )
        if confidence_value is not None:
            confidence_values.append(confidence_value)

    average_confidence = round(sum(confidence_values) / len(confidence_values), 2) if confidence_values else 0.0
    high_confidence_count = sum(1 for value in confidence_values if value >= 90)
    medium_confidence_count = sum(1 for value in confidence_values if 70 <= value < 90)
    low_confidence_count = sum(1 for value in confidence_values if value < 70)

    return {
        "total_transactions": total_transactions,
        "deposits": deposits,
        "withdrawals": withdrawals,
        "total_credits": round(total_credits, 2),
        "total_debits": round(total_debits, 2),
        "average_confidence": average_confidence,
        "high_confidence_count": high_confidence_count,
        "medium_confidence_count": medium_confidence_count,
        "low_confidence_count": low_confidence_count,
    }


def render_sidebar(version: str = "1.0.0") -> dict[str, Any]:
    st.sidebar.image("https://via.placeholder.com/240x120?text=Bank+Statement+Intelligence", use_container_width=True)
    st.sidebar.markdown("### Bank Statement Intelligence")
    st.sidebar.caption(f"Version {version}")
    st.sidebar.markdown("### Pipeline Status")
    st.sidebar.success("Ready")

    with st.sidebar.expander("Settings", expanded=False):
        st.checkbox("Enable Gemini verification", value=True, key="enable_gemini")
        st.text_input("Input folder", value="test_inputs", key="input_folder")
        st.text_input("Output folder", value="output", key="output_folder")

    return {
        "enable_gemini": st.session_state.get("enable_gemini", True),
        "input_folder": st.session_state.get("input_folder", "test_inputs"),
        "output_folder": st.session_state.get("output_folder", "output"),
    }


def render_upload_section() -> tuple[Path | None, dict[str, Any] | None]:
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is None:
        return None, None

    file_size = getattr(uploaded_file, "size", 0) or 0
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_handle:
        temp_handle.write(uploaded_file.getvalue())
        upload_path = Path(temp_handle.name)

    st.success(f"Loaded {uploaded_file.name}")
    st.caption(f"File size: {file_size} bytes")
    return upload_path, {"name": uploaded_file.name, "size_bytes": file_size}


def render_metadata_card(metadata: dict[str, Any]) -> None:
    st.subheader("Statement Information")
    cols = st.columns(3)
    cols[0].metric("Bank Name", metadata.get("bank_name") or "Unknown")
    cols[1].metric("Account Holder", metadata.get("account_holder") or "Unknown")
    cols[2].metric("Account Number", metadata.get("account_number") or "Unknown")

    cols = st.columns(2)
    cols[0].metric("Statement Period", metadata.get("statement_period") or "Unknown")
    cols[1].metric("Statement Year", metadata.get("statement_year") or "Unknown")


def render_summary_cards(summary: dict[str, Any]) -> None:
    st.subheader("Transaction Summary")
    cols = st.columns(5)
    cols[0].metric("Total transactions", summary.get("total_transactions", 0))
    cols[1].metric("Deposits", summary.get("deposits", 0))
    cols[2].metric("Withdrawals", summary.get("withdrawals", 0))
    cols[3].metric("Total credits", f"{summary.get('total_credits', 0):,.2f}")
    cols[4].metric("Total debits", f"{summary.get('total_debits', 0):,.2f}")

    st.subheader("Classification Summary")
    cols = st.columns(4)
    cols[0].metric("Number classified", summary.get("total_transactions", 0))
    cols[1].metric("Average confidence", f"{summary.get('average_confidence', 0):.2f}")
    cols[2].metric("High confidence", summary.get("high_confidence_count", 0))
    cols[3].metric("Medium confidence", summary.get("medium_confidence_count", 0))

    st.metric("Low confidence", summary.get("low_confidence_count", 0))


def render_progress(steps: list[str], completed: int) -> None:
    st.subheader("Processing Progress")
    progress = completed / max(len(steps), 1)
    st.progress(progress)
    for index, step in enumerate(steps, start=1):
        status = "✓" if index <= completed else "•"
        st.text(f"{status} {step}")


def render_results_table(transactions: list[dict[str, Any]]) -> None:
    if not transactions:
        st.info("No transactions available yet.")
        return

    rows: list[dict[str, Any]] = []
    for transaction in transactions:
        debit_value, credit_value = _derive_debit_credit(transaction)
        row: dict[str, Any] = {
            "Date": _get_field_value(transaction, ("date", "Date", "transaction_date")),
            "Description": _get_field_value(transaction, ("description", "Description", "merchant_description")),
            "Debit": debit_value,
            "Credit": credit_value,
        }

        for column_name in [
            "classification_code",
            "classification_name",
            "classification_type",
            "confidence",
            "verification_method",
            "verification_confidence",
            "method",
        ]:
            row[column_name] = _get_field_value(transaction, (column_name, column_name.title()))

        rows.append(row)

    frame = pd.DataFrame(rows)
    frame = frame[[
        "Date",
        "Description",
        "Debit",
        "Credit",
        "classification_code",
        "classification_name",
        "classification_type",
        "confidence",
        "verification_method",
        "verification_confidence",
        "method",
    ]].copy()
    frame.columns = [
        "Date",
        "Description",
        "Debit",
        "Credit",
        "Classification Code",
        "Classification Name",
        "Classification Type",
        "Confidence",
        "Verification Method",
        "Verification Confidence",
        "Method",
    ]
    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_download_button(excel_path: str | None) -> None:
    if not excel_path:
        return
    with open(excel_path, "rb") as handle:
        st.download_button("Download Excel", handle, file_name="classified_statement.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def render_logs(logs: dict[str, Any]) -> None:
    with st.expander("Logs", expanded=False):
        st.write(f"Current phase: {logs.get('current_phase', 'Idle')}")
        st.write(f"Runtime: {logs.get('runtime_seconds', 0):.2f}s")
        st.write("Warnings")
        for warning in logs.get("warnings", []):
            st.warning(warning)
        st.write("Errors")
        for error in logs.get("errors", []):
            st.error(error)
