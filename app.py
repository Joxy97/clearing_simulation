from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


STAGES = [
    "Market State",
    "Proposed Trades",
    "Accepted Trades",
    "Margin Calls",
    "PnL & Returns",
    "Portfolios",
]


def _load_payload(uploaded_file) -> Dict[str, Any]:
    raw = uploaded_file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("utf-8")
    return json.loads(text)


def _metrics_list(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and "metrics" in payload:
        return payload["metrics"]
    if isinstance(payload, list):
        return payload
    return []


def _client_table(cms: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cm_name, cm_data in cms.items():
        for client in cm_data.get("clients", []):
            row = {
                "cm": cm_name,
                "client_id": client.get("client_id"),
                "vip_status": client.get("vip_status"),
                "liquidity_status_end": client.get("liquidity_status_end"),
                "wealth_end": client.get("wealth_end"),
                "collateral_end": client.get("collateral_end"),
                "margin": client.get("margin"),
                "shortfall": client.get("shortfall"),
                "pnl": client.get("pnl"),
                "income_applied": client.get("income_applied"),
            }
            margin_call = client.get("margin_call", {})
            row["margin_called"] = margin_call.get("called")
            row["margin_call_amount"] = margin_call.get("amount")
            row["margin_call_accepted"] = margin_call.get("accepted")
            row["margin_call_liquidated"] = margin_call.get("liquidated")
            rows.append(row)
    return pd.DataFrame(rows)


def _cm_table(cms: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, data in cms.items():
        rows.append(
            {
                "cm": name,
                "cm_funds": data.get("cm_funds"),
                "cm_pnl": data.get("cm_pnl"),
                "default_shortfall": data.get("default_shortfall"),
                "total_client_collateral": data.get("total_client_collateral"),
                "avg_client_collateral": data.get("avg_client_collateral"),
                "min_client_collateral": data.get("min_client_collateral"),
                "avg_client_margin": data.get("avg_client_margin"),
                "total_client_margin": data.get("total_client_margin"),
                "num_active_trades": data.get("num_active_trades"),
                "num_accepted_trades": data.get("num_accepted_trades"),
                "num_zero_collateral_clients": data.get("num_zero_collateral_clients"),
            }
        )
    return pd.DataFrame(rows)


def _ccp_table(ccps: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, data in ccps.items():
        rows.append(
            {
                "ccp": name,
                "ccp_margin": data.get("ccp_margin"),
                "cm_margins": data.get("cm_margins"),
                "cm_funds": data.get("cm_funds"),
                "cm_shortfalls": data.get("cm_shortfalls"),
            }
        )
    return pd.DataFrame(rows)


def _trades_table(cms: Dict[str, Any], accepted_only: bool) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cm_name, cm_data in cms.items():
        for client in cm_data.get("clients", []):
            for trade in client.get("trades", []):
                if accepted_only and not trade.get("accepted", False):
                    continue
                rows.append(
                    {
                        "cm": cm_name,
                        "client_id": client.get("client_id"),
                        "instrument": trade.get("instrument"),
                        "amount": trade.get("amount"),
                        "accepted": trade.get("accepted"),
                    }
                )
    return pd.DataFrame(rows)


def _portfolio_table(cms: Dict[str, Any], key: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cm_name, cm_data in cms.items():
        for client in cm_data.get("clients", []):
            portfolio = client.get(key)
            if portfolio is None:
                continue
            for idx, value in enumerate(portfolio):
                rows.append(
                    {
                        "cm": cm_name,
                        "client_id": client.get("client_id"),
                        "instrument": idx,
                        "position": value,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Clearing Simulation Viewer", layout="wide")
    st.title("Clearing Simulation Viewer")

    with st.sidebar:
        uploaded = st.file_uploader("Upload metrics.json", type=["json"])
        st.markdown("---")
        show_raw = st.checkbox("Show raw JSON", value=False)

    if not uploaded:
        st.info("Upload a metrics.json file to begin.")
        return

    payload = _load_payload(uploaded)
    metrics = _metrics_list(payload)
    if not metrics:
        st.error("No metrics found in uploaded file.")
        return

    day_count = len(metrics)
    day_index = st.slider("Day", min_value=0, max_value=day_count - 1, value=0, step=1)
    stage = st.radio("Stage", STAGES, horizontal=True)

    day_data = metrics[day_index]
    system = day_data.get("system", {})
    cms = day_data.get("cms", {})
    ccps = day_data.get("ccps", {})
    details = day_data.get("details", {})

    st.subheader(f"Day {day_index}")
    cols = st.columns(5)
    cols[0].metric("Total Collateral", system.get("total_client_collateral"))
    cols[1].metric("Avg Collateral", system.get("avg_client_collateral"))
    cols[2].metric("Total CM Funds", system.get("total_cm_funds"))
    cols[3].metric("Active Trades", system.get("num_active_trades"))
    cols[4].metric("Accepted Trades", system.get("num_accepted_trades"))

    st.markdown("---")
    st.subheader("Stage View")

    if stage == "Market State":
        st.write("Market index:", day_data.get("market_index"))
        st.write("Market state:", details.get("market_state"))

    if stage == "Proposed Trades":
        st.dataframe(_trades_table(cms, accepted_only=False), use_container_width=True)

    if stage == "Accepted Trades":
        st.dataframe(_trades_table(cms, accepted_only=True), use_container_width=True)

    if stage == "Margin Calls":
        st.dataframe(_client_table(cms), use_container_width=True)

    if stage == "PnL & Returns":
        st.dataframe(_client_table(cms), use_container_width=True)
        st.write("Real returns:", details.get("real_returns"))

    if stage == "Portfolios":
        st.write("Client portfolios (start of day):")
        st.dataframe(_portfolio_table(cms, "portfolio_start"), use_container_width=True)
        st.write("Client portfolios (end of day):")
        st.dataframe(_portfolio_table(cms, "portfolio_end"), use_container_width=True)

    st.markdown("---")
    st.subheader("CM Summary")
    st.dataframe(_cm_table(cms), use_container_width=True)

    st.subheader("CCP Summary")
    st.dataframe(_ccp_table(ccps), use_container_width=True)

    if show_raw:
        st.markdown("---")
        st.subheader("Raw JSON")
        st.json(payload)


if __name__ == "__main__":
    main()
