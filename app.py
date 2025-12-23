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


def _stage_portfolio_key(stage: str) -> str:
    if stage in ("Market State", "Proposed Trades"):
        return "portfolio_start"
    return "portfolio_end"


def _stage_value(client: Dict[str, Any], key: str, stage: str) -> float:
    if stage in ("Market State", "Proposed Trades"):
        start = client.get("start", {})
        return float(start.get(key, 0.0) or 0.0)
    return float(client.get(f"{key}_end", 0.0) or 0.0)


def _instrument_price_series(metrics: List[Dict[str, Any]], instrument_idx: int) -> List[float]:
    prices: List[float] = []
    price = 100.0
    for day in metrics:
        details = day.get("details", {})
        returns = details.get("real_returns")
        if returns is None:
            continue
        if instrument_idx >= len(returns):
            continue
        price += float(returns[instrument_idx])
        prices.append(price)
    return prices


def _client_rows(cms: Dict[str, Any], cm_name: str) -> List[Dict[str, Any]]:
    cm_data = cms.get(cm_name, {})
    return cm_data.get("clients", [])


def _bar_html(label: str, value: float, max_value: float, color: str) -> str:
    pct = 0.0 if max_value <= 0 else min(100.0, (value / max_value) * 100.0)
    return (
        f'<div class="bar-wrap">'
        f'<div class="bar-fill" style="height:{pct:.2f}%; background:{color};"></div>'
        f"</div>"
        f'<div class="bar-label">{label}<br>{value:,.0f}</div>'
    )


def _portfolio_cells_html(
    portfolio: List[float],
    trades: Dict[int, Dict[str, Any]],
    stage: str,
) -> str:
    cells = []
    for idx, value in enumerate(portfolio):
        cls = "cell"
        text = f"{value:+.0f}"
        trade = trades.get(idx)
        if trade and stage == "Proposed Trades":
            cls += " cell-proposed"
            text = f"{trade['amount']:+.0f}"
        elif trade and stage == "Accepted Trades":
            if trade.get("accepted"):
                cls += " cell-accepted"
            else:
                cls += " cell-rejected"
            text = f"{trade['amount']:+.0f}"
        cells.append(f'<div class="{cls}">{text}</div>')
    return "".join(cells)


def _build_price_matrix(metrics: List[Dict[str, Any]], base_price: float = 100.0) -> List[List[float]]:
    if not metrics:
        return []
    first_returns = metrics[0].get("details", {}).get("real_returns")
    if first_returns is None:
        return []
    n_instruments = len(first_returns)
    prices = [float(base_price)] * n_instruments
    matrix: List[List[float]] = []
    for day in metrics:
        returns = day.get("details", {}).get("real_returns")
        if returns is None or len(returns) != n_instruments:
            return []
        prices = [p + float(r) for p, r in zip(prices, returns)]
        matrix.append(prices[:])
    return matrix


def _client_stats(client: Dict[str, Any], stage: str) -> Dict[str, float]:
    wealth = _stage_value(client, "wealth", stage)
    collateral = _stage_value(client, "collateral", stage)
    margin = float(client.get("margin", 0.0) or 0.0)
    shortfall = float(client.get("shortfall", 0.0) or 0.0)
    trades = client.get("trades", [])
    trade_risk = sum(abs(float(t.get("amount", 0.0))) for t in trades)
    accepted_trades = [t for t in trades if t.get("accepted")]
    accepted_risk = sum(abs(float(t.get("amount", 0.0))) for t in accepted_trades)
    margin_call = client.get("margin_call", {})
    margin_call_amount = float(margin_call.get("amount", 0.0) or 0.0) if margin_call.get("called") else 0.0
    return {
        "wealth": wealth,
        "collateral": collateral,
        "current_risk": margin,
        "potential_risk": margin + max(shortfall, 0.0),
        "trade_risk": trade_risk,
        "trades": float(len(trades)),
        "accepted_trades": float(len(accepted_trades)),
        "accepted_trade_risk": accepted_risk,
        "margin_call": margin_call_amount,
    }


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
    tabs = st.tabs(["Playback", "Charts"])

    with tabs[0]:
        st.subheader("Portfolio Rows")

        cm_names = list(cms.keys())
        if not cm_names:
            st.warning("No CM data available for this day.")
            return

        selected_cm = st.selectbox("Clearing Member", cm_names, index=0)
        client_rows = _client_rows(cms, selected_cm)

        if not client_rows:
            st.warning("No client details available. Ensure output.include_details is true.")
            return

        num_clients = len(client_rows)
        page_size = st.slider("Clients per page", min_value=1, max_value=min(20, num_clients), value=min(10, num_clients))
        total_pages = max(1, (num_clients - 1) // page_size + 1)
        if total_pages == 1:
            page = 1
        else:
            page = st.slider("Page", min_value=1, max_value=total_pages, value=1)

        start_idx = (page - 1) * page_size
        end_idx = min(num_clients, start_idx + page_size)
        clients_page = client_rows[start_idx:end_idx]

        instrument_count = len(clients_page[0].get(_stage_portfolio_key(stage), []) or [])
        instrument_idx = st.slider(
            "Instrument for price chart",
            min_value=0,
            max_value=max(0, instrument_count - 1),
            value=0,
        )

        prices = _instrument_price_series(metrics, instrument_idx)
        if prices:
            st.line_chart(prices, height=220)
        else:
            st.info("Price chart unavailable (real_returns not included). Enable output.include_returns.")

        max_wealth = max(_stage_value(c, "wealth", stage) for c in clients_page)
        max_collateral = max(_stage_value(c, "collateral", stage) for c in clients_page)
        max_risk = max(float(c.get("margin", 0.0) or 0.0) for c in clients_page)

        st.markdown(
            """
            <style>
            .row-wrap { display: flex; align-items: center; gap: 16px; margin-bottom: 12px; }
            .bars { display: flex; gap: 8px; align-items: flex-end; }
            .bar-wrap { width: 14px; height: 70px; background: #ececec; border-radius: 4px; position: relative; }
            .bar-fill { width: 100%; position: absolute; bottom: 0; border-radius: 4px; }
            .bar-label { font-size: 10px; text-align: center; width: 56px; margin-top: 2px; color: #111; }
            .portfolio { display: grid; grid-auto-flow: column; gap: 6px; }
            .cell { width: 56px; height: 40px; border: 1px solid #ccc; display: flex; align-items: center; justify-content: center; font-size: 12px; background: #fff; color: #111; }
            .cell-proposed { background: #fff3b0; border-color: #e2b203; }
            .cell-accepted { background: #c9f5c9; border-color: #2e7d32; }
            .cell-rejected { background: #f8d2d2; border-color: #c62828; }
            .stats-grid { display: grid; grid-template-columns: 120px 120px; gap: 6px 12px; align-items: center; }
            .stats-label { color: #333; font-size: 12px; }
            .stats-value { background: #f4f4f4; border: 1px solid #ddd; padding: 4px 6px; border-radius: 4px; text-align: right; font-size: 12px; color: #111; }
            .portfolio-block { display: grid; gap: 6px; }
            .inst-row, .price-row, .position-row { display: grid; grid-auto-flow: column; gap: 6px; }
            .inst-label { width: 56px; text-align: center; font-size: 10px; color: #555; }
            .price-cell { width: 56px; height: 24px; border: 1px solid #ddd; display: flex; align-items: center; justify-content: center; font-size: 11px; background: #fafafa; color: #111; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        portfolio_key = _stage_portfolio_key(stage)

        client_ids = [c.get("client_id") for c in client_rows]
        selected_client_id = st.selectbox("Client", client_ids, index=0)
        selected_client = next((c for c in client_rows if c.get("client_id") == selected_client_id), client_rows[0])
        stats = _client_stats(selected_client, stage)

        price_matrix = _build_price_matrix(metrics)
        price_units = price_matrix[day_index] if price_matrix else []
        portfolio = selected_client.get(portfolio_key) or []
        trades = {t["instrument"]: t for t in selected_client.get("trades", [])}
        inst_labels = "".join(f'<div class="inst-label">inst_{i+1}</div>' for i in range(len(portfolio)))
        price_cells = "".join(
            f'<div class="price-cell">{price_units[i]:.1f}</div>' if i < len(price_units) else '<div class="price-cell">n/a</div>'
            for i in range(len(portfolio))
        )
        position_cells = _portfolio_cells_html(portfolio, trades, stage)

        stats_html = (
            '<div class="stats-grid">'
            f'<div class="stats-label">wealth</div><div class="stats-value">{stats["wealth"]:,.0f}</div>'
            f'<div class="stats-label">collateral</div><div class="stats-value">{stats["collateral"]:,.0f}</div>'
            f'<div class="stats-label">current risk</div><div class="stats-value">{stats["current_risk"]:,.0f}</div>'
            f'<div class="stats-label">potential risk</div><div class="stats-value">{stats["potential_risk"]:,.0f}</div>'
            f'<div class="stats-label">trade risk</div><div class="stats-value">{stats["trade_risk"]:,.0f}</div>'
            f'<div class="stats-label">trades</div><div class="stats-value">{stats["trades"]:,.0f}</div>'
            f'<div class="stats-label">accepted trades</div><div class="stats-value">{stats["accepted_trades"]:,.0f}</div>'
            f'<div class="stats-label">margin calls</div><div class="stats-value">{stats["margin_call"]:,.0f}</div>'
            "</div>"
        )
        portfolio_html = (
            '<div class="portfolio-block">'
            f'<div class="inst-row">{inst_labels}</div>'
            f'<div class="price-row">{price_cells}</div>'
            f'<div class="position-row">{position_cells}</div>'
            "</div>"
        )

        st.markdown("### Selected Client")
        cols = st.columns([2, 6])
        cols[0].markdown(stats_html, unsafe_allow_html=True)
        cols[1].markdown(portfolio_html, unsafe_allow_html=True)

        st.markdown("### All Clients (Page)")
        for client in clients_page:
            portfolio = client.get(portfolio_key) or []
            trades = {t["instrument"]: t for t in client.get("trades", [])}
            bars_html = (
                '<div class="bars">'
                + _bar_html("W", _stage_value(client, "wealth", stage), max_wealth, "#6cc04a")
                + _bar_html("C", _stage_value(client, "collateral", stage), max_collateral, "#3ba4c7")
                + _bar_html("R", float(client.get("margin", 0.0) or 0.0), max_risk, "#ef5350")
                + "</div>"
            )
            portfolio_html = f'<div class="portfolio">{_portfolio_cells_html(portfolio, trades, stage)}</div>'
            st.markdown(f'<div class="row-wrap">{bars_html}{portfolio_html}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Stage Context")
        if stage == "Market State":
            st.write("Market index:", day_data.get("market_index"))
            st.write("Market state:", details.get("market_state"))
        elif stage == "Margin Calls":
            st.write("Margin calls reflected in risk/shortfall bars.")
        elif stage == "PnL & Returns":
            st.write("PnL impacts wealth and collateral.")

    with tabs[1]:
        st.subheader("Price Charts")
        price_matrix = _build_price_matrix(metrics)
        if not price_matrix:
            st.info("Price charts unavailable (real_returns not included). Enable output.include_returns.")
        else:
            instrument_count = len(price_matrix[0])
            options = list(range(instrument_count))
            selected_insts = st.multiselect("Instruments", options, default=options[: min(3, instrument_count)])
            if selected_insts:
                data = {
                    f"inst_{i+1}": [row[i] for row in price_matrix]
                    for i in selected_insts
                }
                st.line_chart(pd.DataFrame(data))

        st.subheader("Client Histories")
        cm_names = list(cms.keys())
        if cm_names:
            cm_hist = st.selectbox("CM for client history", cm_names, index=0, key="cm_hist")
            clients = _client_rows(cms, cm_hist)
            if clients:
                client_ids = [c.get("client_id") for c in clients]
                client_id = st.selectbox("Client ID", client_ids, index=0, key="client_hist")
                wealth_series = []
                collateral_series = []
                margin_series = []
                for day in metrics:
                    cm_day = day.get("cms", {}).get(cm_hist, {})
                    client_day = next((c for c in cm_day.get("clients", []) if c.get("client_id") == client_id), None)
                    if not client_day:
                        continue
                    wealth_series.append(client_day.get("wealth_end", 0.0))
                    collateral_series.append(client_day.get("collateral_end", 0.0))
                    margin_series.append(client_day.get("margin", 0.0))
                if wealth_series:
                    st.line_chart(
                        pd.DataFrame(
                            {
                                "wealth": wealth_series,
                                "collateral": collateral_series,
                                "margin": margin_series,
                            }
                        )
                    )
                else:
                    st.info("Client history unavailable. Ensure output.include_details is true.")

        st.subheader("CM Histories")
        if cm_names:
            cm_hist2 = st.selectbox("CM for CM history", cm_names, index=0, key="cm_hist2")
            cm_funds = []
            total_coll = []
            avg_margin = []
            for day in metrics:
                cm_day = day.get("cms", {}).get(cm_hist2, {})
                cm_funds.append(cm_day.get("cm_funds", 0.0))
                total_coll.append(cm_day.get("total_client_collateral", 0.0))
                avg_margin.append(cm_day.get("avg_client_margin", 0.0))
            st.line_chart(
                pd.DataFrame(
                    {
                        "cm_funds": cm_funds,
                        "total_collateral": total_coll,
                        "avg_margin": avg_margin,
                    }
                )
            )

    if show_raw:
        st.markdown("---")
        st.subheader("Raw JSON")
        st.json(payload)


if __name__ == "__main__":
    main()
