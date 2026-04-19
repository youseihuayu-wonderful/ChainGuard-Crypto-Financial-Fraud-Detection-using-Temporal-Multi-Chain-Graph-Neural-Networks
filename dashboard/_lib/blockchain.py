"""
Real-Time Blockchain Scanner — Etherscan API Integration
WHO: Investigators, Operations
WHAT: Look up real Ethereum addresses and transactions via Etherscan public API.

NOTE: Uses the FREE Etherscan API (no API key needed for basic queries).
Risk assessment uses the rule-based engine, not GNN.
"""

import streamlit as st
import pandas as pd
import requests
import re
import time

from _lib.i18n import t

ETHERSCAN_BASE = "https://api.etherscan.io/api"

# Regex patterns for Ethereum addresses and tx hashes
ETH_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
ETH_TX_HASH_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")


def _is_eth_address(s):
    return bool(ETH_ADDRESS_RE.match(s.strip()))


def _is_eth_tx_hash(s):
    return bool(ETH_TX_HASH_RE.match(s.strip()))


def _fetch_balance(address):
    """Fetch ETH balance for an address via Etherscan API."""
    try:
        resp = requests.get(
            ETHERSCAN_BASE,
            params={
                "module": "account",
                "action": "balance",
                "address": address,
                "tag": "latest",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "1":
            wei = int(data["result"])
            return wei / 1e18  # Convert Wei to ETH
        else:
            return None
    except Exception:
        return None


def _fetch_transactions(address, count=10):
    """Fetch last N transactions for an address via Etherscan API."""
    try:
        resp = requests.get(
            ETHERSCAN_BASE,
            params={
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": 0,
                "endblock": 99999999,
                "page": 1,
                "offset": count,
                "sort": "desc",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "1" and isinstance(data.get("result"), list):
            return data["result"]
        else:
            return []
    except Exception:
        return []


def _assess_risk(transactions):
    """
    Rule-based risk assessment from transaction patterns.
    Returns (risk_score, factors) similar to scanner.py engine.
    """
    if not transactions:
        return 0.1, [("No transaction data available", 0.0, "LOW")]

    risk = 0.08
    factors = []

    # Analyze patterns
    n_tx = len(transactions)
    values_eth = []
    unique_targets = set()
    has_contract_calls = False
    has_failed = False
    total_gas_used = 0

    for tx in transactions:
        val_wei = int(tx.get("value", "0"))
        val_eth = val_wei / 1e18
        values_eth.append(val_eth)

        to_addr = tx.get("to", "")
        if to_addr:
            unique_targets.add(to_addr.lower())

        if tx.get("input", "0x") != "0x":
            has_contract_calls = True

        if tx.get("isError", "0") == "1":
            has_failed = True

        total_gas_used += int(tx.get("gasUsed", "0"))

    total_value = sum(values_eth)
    avg_value = total_value / max(n_tx, 1)
    n_targets = len(unique_targets)

    # Rule 1: High total volume
    if total_value > 100:
        bonus = 0.20
        risk += bonus
        factors.append((f"High total volume ({total_value:.2f} ETH in last {n_tx} tx)", bonus, "HIGH"))
    elif total_value > 10:
        bonus = 0.10
        risk += bonus
        factors.append((f"Moderate volume ({total_value:.2f} ETH)", bonus, "MEDIUM"))

    # Rule 2: Fan-out pattern (many unique recipients)
    if n_targets > 7:
        bonus = 0.15
        risk += bonus
        factors.append((f"Fan-out: {n_targets} unique recipients", bonus, "HIGH"))
    elif n_targets > 4:
        bonus = 0.08
        risk += bonus
        factors.append((f"Multiple recipients: {n_targets}", bonus, "MEDIUM"))

    # Rule 3: Contract interactions (potential mixing/DeFi)
    if has_contract_calls:
        bonus = 0.10
        risk += bonus
        factors.append(("Smart contract interactions detected", bonus, "MEDIUM"))

    # Rule 4: Failed transactions (potential probing)
    if has_failed:
        bonus = 0.08
        risk += bonus
        factors.append(("Failed transactions detected", bonus, "MEDIUM"))

    # Rule 5: Small uniform amounts (structuring)
    if n_tx >= 3 and avg_value < 0.5 and total_value > 1:
        bonus = 0.12
        risk += bonus
        factors.append(("Possible structuring: small uniform amounts", bonus, "MEDIUM"))

    # Rule 6: High gas usage (complex operations)
    avg_gas = total_gas_used / max(n_tx, 1)
    if avg_gas > 100000:
        bonus = 0.06
        risk += bonus
        factors.append(("High gas usage (complex operations)", bonus, "LOW"))

    risk = min(0.98, max(0.02, risk))

    if not factors:
        factors.append(("No significant risk factors detected", 0.0, "LOW"))

    return risk, factors


def render(DATA, navigate_to):
    """Render the Real-Time Blockchain Scanner page."""
    st.markdown(f"# :link: {t('blockchain_title')}")
    st.markdown(t("blockchain_subtitle"))
    st.caption(t("blockchain_caption"))

    st.markdown("---")

    # Input section
    col_input, col_info = st.columns([2, 1])

    with col_input:
        st.markdown(f"### {t('enter_address_or_hash')}")
        with st.form("blockchain_form"):
            query = st.text_input(
                t("eth_address_or_hash"),
                placeholder="0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18",
                key="blockchain_query",
            )
            lookup_btn = st.form_submit_button(t("search"), type="primary", use_container_width=True)

    with col_info:
        st.markdown(f"### {t('supported_lookups')}")
        st.markdown(
            '<div class="glass-card">'
            f'<p style="color:#E5E7EB; margin:0"><strong style="color:#00D4AA">{t("address_desc")}</strong> '
            f'{t("address_detail")}</p>'
            '<br>'
            f'<p style="color:#E5E7EB; margin:0"><strong style="color:#3B82F6">{t("tx_hash_desc")}</strong> '
            f'{t("tx_hash_detail")}</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if lookup_btn and query:
        query = query.strip()

        if _is_eth_address(query):
            _render_address_lookup(query)
        elif _is_eth_tx_hash(query):
            _render_tx_lookup(query)
        else:
            st.error(t("invalid_input"))

    elif not lookup_btn:
        # Show example addresses
        st.markdown(f"### {t('example_addresses')}")
        st.markdown(t("try_example_addresses"))

        examples = [
            ("Ethereum Foundation", "0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe"),
            ("Vitalik Buterin", "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"),
            ("Binance Hot Wallet", "0x28C6c06298d514Db089934071355E5743bf21d60"),
        ]

        for name, addr in examples:
            st.markdown(
                f'<div class="stat-row">'
                f'<span style="color:#E5E7EB">{name}</span>'
                f'<code style="color:#00D4AA; font-size:0.8rem">{addr[:10]}...{addr[-6:]}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.info(t("search_prompt"))


def _render_address_lookup(address):
    """Render results for an Ethereum address lookup."""
    st.markdown(f"### {t('address_label')}: `{address[:10]}...{address[-6:]}`")

    with st.spinner(t("fetching_etherscan")):
        balance = _fetch_balance(address)
        transactions = _fetch_transactions(address, count=10)

    if balance is None and not transactions:
        st.warning(t("fetch_failed"))
        return

    # Balance display
    m1, m2, m3 = st.columns(3)
    with m1:
        if balance is not None:
            st.metric(t("eth_balance"), f"{balance:.4f} ETH")
        else:
            st.metric(t("eth_balance"), "N/A")

    with m2:
        st.metric(t("transactions_found"), len(transactions))

    with m3:
        if transactions:
            total_val = sum(int(tx.get("value", "0")) / 1e18 for tx in transactions)
            st.metric(t("total_value_last10"), f"{total_val:.4f} ETH")
        else:
            st.metric(t("total_value_last10"), "N/A")

    # Transaction list
    if transactions:
        st.markdown(f"### {t('recent_transactions')}")

        tx_rows = []
        for tx in transactions:
            val_eth = int(tx.get("value", "0")) / 1e18
            ts = int(tx.get("timeStamp", "0"))
            time_str = time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts)) if ts else "N/A"
            tx_hash = tx.get("hash", "N/A")
            from_addr = tx.get("from", "N/A")
            to_addr = tx.get("to", "N/A")
            is_error = tx.get("isError", "0") == "1"
            gas_used = int(tx.get("gasUsed", "0"))

            direction = "OUT" if from_addr.lower() == address.lower() else "IN"

            tx_rows.append({
                t("time_col"): time_str,
                t("direction_col"): direction,
                t("value_eth_col"): f"{val_eth:.6f}",
                t("from_label"): f"{from_addr[:8]}...{from_addr[-4:]}" if len(from_addr) > 12 else from_addr,
                t("to_label"): f"{to_addr[:8]}...{to_addr[-4:]}" if len(to_addr) > 12 else to_addr,
                t("gas_used_col"): f"{gas_used:,}",
                t("status_col"): t("failed") if is_error else t("success"),
                "Tx Hash": f"{tx_hash[:10]}...{tx_hash[-6:]}" if len(tx_hash) > 16 else tx_hash,
            })

        df = pd.DataFrame(tx_rows)
        st.dataframe(df, width="stretch", hide_index=True, height=400)

        # Risk Assessment
        st.markdown("---")
        st.markdown(f"### {t('risk_assessment')}")
        st.caption(t("risk_assessment_caption"))

        risk_score, factors = _assess_risk(transactions)
        level = "HIGH" if risk_score > 0.7 else ("MEDIUM" if risk_score > 0.4 else "LOW")
        color = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#00D4AA"}[level]
        css_class = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}[level]

        level_text = {"HIGH": t("high_risk"), "MEDIUM": t("medium_risk"), "LOW": t("low_risk")}[level]
        st.markdown(
            f'<div class="{css_class}" style="text-align:center; padding:20px">'
            f'<h2 style="color:{color}; margin:0">{level_text}</h2>'
            f'<h1 style="color:{color}; margin:0; font-size:3rem">{risk_score:.0%}</h1>'
            f'<p style="color:#9CA3AF; margin:4px 0 0 0; font-size:0.8rem">'
            f'{t("rule_based_engine")} | {len(factors)} {t("factors_analyzed")}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(f"#### {t('risk_factors')}")
        for name, weight, sev in sorted(factors, key=lambda x: -x[1]):
            sc_color = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#00D4AA"}[sev]
            pct = f"+{weight:.0%}" if weight > 0 else "0%"
            st.markdown(
                f'<div class="stat-row">'
                f'<span style="color:#E5E7EB; font-size:0.9rem">{name}</span>'
                f'<span style="color:{sc_color}; font-weight:700; font-family:JetBrains Mono,monospace">{pct}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info(t("no_tx_found"))


def _render_tx_lookup(tx_hash):
    """Render results for a transaction hash lookup."""
    st.markdown(f"### {t('transaction_label')}: `{tx_hash[:10]}...{tx_hash[-6:]}`")

    with st.spinner(t("fetching_tx")):
        try:
            resp = requests.get(
                ETHERSCAN_BASE,
                params={
                    "module": "proxy",
                    "action": "eth_getTransactionByHash",
                    "txhash": tx_hash,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            tx = data.get("result")
        except Exception:
            tx = None

    if not tx:
        st.warning(t("tx_fetch_failed"))
        return

    # Display transaction details
    from_addr = tx.get("from", "N/A")
    to_addr = tx.get("to", "N/A") or "Contract Creation"
    value_hex = tx.get("value", "0x0")
    value_eth = int(value_hex, 16) / 1e18 if value_hex else 0
    gas_hex = tx.get("gas", "0x0")
    gas = int(gas_hex, 16) if gas_hex else 0
    block_hex = tx.get("blockNumber", "0x0")
    block = int(block_hex, 16) if block_hex else 0
    nonce_hex = tx.get("nonce", "0x0")
    nonce = int(nonce_hex, 16) if nonce_hex else 0

    m1, m2 = st.columns(2)
    with m1:
        st.metric(t("value"), f"{value_eth:.6f} ETH")
        st.metric(t("block"), f"{block:,}")
    with m2:
        st.metric(t("gas_limit"), f"{gas:,}")
        st.metric(t("nonce"), nonce)

    st.markdown(f"#### {t('addresses')}")
    st.markdown(
        f'<div class="stat-row">'
        f'<span style="color:#9CA3AF">{t("from_label")}</span>'
        f'<code style="color:#00D4AA">{from_addr}</code>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="stat-row">'
        f'<span style="color:#9CA3AF">{t("to_label")}</span>'
        f'<code style="color:#3B82F6">{to_addr}</code>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Check if it's a contract call
    input_data = tx.get("input", "0x")
    if input_data and input_data != "0x":
        st.markdown(f"#### {t('contract_interaction')}")
        st.markdown(
            '<div class="risk-medium">'
            f'<strong style="color:#F59E0B">{t("smart_contract_detected")}</strong><br>'
            f'<span style="color:#E5E7EB">{t("input_data_length").format(length=len(input_data))}</span><br>'
            f'<span style="color:#9CA3AF">{t("method_id").format(method=input_data[:10])}</span>'
            '</div>',
            unsafe_allow_html=True,
        )
