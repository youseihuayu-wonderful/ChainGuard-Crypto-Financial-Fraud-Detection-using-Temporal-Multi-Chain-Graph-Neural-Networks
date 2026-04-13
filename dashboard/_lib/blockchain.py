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
    st.markdown("# :link: Real-Time Blockchain Scanner")
    st.markdown("Look up real Ethereum addresses and transaction hashes via the Etherscan public API.")
    st.caption("Live data from Etherscan API. Risk assessment uses rule-based engine, not GNN.")

    st.markdown("---")

    # Input section
    col_input, col_info = st.columns([2, 1])

    with col_input:
        st.markdown("### Enter Address or Transaction Hash")
        query = st.text_input(
            "Ethereum address (0x...) or transaction hash",
            placeholder="0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18",
            key="blockchain_query",
        )

        lookup_btn = st.button("Search", type="primary", use_container_width=True, key="blockchain_search")

    with col_info:
        st.markdown("### Supported Lookups")
        st.markdown(
            '<div class="glass-card">'
            '<p style="color:#E5E7EB; margin:0"><strong style="color:#00D4AA">Address</strong> '
            '(0x + 40 hex chars)<br>Shows balance and recent transactions</p>'
            '<br>'
            '<p style="color:#E5E7EB; margin:0"><strong style="color:#3B82F6">Tx Hash</strong> '
            '(0x + 64 hex chars)<br>Shows transaction details</p>'
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
            st.error(
                "Invalid input. Please enter a valid Ethereum address (0x + 40 hex characters) "
                "or transaction hash (0x + 64 hex characters)."
            )

    elif not lookup_btn:
        # Show example addresses
        st.markdown("### Example Addresses")
        st.markdown("Try these well-known Ethereum addresses:")

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
        st.info("Enter an address or transaction hash above and click **Search** to begin.")


def _render_address_lookup(address):
    """Render results for an Ethereum address lookup."""
    st.markdown(f"### Address: `{address[:10]}...{address[-6:]}`")

    with st.spinner("Fetching data from Etherscan..."):
        balance = _fetch_balance(address)
        transactions = _fetch_transactions(address, count=10)

    if balance is None and not transactions:
        st.warning(
            "Could not fetch data from Etherscan. This may be due to rate limiting "
            "(max 5 requests/sec without API key) or network issues. Please try again in a moment."
        )
        return

    # Balance display
    m1, m2, m3 = st.columns(3)
    with m1:
        if balance is not None:
            st.metric("ETH Balance", f"{balance:.4f} ETH")
        else:
            st.metric("ETH Balance", "N/A")

    with m2:
        st.metric("Transactions Found", len(transactions))

    with m3:
        if transactions:
            total_val = sum(int(tx.get("value", "0")) / 1e18 for tx in transactions)
            st.metric("Total Value (Last 10)", f"{total_val:.4f} ETH")
        else:
            st.metric("Total Value (Last 10)", "N/A")

    # Transaction list
    if transactions:
        st.markdown("### Recent Transactions")

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
                "Time": time_str,
                "Direction": direction,
                "Value (ETH)": f"{val_eth:.6f}",
                "From": f"{from_addr[:8]}...{from_addr[-4:]}" if len(from_addr) > 12 else from_addr,
                "To": f"{to_addr[:8]}...{to_addr[-4:]}" if len(to_addr) > 12 else to_addr,
                "Gas Used": f"{gas_used:,}",
                "Status": "Failed" if is_error else "Success",
                "Tx Hash": f"{tx_hash[:10]}...{tx_hash[-6:]}" if len(tx_hash) > 16 else tx_hash,
            })

        df = pd.DataFrame(tx_rows)
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)

        # Risk Assessment
        st.markdown("---")
        st.markdown("### Risk Assessment")
        st.caption("Rule-based assessment from transaction patterns. Not a GNN prediction.")

        risk_score, factors = _assess_risk(transactions)
        level = "HIGH" if risk_score > 0.7 else ("MEDIUM" if risk_score > 0.4 else "LOW")
        color = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#00D4AA"}[level]
        css_class = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}[level]

        st.markdown(
            f'<div class="{css_class}" style="text-align:center; padding:20px">'
            f'<h2 style="color:{color}; margin:0">{level} RISK</h2>'
            f'<h1 style="color:{color}; margin:0; font-size:3rem">{risk_score:.0%}</h1>'
            f'<p style="color:#9CA3AF; margin:4px 0 0 0; font-size:0.8rem">'
            f'Rule-based engine | {len(factors)} factor{"s" if len(factors) != 1 else ""} analyzed</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Risk Factors")
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
        st.info("No transactions found for this address, or the address has no activity on Ethereum mainnet.")


def _render_tx_lookup(tx_hash):
    """Render results for a transaction hash lookup."""
    st.markdown(f"### Transaction: `{tx_hash[:10]}...{tx_hash[-6:]}`")

    with st.spinner("Fetching transaction from Etherscan..."):
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
        st.warning(
            "Could not fetch transaction data. This may be due to Etherscan rate limiting "
            "(max 5 requests/sec without API key) or the transaction hash may not exist. "
            "Please try again in a moment."
        )
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
        st.metric("Value", f"{value_eth:.6f} ETH")
        st.metric("Block", f"{block:,}")
    with m2:
        st.metric("Gas Limit", f"{gas:,}")
        st.metric("Nonce", nonce)

    st.markdown("#### Addresses")
    st.markdown(
        f'<div class="stat-row">'
        f'<span style="color:#9CA3AF">From</span>'
        f'<code style="color:#00D4AA">{from_addr}</code>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="stat-row">'
        f'<span style="color:#9CA3AF">To</span>'
        f'<code style="color:#3B82F6">{to_addr}</code>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Check if it's a contract call
    input_data = tx.get("input", "0x")
    if input_data and input_data != "0x":
        st.markdown("#### Contract Interaction")
        st.markdown(
            '<div class="risk-medium">'
            '<strong style="color:#F59E0B">Smart Contract Call Detected</strong><br>'
            f'<span style="color:#E5E7EB">Input data length: {len(input_data)} characters</span><br>'
            f'<span style="color:#9CA3AF">Method ID: {input_data[:10]}</span>'
            '</div>',
            unsafe_allow_html=True,
        )
