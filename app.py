import json
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import websocket
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

st.set_page_config(page_title="Deriv AI Demo Dashboard", page_icon="📈", layout="wide")

PUBLIC_WS = "wss://ws.derivws.com/websockets/v3?app_id=1089"
DEFAULT_SYMBOLS = ["R_10", "R_25", "R_50"]
FEATURES = ["ret_1", "ret_3", "ret_5", "vol_10", "sma_gap", "rsi_14"]

st.markdown("""
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(120,120,120,.08); padding: 12px; border-radius: 10px;}
.small-note {font-size: .88rem; opacity: .8;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Deriv AI — Demo Signal Dashboard")
st.caption("Market monitoring and research tool • No automatic contract purchases")

if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = []
if "journal" not in st.session_state:
    st.session_state.journal = []
if "last_signal" not in st.session_state:
    st.session_state.last_signal = {}

def ws_request(payload: dict[str, Any]) -> dict[str, Any]:
    """One public, unauthenticated Deriv WebSocket request."""
    ws = websocket.create_connection(PUBLIC_WS, timeout=25, enable_multithread=False)
    try:
        ws.send(json.dumps(payload))
        deadline = time.time() + 25
        while time.time() < deadline:
            raw = ws.recv()
            msg = json.loads(raw)
            # Ignore non-matching messages if the API sends extras.
            if msg.get("msg_type") or msg.get("error"):
                return msg
        raise TimeoutError("No Deriv response received within 25 seconds.")
    finally:
        ws.close()

@st.cache_data(ttl=20, show_spinner=False)
def get_history(symbol: str, count: int = 1500) -> pd.DataFrame:
    response = ws_request({
        "ticks_history": symbol,
        "count": count,
        "end": "latest",
        "style": "ticks"
    })
    if response.get("error"):
        raise RuntimeError(response["error"].get("message", "Deriv API error"))
    hist = response.get("history", {})
    prices = pd.to_numeric(pd.Series(hist.get("prices", [])), errors="coerce")
    times = pd.to_numeric(pd.Series(hist.get("times", [])), errors="coerce")
    df = pd.DataFrame({"epoch": times, "price": prices}).dropna()
    if df.empty:
        raise RuntimeError("Deriv returned no tick history for this symbol.")
    df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    return df.reset_index(drop=True)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def make_features(prices: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=prices.index)
    f["ret_1"] = prices.pct_change(1)
    f["ret_3"] = prices.pct_change(3)
    f["ret_5"] = prices.pct_change(5)
    f["vol_10"] = f["ret_1"].rolling(10).std()
    sma10 = prices.rolling(10).mean()
    sma30 = prices.rolling(30).mean()
    f["sma_gap"] = (sma10 - sma30) / prices
    f["rsi_14"] = rsi(prices)
    return f.replace([np.inf, -np.inf], np.nan)

def train_and_signal(df: pd.DataFrame):
    prices = df["price"].astype(float)
    feats = make_features(prices)
    # Predict whether price is higher five ticks after each feature row.
    future = prices.shift(-5)
    target = (future > prices).astype(int)
    data = feats.copy()
    data["target"] = target
    data = data.dropna()
    if len(data) < 250:
        raise RuntimeError(f"Not enough usable observations ({len(data)}). Need at least 250.")
    X = data[FEATURES]
    y = data["target"]
    split = int(len(data) * 0.8)
    if split < 100 or len(data) - split < 30:
        raise RuntimeError("Not enough data for chronological train/test split.")
    model = RandomForestClassifier(
        n_estimators=250, max_depth=5, min_samples_leaf=8,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    model.fit(X.iloc[:split], y.iloc[:split])
    test_pred = model.predict(X.iloc[split:])
    test_acc = accuracy_score(y.iloc[split:], test_pred)
    test_bal = balanced_accuracy_score(y.iloc[split:], test_pred)

    latest_features = feats.iloc[[-1]][FEATURES]
    if latest_features.isna().any(axis=None):
        raise RuntimeError("Latest feature row is incomplete; refresh data.")
    proba = model.predict_proba(latest_features)[0]
    classes = list(model.classes_)
    p_up = float(proba[classes.index(1)]) if 1 in classes else 0.0
    p_down = float(proba[classes.index(0)]) if 0 in classes else 0.0
    direction = "CALL / UP" if p_up >= p_down else "PUT / DOWN"
    confidence = max(p_up, p_down)
    # Conservative research threshold; still not a verified win probability.
    signal = direction if confidence >= 0.60 else "WAIT"
    return {
        "direction": direction, "signal": signal, "confidence": confidence,
        "p_up": p_up, "p_down": p_down, "test_accuracy": test_acc,
        "test_balanced": test_bal, "observations": len(data),
        "last_price": float(prices.iloc[-1]), "last_time": df["time"].iloc[-1]
    }

def ensure_state():
    if "risk_pct" not in st.session_state: st.session_state.risk_pct = 0.5
    if "daily_limit_pct" not in st.session_state: st.session_state.daily_limit_pct = 2.0
    if "balance" not in st.session_state: st.session_state.balance = 10.0
    if "stake" not in st.session_state: st.session_state.stake = 0.05

ensure_state()

with st.sidebar:
    st.header("⚙️ Settings")
    symbol = st.selectbox("Synthetic index", DEFAULT_SYMBOLS)
    count = st.select_slider("Historical ticks", options=[500, 1000, 1500, 2000, 3000], value=1500)
    st.divider()
    st.subheader("Risk calculator (demo only)")
    st.number_input("Demo balance", min_value=0.0, value=float(st.session_state.balance), step=1.0, key="balance")
    st.slider("Risk per idea (%)", 0.1, 2.0, key="risk_pct", step=0.1)
    st.slider("Daily stop (%)", 0.5, 5.0, key="daily_limit_pct", step=0.5)
    risk_amount = st.session_state.balance * st.session_state.risk_pct / 100
    st.metric("Suggested maximum risk amount", f"{risk_amount:.2f}")
    st.caption("If Deriv's minimum stake is above this amount, skip the trade. This dashboard does not enforce broker-side limits.")

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("▶ Start monitoring", type="primary", use_container_width=True):
        st.session_state.running = True
with col2:
    if st.button("■ Stop", use_container_width=True):
        st.session_state.running = False
with col3:
    if st.button("↻ Refresh market data", use_container_width=True):
        get_history.clear()

st.info("DEMO / SIGNAL-ONLY MODE. The app never sends a buy order or accesses your account. A displayed model score is not a guaranteed chance of winning.")

if st.session_state.running:
    try:
        with st.spinner(f"Fetching {symbol} ticks and calculating signal..."):
            market = get_history(symbol, count)
            result = train_and_signal(market)
        st.session_state.last_signal[symbol] = result
        st.session_state.history.append({
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol, "signal": result["signal"],
            "direction": result["direction"],
            "model_score": result["confidence"],
            "price": result["last_price"]
        })
        st.session_state.history = st.session_state.history[-500:]
        st.success(f"Updated {symbol} successfully.")
    except Exception as exc:
        st.error(f"Market/model error: {exc}")
else:
    st.caption("Monitoring is stopped. Press Start monitoring or Refresh market data to calculate a signal.")

result = st.session_state.last_signal.get(symbol)
if result:
    a, b, c, d = st.columns(4)
    a.metric("Latest price", f'{result["last_price"]:.5f}')
    b.metric("Signal", result["signal"])
    c.metric("Model score", f'{result["confidence"]*100:.1f}%')
    d.metric("Backtest accuracy*", f'{result["test_accuracy"]*100:.1f}%')
    st.subheader("Model direction scores")
    st.progress(min(1.0, result["p_up"]), text=f'UP score: {result["p_up"]*100:.1f}%')
    st.progress(min(1.0, result["p_down"]), text=f'DOWN score: {result["p_down"]*100:.1f}%')
    st.caption(f'*Chronological holdout accuracy on historical tick labels, not a forward guarantee. Balanced accuracy: {result["test_balanced"]*100:.1f}%. Training observations: {result["observations"]}.')
    st.caption(f'Last tick timestamp: {result["last_time"]}')
else:
    st.warning("No signal yet. Start monitoring to fetch data and calculate a research signal.")

st.divider()
st.subheader("🧾 Signal history")
hist_df = pd.DataFrame(st.session_state.history)
if not hist_df.empty:
    st.dataframe(hist_df.sort_values("time_utc", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("Download signal history CSV", hist_df.to_csv(index=False), "deriv_signal_history.csv", "text/csv")
else:
    st.write("No signals recorded in this app session.")

st.subheader("📒 Demo trade journal")
with st.form("journal_form", clear_on_submit=True):
    j1, j2, j3, j4 = st.columns(4)
    with j1: j_symbol = st.selectbox("Symbol", DEFAULT_SYMBOLS, key="j_symbol")
    with j2: j_direction = st.selectbox("Direction", ["CALL / UP", "PUT / DOWN"], key="j_direction")
    with j3: j_stake = st.number_input("Stake", min_value=0.0, value=0.05, step=0.01, key="j_stake")
    with j4: j_pl = st.number_input("Result P/L (+/-)", value=0.0, step=0.01, key="j_pl")
    submitted = st.form_submit_button("Save completed demo trade")
    if submitted:
        st.session_state.journal.append({
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": j_symbol, "direction": j_direction,
            "stake": j_stake, "profit_loss": j_pl
        })
        st.rerun()

journal_df = pd.DataFrame(st.session_state.journal)
if not journal_df.empty:
    total_pl = float(journal_df["profit_loss"].sum())
    wins = int((journal_df["profit_loss"] > 0).sum())
    losses = int((journal_df["profit_loss"] < 0).sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Journal P/L", f"{total_pl:.2f}")
    m2.metric("Wins", wins)
    m3.metric("Losses", losses)
    st.dataframe(journal_df.iloc[::-1], use_container_width=True, hide_index=True)
    st.download_button("Download trade journal CSV", journal_df.to_csv(index=False), "deriv_demo_trade_journal.csv", "text/csv")
else:
    st.write("No completed trades recorded. Add results manually after demo trades.")

with st.expander("How to interpret this app / important limitations"):
    st.markdown("""
    - This starter is a **signal research dashboard**, not an automated trading bot.
    - `R_10`, `R_25`, and `R_50` are requested from Deriv. If a symbol is unavailable for your region/account, the API will return an error.
    - The Random Forest score is an uncalibrated model score. It must not be described as a real 90% win probability.
    - The historical holdout metric is not a full options backtest. It does not model proposal prices, payout, spread, execution delay, contract settlement, or fees.
    - Streamlit Community Cloud may sleep/restart apps. Session-only history and journal entries can be lost on restart; download CSVs to keep records.
    - No API token is needed for this public market-data demo. Never put a Deriv password or API token in GitHub source code.
    - Real trading is intentionally not implemented. Before any live deployment, add authenticated OTP WebSocket handling, contract proposals, execution safeguards, account-mode verification, persistent storage, and independent testing.
    """)
