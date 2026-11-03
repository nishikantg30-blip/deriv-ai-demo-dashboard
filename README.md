# Deriv AI Demo Signal Dashboard

A beginner-friendly Streamlit dashboard for Deriv Synthetic Indices. It fetches public historical ticks, builds a small Random Forest model, displays directional research signals, and provides session-only signal history and a manual demo-trade journal.

## Safety and scope

- Demo / signal-only starter. It does **not** place trades.
- No Deriv account token is required for public market data.
- Model confidence is an uncalibrated score, not a verified probability of winning.
- The displayed holdout accuracy is not an options profitability backtest.
- Session data can be lost when the app restarts. Download CSVs regularly.
- Never commit API tokens, passwords, or other secrets to GitHub.
- No claim of 90% accuracy or guaranteed profit is made.

## Files

- `app.py` — Streamlit dashboard and public Deriv market-data client
- `requirements.txt` — Python dependencies
- `README.md` — deployment instructions

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create/sign in to GitHub.
2. Create a new repository, e.g. `deriv-ai-demo-dashboard`.
3. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Click **Create app**.
6. Choose the repository, branch (`main`), and main file path (`app.py`).
7. Click **Deploy**. No secrets are required for this public-data demo.
8. Open the generated `*.streamlit.app` URL on desktop or mobile.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Troubleshooting

- If Deriv returns HTTP 520 or a WebSocket connection error, wait and retry; check whether your network blocks WebSockets. Community Cloud may have different network behavior from Colab.
- If a symbol is unavailable, try another active synthetic symbol listed by Deriv.
- If the model says there are too few observations, increase the Historical ticks setting.
- If app startup fails, open the Streamlit app logs and check dependency installation errors.

## Before considering real-money trading

This project is not a live-trading system. Real trading requires a separate, audited implementation using Deriv's current authenticated OTP WebSocket flow, account-mode checks, contract proposals, execution and settlement monitoring, strict risk limits, and persistent trade records. Do not add a real token to this starter app.
