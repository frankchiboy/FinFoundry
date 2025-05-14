# FinFoundry

📈 A Streamlit-powered AI dashboard for stock analysis, prediction, and insight generation — built with machine learning and technical indicators.

## 🔍 Features

- Fetch real-time stock data via `yfinance`
- Calculate technical indicators: SMA, RSI (with `ta`)
- Predict next-day closing price using `RandomForestRegressor`
- Visualize price trends, indicators, and model outputs
- Ready to integrate with LLMs (e.g. Finsight) for intelligent explanations

## 🧠 Powered By

- `Streamlit` – interactive web UI
- `yfinance` – stock data API
- `ta` – technical analysis library
- `scikit-learn` – machine learning
- `joblib` – model serialization
- (Optional) `openai` or `finsight API` – LLM-powered insight

## 🚀 Getting Started

```bash
git clone https://github.com/yourname/FinFoundry.git
cd FinFoundry
pip install -r requirements.txt
streamlit run stock_dashboard.py