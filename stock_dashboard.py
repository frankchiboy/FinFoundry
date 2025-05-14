import yfinance as yf
import streamlit as st
import pandas as pd
import ta

lang = st.selectbox("🌐 Language 語言", ["English", "中文"], index=0)
is_en = (lang == "English")

# UI
st.title("📈 Stock Strategy Prediction Dashboard" if is_en else "📈 投資策略預測分析儀表板")
ticker = st.text_input("Ticker Symbol" if is_en else "輸入股票代碼", "AAPL")
start = st.date_input("Start Date" if is_en else "起始日期", pd.to_datetime("2022-01-01"))
end = st.date_input("End Date" if is_en else "結束日期", pd.to_datetime("today"))

# 抓資料
df = yf.download(ticker, start=start, end=end, auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.rename_axis("Date").reset_index()
df.set_index("Date", inplace=True)

# 技術指標
df["SMA20"] = df["Close"].rolling(window=20).mean()
# RSI 維度修正
df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"].squeeze()).rsi().squeeze()

# 顯示
st.line_chart(df[["Close", "SMA20"]])
st.line_chart(df[["RSI"]])

st.dataframe(df.tail())

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# 建立預測目標：明日收盤價
df["Target"] = df["Close"].shift(-1)
features = ["Close", "SMA20", "RSI"]
df = df.dropna(subset=features + ["Target"])

# 拆分資料
X = df[features]
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# 訓練模型
model = RandomForestRegressor()
model.fit(X_train, y_train)
joblib.dump(model, "stock_model.pkl")

# 載入模型與預測
model = joblib.load("stock_model.pkl")
latest = df[features].iloc[-1:]
prediction = model.predict(latest)[0]

st.metric("📊 Predicted Close Price (Next Day)" if is_en else "📊 預測明日收盤價", f"${prediction:.2f}")

# === Finsight LLM 解釋區塊 ===
import requests

# 自然語言分析輸入
st.subheader("🤖 LLM Insight" if is_en else "🤖 LLM 解釋")
question = st.text_area(
    "Ask the model about this prediction" if is_en else "請問模型與這次預測有關的問題",
    "Please explain the reasoning behind today's prediction."
)

if st.button("Generate Explanation" if is_en else "產生解釋"):
    base_url = "http://localhost:11434/api/generate"
    try:
        prompt = f"""
        Stock Ticker: {ticker}
        Current Close: {df['Close'].iloc[-1]:.2f}
        SMA20: {df['SMA20'].iloc[-1]:.2f}
        RSI: {df['RSI'].iloc[-1]:.2f}
        Predicted Close Tomorrow: {prediction:.2f}
        Question: {question}
        """
        res = requests.post(base_url, json={
            "model": "llama4:latest",
            "prompt": prompt,
            "stream": False
        })
        res.raise_for_status()
        result = res.json().get("response", "No response.")
    except Exception as e:
        result = f"Error: {str(e)}"
    st.markdown("### 🧠 LLM Response" if is_en else "### 🧠 模型回應")
    st.write(result)
