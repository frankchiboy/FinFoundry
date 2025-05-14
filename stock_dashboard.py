import yfinance as yf
import streamlit as st
import pandas as pd
import ta

# UI
st.title("📈 投資策略預測分析儀表板")
ticker = st.text_input("輸入股票代碼", "AAPL")
start = st.date_input("起始日期", pd.to_datetime("2022-01-01"))
end = st.date_input("結束日期", pd.to_datetime("today"))

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
st.metric("📊 預測明日收盤價", f"${prediction:.2f}")
