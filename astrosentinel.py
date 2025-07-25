import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
import os

def get_data():
    data = yf.download('GC=F', period='3d', interval='15m', progress=False)
    return data[['Close']].dropna()

def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0.9, 1.1))
    scaled = scaler.fit_transform(data)
    return scaled, scaler

def create_sequences(data, length=16):
    X, y = [], []
    for i in range(length, len(data)):
        X.append(data[i-length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def build_model():
    model = Sequential([
        Input(shape=(16,1)),
        BatchNormalization(),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

st.title("🏆 XAU/USD (Gold) Signal App")
data = get_data()
scaled_data, scaler = prepare_data(data)
X, y = create_sequences(scaled_data)

model = build_model()
model.fit(X, y, epochs=10, batch_size=32, verbose=0)

current_price = data.iloc[-1]['Close']
predicted_price = scaler.inverse_transform(model.predict(X[-1].reshape(1,16,1)))[0][0]
pct_change = ((predicted_price - current_price) / current_price) * 100

if predicted_price > current_price + 3.5:
    signal = "🚀BUY (Strong)"
elif predicted_price > current_price + 1.5:
    signal = "📈Buy"
elif predicted_price < current_price - 3.5:
    signal = "💥SELL (Strong)"
elif predicted_price < current_price - 1.5:
    signal = "📉Sell"
else:
    signal = "⭐️Hold"

st.subheader("📊 Signal Summary")
st.write(f"**Signal:** {signal}")
st.write(f"**Current Price:** ${current_price:.2f}")
st.write(f"**Predicted Price:** ${predicted_price:.2f}")
st.write(f"**Expected Change:** {pct_change:.2f}%")
st.caption(f"Generated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
