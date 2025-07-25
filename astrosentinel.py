import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization

# API Configuration
TWELVE_DATA_API = "245d89de7bc6459baaebf1de49a8621d"
SYMBOL = "XAU/USD"
INTERVAL = "15min"
OUTPUT_SIZE = 100  # Number of data points to fetch

# Streamlit App Title with Enhanced Styling
st.markdown("""
    <style>
        .title {
            font-size: 32px !important;
            font-weight: bold !important;
            color: #4f8bf9 !important;
            text-align: center;
            margin-bottom: 25px;
        }
        .version {
            font-size: 14px;
            color: #7f7f7f;
            text-align: center;
            margin-top: -15px;
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">AstroSentinel</div>', unsafe_allow_html=True)
st.markdown('<div class="version">V1.0 - Gold Trading Signals</div>', unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_realtime_data():
    """Fetch real-time gold price data from Twelve Data API"""
    url = f"https://api.twelvedata.com/time_series?symbol={SYMBOL}&interval={INTERVAL}&outputsize={OUTPUT_SIZE}&apikey={TWELVE_DATA_API}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'values' not in data:
            st.error("⚠️ No price data available in API response")
            return pd.DataFrame()
        
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['close'] = df['close'].astype(float)
        return df.set_index('datetime')[['close']].sort_index()
    
    except Exception as e:
        st.error(f"🚨 API Error: {str(e)}")
        return pd.DataFrame()

def prepare_data(data):
    """Normalize data between 0.9 and 1.1"""
    scaler = MinMaxScaler(feature_range=(0.9, 1.1))
    scaled = scaler.fit_transform(data)
    return scaled, scaler

def create_sequences(data, length=16):
    """Create LSTM sequences"""
    X, y = [], []
    for i in range(length, len(data)):
        X.append(data[i-length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def build_model():
    """Build Keras LSTM model"""
    model = Sequential([
        Input(shape=(16, 1)),
        BatchNormalization(),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Main Execution
data = get_realtime_data()

if not data.empty:
    with st.spinner('🔄 Processing market data...'):
        scaled_data, scaler = prepare_data(data)
        X, y = create_sequences(scaled_data)

        model = build_model()
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        # Generate Prediction
        current_price = data.iloc[-1]['close']
        predicted_price = scaler.inverse_transform(model.predict(X[-1].reshape(1, 16, 1)))[0][0]
        pct_change = ((predicted_price - current_price) / current_price) * 100

        # Determine Signal
        if predicted_price > current_price + 3.5:
            signal = "🚀 BUY (Strong)"
            signal_color = "green"
        elif predicted_price > current_price + 1.5:
            signal = "📈 Buy"
            signal_color = "lightgreen"
        elif predicted_price < current_price - 3.5:
            signal = "💥 SELL (Strong)"
            signal_color = "red"
        elif predicted_price < current_price - 1.5:
            signal = "📉 Sell"
            signal_color = "pink"
        else:
            signal = "⭐️ Hold"
            signal_color = "gray"

    # Display Results
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 📊 Current Price\n"
                   f"<h2 style='color:#4f8bf9'>${current_price:.2f}</h2>", 
                   unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"### 🔮 Predicted Price\n"
                   f"<h2 style='color:#4f8bf9'>${predicted_price:.2f}</h2>", 
                   unsafe_allow_html=True)
    
    st.markdown(f"### 🎯 Trading Signal\n"
               f"<h2 style='color:{signal_color}'>{signal}</h2>"
               f"<p>Expected Change: {pct_change:.2f}%</p>",
               unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📈 Price History (Last 50 Periods)")
    st.line_chart(data.tail(50))
    
    st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
else:
    st.warning("No data available. Please check your API connection.")
