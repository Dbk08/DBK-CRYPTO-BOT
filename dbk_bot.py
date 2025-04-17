# DBK Crypto Bot IA - Bot de trading automatique sur cryptomonnaies
import pandas as pd
import numpy as np
import requests
import datetime
import ta
import time
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Fonction pour envoyer un message via Telegram
def send_telegram_message(message):
    bot_token = st.secrets["BOT_TOKEN"]
    chat_id = st.secrets["CHAT_ID"]
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    requests.post(url, data=payload)

# Récupération des données Binance
def get_binance_data(symbol="BTCUSDT", interval="1h", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

# Ajout des indicateurs
def add_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["sma"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    df.dropna(inplace=True)
    return df

# Préparation des données IA
def prepare_data(df):
    df["target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)
    features = ["rsi", "macd", "sma"]
    X = df[features]
    y = df["target"]
    return X, y

# Entraînement du modèle IA
def train_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# Prédiction du signal
def predict_signal(model, latest_data):
    signal = model.predict(latest_data)[-1]
    return "BUY" if signal == 1 else "SELL"

# Interface Streamlit
st.title("DBK CRYPTO BOT IA")
st.subheader("Analyse IA de BTC/USDT")

with st.spinner("Chargement des données et entraînement du modèle..."):
    df = get_binance_data()
    df = add_indicators(df)
    X, y = prepare_data(df)
    model = train_model(X, y)
    signal = predict_signal(model, X.tail(1))

st.success(f"Signal actuel : {signal}")
st.line_chart(df[["close", "sma"]].tail(50))

if signal == "BUY":
    st.info("Le bot recommande d'acheter.")
else:
    st.info("Le bot recommande de vendre.")

# Envoi automatique sur Telegram
send_telegram_message(f"Signal DBK CRYPTO BOT : {signal}")
