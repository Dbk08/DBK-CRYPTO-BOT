
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
import datetime
import json
import os

# ============================
# TELEGRAM CONFIGURATION
# ============================
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# ============================
# BINANCE DATA FETCHING
# ============================
def get_binance_data(symbol="BTCUSDT", interval="1h", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = json.loads(response.text)

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# ============================
# ADD TECHNICAL INDICATORS
# ============================
def add_indicators(df):
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma21'] = df['close'].rolling(window=21).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ============================
# MODEL TRAINING
# ============================
def train_model(X, y):
    if len(X) < 10:
        st.error("Pas assez de données pour entraîner le modèle.")
        return None
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

def predict_signal(model, X):
    return model.predict(X)[0]

# ============================
# STREAMLIT UI
# ============================
st.set_page_config(page_title="DBK CRYPTO BOT IA", layout="centered")
st.title("DBK CRYPTO BOT IA")
st.subheader("Analyse IA de BTC/USDT")

df = get_binance_data()
df = add_indicators(df)

st.write("Aperçu des données :", df.tail())

# Préparation des données
df['future'] = df['close'].shift(-1)
df.dropna(inplace=True)
df['signal'] = np.where(df['future'] > df['close'], 1, 0)

features = ['close', 'ma7', 'ma21', 'rsi']
X = df[features]
y = df['signal']

# Entraînement du modèle
model = train_model(X, y)

if model:
    signal = predict_signal(model, X.tail(1))
    st.success(f"Signal actuel : {'ACHETER' if signal == 1 else 'VENDRE'}")

    # Envoi automatique du signal sur Telegram
    if BOT_TOKEN and CHAT_ID:
        message = f"DBK Crypto Bot IA - Signal BTC/USDT : {'ACHETER' if signal == 1 else 'VENDRE'}"
        requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}")
else:
    st.warning("Modèle non disponible.")

# ============================
# BOUTON DE TEST TELEGRAM
# ============================
def send_test_message():
    bot_token = os.getenv("BOT_TOKEN")
    chat_id = os.getenv("CHAT_ID")
    message = "Test de message : Le bot DBK Crypto Bot IA fonctionne !"

    if bot_token and chat_id:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message
        }
        response = requests.post(url, data=data)

        if response.status_code == 200:
            st.success("Message test envoyé avec succès sur Telegram.")
        else:
            st.error(f"Erreur lors de l'envoi : {response.text}")
    else:
        st.error("BOT_TOKEN ou CHAT_ID introuvable dans les secrets Streamlit.")

if st.button("Envoyer un message test Telegram"):
    send_test_message()
