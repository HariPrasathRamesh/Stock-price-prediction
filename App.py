
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import datetime
import streamlit as st

def get_stock_data(stock_symbol, start_date, end_date):
    try:
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        if df.empty:
            return None
        return df[['Close']]  # Use only closing prices
    except Exception as e:
        return None

def prepare_data(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X = np.array(range(len(df_scaled))).reshape(-1, 1)
    y = df_scaled

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict_stock_price(stock_symbol):
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')

    df = get_stock_data(stock_symbol, start_date, end_date)
    if df is None:
        return "Error: Invalid stock ticker or data unavailable."

    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    model = train_model(X_train, y_train)

    next_day = np.array([[len(df) + 1]])  
    predicted_price_scaled = model.predict(next_day)
    predicted_price = scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))[0][0]

    return f"Predicted closing price for tomorrow: ${predicted_price:.2f}"

st.title("📈 Stock Price Prediction App")
st.write("Enter a stock ticker symbol to predict its closing price for the next day.")

# User input
stock_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)").upper()

if st.button("Predict"):
    if stock_symbol:
        result = predict_stock_price(stock_symbol)
        st.subheader("Prediction Result")
        st.write(result)
    else:
        st.warning("⚠️ Please enter a valid stock ticker.")
