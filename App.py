import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
def format_stock_symbol(stock_symbol):
    stock_symbol = stock_symbol.upper().strip()

    if stock_symbol.endswith(('.NS', '.BO', '.BS')):
        return stock_symbol
    
    if stock_symbol in get_nse_stock_list():
        return stock_symbol + ".NS"
    elif stock_symbol in get_bse_stock_list():
        return stock_symbol + ".BO"
    else:  
        return stock_symbol  
def get_nse_stock_list():
    return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "TATASTEEL"]  

def get_bse_stock_list():
    return ["500325", "532540", "500112", "500180"]  
def get_stock_data(stock_symbol, start_date, end_date):
    try:
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        if df.empty:
            return None
        return df[['Close']]
    except Exception:
        return None
def get_fundamental_data(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol)
        pe_ratio = stock.info.get("trailingPE", None)
        eps = stock.info.get("trailingEps", None)

        if pe_ratio is None or eps is None:
            return None, None, None
        
        predicted_price = pe_ratio * eps
        return predicted_price, pe_ratio, eps
    except Exception:
        return None, None, None

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
def predict_stock_price_technical(stock_symbol):
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    
    df = get_stock_data(stock_symbol, start_date, end_date)
    if df is None or df.empty:
        return None
    
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    model = train_model(X_train, y_train)
    
    next_day = np.array([[len(df) + 1]])  
    predicted_price_scaled = model.predict(next_day)
    predicted_price = scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))[0][0]
    
    return predicted_price
def get_final_prediction(stock_symbol):
    technical_price = predict_stock_price_technical(stock_symbol)
    fundamental_price, pe_ratio, eps = get_fundamental_data(stock_symbol)
    
    if technical_price is None and fundamental_price is None:
        return None, None, None, None, None
    
    if technical_price is None:
        return None, fundamental_price, fundamental_price, pe_ratio, eps
    
    if fundamental_price is None:
        return technical_price, None, technical_price, None, None
    
    final_price = (technical_price + fundamental_price) / 2
    return technical_price, fundamental_price, final_price, pe_ratio, eps
def get_tradingview_url(stock_symbol):
    if stock_symbol.endswith(".NS"):
        symbol = stock_symbol.replace(".NS", "")
        return f"https://in.tradingview.com/symbols/NSE-{symbol}/"
    elif stock_symbol.endswith(".BO"):
        symbol = stock_symbol.replace(".BO", "")
        return f"https://in.tradingview.com/symbols/BSE-{symbol}/"
    else:  
        return f"https://www.tradingview.com/symbols/{stock_symbol}/"
st.title("Stock Price Prediction ")
stock_symbol = st.text_input("Enter Stock Ticker").upper()
stock_symbol = format_stock_symbol(stock_symbol)

if st.button("Predict"):
    if stock_symbol:
        technical_price, fundamental_price, final_prediction, pe_ratio, eps = get_final_prediction(stock_symbol)
        st.subheader("Analysis Results :")
        if pe_ratio is not None and eps is not None:
            st.write(f"**P/E Ratio:** {pe_ratio:.2f}")
            st.write(f"**Earnings Per Share (EPS):** {eps:.2f}")
        else:
            st.warning("P/E Ratio and EPS data unavailable.")

        if fundamental_price is not None:
            st.write(f"**Fundamental Analysis Prediction:** ₹{fundamental_price:.2f}")
        else:
            st.warning("Fundamental Analysis data unavailable.")

        if technical_price is not None:
            st.write(f"**Technical Analysis Prediction:** ₹{technical_price:.2f}")
        else:
            st.warning("Technical Analysis data unavailable.")

        if final_prediction is not None:
            st.subheader("Final Predicted Stock Price:")
            st.write(f"**Predicted Price (Avg of Technical & Fundamental):** ₹{final_prediction:.2f}")
        else:
            st.error("Could not predict stock price due to missing data.")
        tradingview_url = get_tradingview_url(stock_symbol)
        st.markdown(f"[🔗 View {stock_symbol} Chart on TradingView]({tradingview_url})", unsafe_allow_html=True)

    else:
        st.warning("Please enter a valid stock ticker.")
