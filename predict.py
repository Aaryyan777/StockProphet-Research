import joblib
import pandas as pd
from prophet import Prophet
import yfinance as yf
import numpy as np
from datetime import date, timedelta

def calc_rsi(d, window=14):
    dif = d.diff(1)
    gn = dif.where(dif > 0, 0)
    ls = -dif.where(dif < 0, 0)
    ag = gn.ewm(com=window - 1, min_periods=window).mean()
    al = ls.ewm(com=window - 1, min_periods=window).mean()
    rs_val = ag / al
    rsi_val = 100 - (100 / (1 + rs_val))
    return rsi_val

def predict_stock(days=7):
    try:
        model = joblib.load('prophet_model.joblib')
    except FileNotFoundError:
        print("Error: prophet_model.joblib not found. Run stock_forecaster.py first.")
        return

    tkr = "AAPL"
    today = date.today()
    data = yf.download(tkr, start="2020-01-01", end=today.strftime('%Y-%m-%d'))

    df_proc = data.reset_index()
    df_proc = df_proc[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
    df_proc.columns = ['ds', 'y', 'Open', 'High', 'Low', 'Volume']

    df_proc['SMA_10'] = df_proc['y'].rolling(window=10).mean()
    df_proc['SMA_30'] = df_proc['y'].rolling(window=30).mean()
    df_proc['RSI'] = calc_rsi(df_proc['y'])

    df_proc.dropna(inplace=True)

    last_date = df_proc['ds'].max()
    fut_dates = [last_date + timedelta(days=x) for x in range(1, days + 1)]
    fut_df = pd.DataFrame({'ds': fut_dates})

    last_sma_10 = df_proc['SMA_10'].iloc[-1]
    last_sma_30 = df_proc['SMA_30'].iloc[-1]
    last_rsi = df_proc['RSI'].iloc[-1]

    fut_df['SMA_10'] = last_sma_10
    fut_df['SMA_30'] = last_sma_30
    fut_df['RSI'] = last_rsi

    forecast = model.predict(fut_df)

    print(f"\nPredicted Stock Prices for the next {days} days:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

if __name__ == "__main__":
    predict_stock(days=5)
