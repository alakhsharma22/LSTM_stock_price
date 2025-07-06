import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def download_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        cols, wanted = [], {"open","high","low","close","volume"}
        for c in df.columns:
            l0, l1 = str(c[0]).lower(), str(c[1]).lower()
            if l0 in wanted and l1 not in wanted:
                cols.append(l0)
            elif l1 in wanted:
                cols.append(l1)
            else:
                cols.append(l0)
        df.columns = cols
    else:
        df.columns = df.columns.str.lower()
    needed = ["open","high","low","close","volume"]
    df = df[needed]
    df.columns = ["Open","High","Low","Close","Volume"]
    return df.dropna()

def add_technical_indicators(df):
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    std20      = df["Close"].rolling(20).std()
    df["BB_up"]  = df["MA20"] + 2 * std20
    df["BB_low"] = df["MA20"] - 2 * std20

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = -delta.clip(upper=0).rolling(14).mean()
    rs    = gain / loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["Open_prev"] = df["Open"].shift(1)
    df["OC_diff"]   = df["Open"].shift(1) - df["Close"].shift(1)
    return df.dropna()

def prepare_data(feature_cols, target_cols, ticker, start, end):
    df = download_data(ticker, start, end)
    df = add_technical_indicators(df)
    feats = df[feature_cols].values
    tars  = df[target_cols].values

    feat_scaler = MinMaxScaler().fit(feats)
    tar_scaler  = MinMaxScaler().fit(tars)
    feats_scaled = feat_scaler.transform(feats)
    tars_scaled  = tar_scaler.transform(tars)

    return feats_scaled, tars_scaled, feat_scaler, tar_scaler
