import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import datetime
import os


# Feature deriving
# Distance from the moving averages
def create_feature_cols(df):
    for m in [10, 20, 30, 50, 100]:
        df[f'feat_dist_from_ma_{m}'] = df['Close']/df['Close'].rolling(m).mean()-1

    # Distance from n day max/min
    for m in [3, 5, 10, 15, 20, 30, 50, 100]:
        df[f'feat_dist_from_max_{m}'] = df['Close']/df['High'].rolling(m).max()-1
        df[f'feat_dist_from_min_{m}'] = df['Close']/df['Low'].rolling(m).min()-1

    # Price distance
    for m in [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 100]:
        df[f'feat_price_dist_{m}'] = df['Close']/df['Close'].shift(m)-1

    # Target = if the price above the 20 ma in 5 days time
    df['target_ma'] = df['Close'].rolling(20).mean()
    df['price_above_ma'] = df['Close'] > df['target_ma']
    return df

def get_sp500_tickers():
    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0]
    df = pd.read_html(str(table))[0]
    return df['Symbol'].tolist()