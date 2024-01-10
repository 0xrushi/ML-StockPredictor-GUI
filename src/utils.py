import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import datetime
import os
import io
import nsepython

def get_sp500_tickers():
    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0]
    df = pd.read_html(str(table))[0]
    return df['Symbol'].tolist()

# def get_sp500_tickers():
#     symbols = nsepython.nse_eq_symbols()
#     symbols = [s + ".NS" for s in symbols]
#     return symbols