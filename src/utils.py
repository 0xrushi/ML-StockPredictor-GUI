import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import os
import io
import nsepython
import logging
from datetime import timedelta, datetime

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def my_yf_download(ticker, cache_dir="cache", end: str=None):
    """
    Downloads financial data for a given ticker and stores it in a cache directory.
    
    Parameters:
        ticker (str): The ticker symbol of the financial instrument.
        cache_dir (str, optional): The directory where the data will be cached. Defaults to "cache".
        end (str, optional): The end date for the data download. Defaults to None.
        
    Returns:
        pandas.DataFrame: A DataFrame containing the downloaded financial data.
    """
    # Ensure cache directory exists
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # File path for cached data
    file_path = os.path.join(cache_dir, f"{ticker}.csv")

    if os.path.exists(file_path):
        df_cached = pd.read_csv(file_path)
        df_cached['Date'] = pd.to_datetime(df_cached['Date'])
        
        # Get the last date from the cached data
        if not df_cached.empty:
            last_date = df_cached['Date'].max()
        else:
            last_date = None

        # Define the start and end dates for new data download
        start_date = last_date + pd.Timedelta(days=1) if last_date else None
        end_date = end if end else datetime.now()

        # Download new data
        if not start_date or start_date.date() < end_date.date():
            print(f"Downloading new data for {ticker} from {start_date} to {end_date}")
            df_new = yf.download(ticker, start=start_date, end=end_date).reset_index()
            
            # Append new data to the file
            if not df_new.empty:
                df_new.to_csv(file_path, mode='a', header=False, index=False)

    else:
        print(f"Downloading full data for {ticker}")
        df_cached = yf.download(ticker, end=end_date).reset_index()
        df_cached.to_csv(file_path, index=False)

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.drop_duplicates(subset='Date', keep='last', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_sp500_tickers() -> list:
    """
    Retrieves the tickers of the companies listed in the S&P 500 index.

    Returns:
        list: A list of ticker symbols representing the companies in the S&P 500 index.
    """
    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0]
    df = pd.read_html(str(table))[0]
    return df['Symbol'].tolist()

# def get_sp500_tickers():
#     symbols = nsepython.nse_eq_symbols()
#     symbols = [s + ".NS" for s in symbols]
#     return symbols