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
from NSEDownload import stocks

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def convert_date_to_string(df, column_name):
    """
    Converts the specified date column to a string if it's in datetime or date format.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the date column.
    - column_name (str): The name of the column to check and convert.

    Returns:
    - pandas.DataFrame: The DataFrame with the date column converted to a string.
    """
    if column_name in df.columns:
        # Check if the column is either datetime64[ns] or date
        if pd.api.types.is_datetime64_any_dtype(df[column_name]) or pd.api.types.is_dtype(df[column_name], 'date'):
            df[column_name] = df[column_name].astype(str)
            print(f"Column '{column_name}' converted to string.")
        else:
            print(f"Column '{column_name}' is not a datetime or date type.")
    else:
        print(f"Column '{column_name}' does not exist in the DataFrame.")

    return df

def convert_string_to_date(df, column_name):
    """
    Converts the specified column from string to datetime if possible.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the column.
    - column_name (str): The name of the column to check and convert.

    Returns:
    - pandas.DataFrame: The DataFrame with the column converted to datetime.
    """
    if column_name in df.columns:
        try:
            df[column_name] = pd.to_datetime(df[column_name])
            print(f"Column '{column_name}' converted to datetime.")
        except Exception as e:
            print(f"Conversion failed: {e}")
    else:
        print(f"Column '{column_name}' does not exist in the DataFrame.")

    return df

def setup_logger(stock_name: str) -> logging.Logger:
    """
    Creates and configures a logger for the specified stock name.
    
    Parameters:
        stock_name (str): The name of the stock for which the logger is being setup.
    
    Returns:
        logger (Logger): The configured logger object.
    """
    if not os.path.exists('logs'):
        os.makedirs('logs')

    logger = logging.getLogger(stock_name)

    logger.setLevel(logging.DEBUG)

    # Create a file handler which logs messages
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(f'logs/model_{stock_name}_{timestamp}.log')
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

def my_yf_download(ticker: str, cache_dir="cache", end: str=None):
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
    
    end_date = end if end else datetime.now()
    
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

        # Download new data
        if not start_date or (start_date.date() < end_date.date()):
            print(f"Downloading new data for {ticker} from {start_date} to {end_date}")
            df_new = yf.download(ticker, start=start_date.date(), end=end_date.date()).reset_index()
            
            # Append new data to the file
            if not df_new.empty:
                df_new.to_csv(file_path, mode='a', header=False, index=False)

    else:
        print(f"Downloading full data for {ticker}")
        df_cached = yf.download(ticker, end=end_date.date()).reset_index()
        df_cached.to_csv(file_path, index=False)

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.drop_duplicates(subset='Date', keep='last', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def my_nse_download(ticker: str, cache_dir="cache", start: str=None, end: str=None):
    """
    Downloads financial data for a given NSE ticker and stores it in a cache directory.
    
    Parameters:
        ticker (str): The ticker symbol of the financial instrument.
        cache_dir (str, optional): The directory where the data will be cached. Defaults to "cache".
        start (str, optional): The start date for the data download. Defaults to None.
        end (str, optional): The end date for the data download. Defaults to None.
        
    Returns:
        pandas.DataFrame: A DataFrame containing the downloaded financial data.
    """
    # Ensure cache directory exists
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    file_path = os.path.join(cache_dir, f"{ticker}.csv")
    
    end_date = end.strftime('%d-%m-%Y') if end else datetime.now().strftime('%d-%m-%Y')
    start_date = start.strftime('%d-%m-%Y') if start else "01-01-2000"

    if os.path.exists(file_path):
        df_cached = pd.read_csv(file_path)
        df_cached['Date'] = pd.to_datetime(df_cached['Date'])

        # Get the last date from the cached data
        if not df_cached.empty:
            last_date = df_cached['Date'].max().strftime('%d-%m-%Y')
        else:
            last_date = None

        # Redefine start date for new data download based on cached data
        if last_date:
            start_date = last_date

        print(f"Downloading new data for {ticker} from {start_date} to {end_date}")
        df_new = stocks.get_data(stock_symbol=ticker, start_date=start_date, end_date=end_date)
        df_new['Date'] = df_new.index
        df_new = df_new [['Date', 'Open Price', 'High Price', 'Low Price', 'Close Price', 'Total Traded Quantity']]
        df_new.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df_new.reset_index(drop=True)
        
        # Append new data to the file
        if not df_new.empty:
            df_new.to_csv(file_path, mode='a', header=False, index=False)
    else:
        print(f"Downloading full data for {ticker}")
        df_cached = stocks.get_data(stock_symbol=ticker, start_date=start_date, end_date=end_date)
        df_cached['Date'] = df_cached.index
        df_cached = df_cached [['Date', 'Open Price', 'High Price', 'Low Price', 'Close Price', 'Total Traded Quantity']]
        df_cached.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df_cached.reset_index(drop=True)
        df_cached.to_csv(file_path, index=False)

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.tz_localize(None)
    df['Date'] = df['Date'].dt.date
    # df = df.drop_duplicates(subset='Date', keep='last', inplace=True)
    # df = df.reset_index(drop=True, inplace=True)
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

def get_nse_tickers():
    data=pd.read_csv("./index/nse_equities.csv")
    symbols=data["symbol"]
    #symbols = [s + ".NS" for s in symbols]
    return symbols

def get_nse_top_gainers_tickers():
    data=pd.read_csv("./index/nse_top_gainers.csv")
    symbols=data["symbol"]
    #symbols = [s + ".NS" for s in symbols]
    return symbols