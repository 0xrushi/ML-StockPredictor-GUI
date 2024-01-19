import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import nsepython
import io
from datetime import datetime, date, timedelta
import os
from backtrader_plotly.plotter import BacktraderPlotly
from backtrader_plotly.scheme import PlotScheme
import backtrader.analyzers as btanalyzers
import plotly.io

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
import pickle
from cache_utils import save_model_and_training_date, should_retrain
from plotting_utils import (
    plot_confusion_matrix,
    get_precision_curve,
    plot_roc_curve,
    plot_feature_importances,
    plot_candlesticks,
)
from utils import get_sp500_tickers, get_nse_tickers, get_nse_top_gainers_tickers
from data_processing import (
    create_feature_cols,
    check_if_today_starts_with_vertical_green_overlay,
)
import backtrader as bt

# from models import train_model, test_model
from Models.predictive_sma20_crossover_model import PredictiveSma20CrossoverModel
from Models.predictive_macd_crossover_model import PredictiveMacdCrossoverModel
from strategies.backtesting import backtest_strategy
from utils import setup_logger


def test_date_input_handler():
    """
    Handles the input for the test date, initializes session state variables if not present, and provides buttons for moving the date forward or backward.

    Parameters:
    None

    Returns:
    None
    """
    # Initialize session state variables if not present
    if "curr_date" not in st.session_state:
        st.session_state.curr_date = datetime.now().date()
    if "update_flag" not in st.session_state:
        st.session_state.update_flag = False

    # Buttons for moving the date forward or backward
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("←", key="back"):
            st.session_state.curr_date -= timedelta(days=1)
            st.session_state.update_flag = not st.session_state.update_flag
    with col2:
        st.write("use these arrows to traverse the calendar")
    with col3:
        if st.button("→", key="forward"):
            st.session_state.curr_date += timedelta(days=1)
            st.session_state.update_flag = not st.session_state.update_flag


def main():
    index=['NSE STOCKS','US STOCKS',"NSE TOP GAINERS",'NIFTY 50', 'NIFTY NEXT 50', 'NIFTY IT', 'NIFTY BANK', 'INDIA VIX', 'NIFTY 100', 'NIFTY 500', 'NIFTY MIDCAP 100', 'NIFTY MIDCAP 50', 'NIFTY INFRA', 'NIFTY REALTY', 'NIFTY FMCG', 'NIFTY GS 8 13YR', 'NIFTY IND DIGITAL', 'NIFTY MICROCAP250', 'NIFTY PSE', 'NIFTY100 LOWVOL30', 'NIFTY50 VALUE 20', 'NIFTY ALPHALOWVOL', 'NIFTY ENERGY', 'NIFTY M150 QLTY50', 'NIFTY MIDCAP 150', 'NIFTY PSU BANK', 'NIFTY SMLCAP 250', 'NIFTY50 PR 2X LEV', 'NIFTY50 TR 1X INV', 'NIFTY 200', 'NIFTY MID LIQ 15', 'NIFTY PVT BANK', 'NIFTY SERV SECTOR', 'NIFTY100 ESG', 'NIFTY100 LIQ 15', 'NIFTY100ESGSECLDR', 'NIFTY AUTO', 'NIFTY CONSUMPTION', 'NIFTY GS 10YR CLN', 'NIFTY GS 11 15YR', 'NIFTY GS COMPSITE', 'NIFTY INDIA MFG', 'NIFTY METAL', 'NIFTY SMLCAP 50', 'NIFTY200 ALPHA 30', 'NIFTY50 TR 2X LEV', 'NIFTY ALPHA 50', 'NIFTY COMMODITIES', 'NIFTY DIV OPPS 50', 'NIFTY FINSRV25 50', 'NIFTY GROWSECT 15', 'NIFTY HEALTHCARE', 'NIFTY MID SELECT', 'NIFTY OIL AND GAS', 'INDEX1 NSETEST', 'INDEX2 NSETEST', 'NIFTY GS 10YR', 'NIFTY GS 15YRPLUS', 'NIFTY MEDIA', 'NIFTY50 DIV POINT', 'NIFTY CONSR DURBL', 'NIFTY CPSE', 'NIFTY FIN SERVICE', 'NIFTY GS 4 8YR', 'NIFTY MIDSML 400', 'NIFTY MNC', 'NIFTY SMLCAP 100', 'NIFTY200 QUALTY30', 'NIFTY50 PR 1X INV', 'NIFTY LARGEMID250', 'NIFTY PHARMA', 'NIFTY TOTAL MKT', 'NIFTY100 EQL WGT', 'NIFTY100 QUALTY30', 'NIFTY200 MOMENTM30', 'NIFTY50 EQL WGT', 'NIFTY500 MULTICAP', 'Nifty Midcap150 Momentum 50', 'NIFTY100 ALPHA 30', 'NIFTY CORE HOUSING', 'NIFTY MOBILITY', 'NIFTY MIDCAP150 MOMENTUM 50', 'NIFTY REITS & INVITS', 'NIFTY SME EMERGE', 'NIFTY INDIA DEFENCE', 'NIFTY HOUSING', 'NIFTY TRANSPORTATION & LOGISTICS', 'NIFTY200 ALPHA 30', 'NIFTY MIDSMALL IT & TELECOM', 'NIFTY MIDSMALL HEALTHCARE', 'NIFTY MIDSMALL FINANCIAL SERVICES', 'NIFTY MIDSMALL INDIA CONSUMPTION', 'NIFTY LOW VOLATILITY 50', 'NIFTY HIGH BETA 50', 'NIFTY500 SHARIAH', 'NIFTY50 SHARIAH', 'NIFTY ADITYA BIRLA GROUP', 'NIFTY MAHINDRA GROUP', 'NIFTY TATA GROUP', 'NIFTY TATA GROUP 25% CAP', 'NIFTY SHARIAH 25', 'NIFTY 50 FUTURES INDEX', 'NIFTY 50 ARBITRAGE', 'NIFTY QUALITY LOW-VOLATILITY 30', 'NIFTY ALPHA QUALITY LOW-VOLATILITY 30', 'NIFTY ALPHA QUALITY VALUE LOW-VOLATILITY 30', 'NIFTY FINANCIAL SERVICES EX-BANK', 'NIFTY100 ENHANCED ESG', 'NIFTY NON-CYCLICAL CONSUMER', 'NIFTY SMALLCAP250 QUALITY 50', 'NIFTY50 USD', 'NIFTY 1D RATE INDEX', 'NIFTY G-SEC JUN 2036 INDEX', 'NIFTY G-SEC SEP 2032 INDEX', 'NIFTY G-SEC DEC 2029 INDEX', 'NIFTY G-SEC OCT 2028 INDEX', 'NIFTY G-SEC APR 2029 INDEX', 'NIFTY G-SEC MAY 2029 INDEX', 'NIFTY INDIA SOVEREIGN GREEN BOND JAN 2033 INDEX', 'NIFTY INDIA SOVEREIGN GREEN BOND JAN 2028 INDEX', 'NIFTY G-SEC JUL 2027 INDEX', 'NIFTY G-SEC JUL 2033 INDEX', 'NIFTY 10 YEAR SDL INDEX', 'NIFTY BHARAT BOND INDEX - APRIL 2030', 'NIFTY BHARAT BOND INDEX - APRIL 2025', 'NIFTY CPSE BOND PLUS SDL SEP 2024 50:50 INDEX', 'NIFTY SDL APR 2026 TOP 20 EQUAL WEIGHT INDEX', 'NIFTY AAA BOND PLUS SDL APR 2026 50:50 INDEX', 'NIFTY SDL PLUS PSU BOND SEP 2026 60:40 INDEX', 'NIFTY PSU BOND PLUS SDL SEP 2027 40:60 INDEX', 'NIFTY PSU BOND PLUS SDL APR 2027 50:50 INDEX', 'NIFTY AAA BOND PLUS SDL APR 2026 70:30 INDEX', 'NIFTY AAA BOND PLUS SDL APR 2031 70:30 INDEX', 'NIFTY BHARAT BOND INDEX - APRIL 2032', 'NIFTY CPSE BOND PLUS SDL SEP 2026 50:50 INDEX', 'NIFTY SDL APR 2027 INDEX', 'NIFTY SDL APR 2027 TOP 12 EQUAL WEIGHT INDEX', 'NIFTY SDL APR 2032 TOP 12 EQUAL WEIGHT INDEX', 'NIFTY SDL PLUS G-SEC JUN 2028 30:70 INDEX', 'NIFTY SDL PLUS AAA PSU BOND DEC 2027 60:40 INDEX', 'NIFTY SDL JUN 2027 INDEX', 'NIFTY SDL SEP 2027 INDEX', 'NIFTY AAA CPSE BOND PLUS SDL APR 2027 60:40 INDEX', 'NIFTY AQLV 30 PLUS 5YR G-SEC 70:30 INDEX', 'NIFTY MULTI ASSET - EQUITY : DEBT : ARBITRAGE : REITS/INVITS (50:20:20:10) INDEX', 'NIFTY MULTI ASSET - EQUITY : ARBITRAGE : REITS/INVITS (50:40:10) INDEX', 'NIFTY SDL SEP 2025 INDEX', 'NIFTY SDL DEC 2028 INDEX', 'NIFTY SDL PLUS AAA PSU BOND JUL 2028 60:40 INDEX', 'NIFTY AAA PSU BOND PLUS SDL APR 2026 50:50 INDEX', 'NIFTY AAA PSU BOND PLUS SDL SEP 2026 50:50 INDEX', 'NIFTY SDL PLUS AAA PSU BOND JUL 2033 60:40 INDEX', 'NIFTY SDL PLUS G-SEC JUN 2028 70:30 INDEX', 'NIFTY SDL SEP 2026 INDEX', 'NIFTY BHARAT BOND INDEX - APRIL 2033', 'NIFTY SDL SEP 2026 V1 INDEX', 'NIFTY SDL JUL 2026 INDEX', 'NIFTY SDL DEC 2026 INDEX', 'NIFTY SDL PLUS G-SEC SEP 2027 50:50 INDEX', 'NIFTY SDL PLUS AAA PSU BOND APR 2026 75:25 INDEX', 'NIFTY SDL PLUS G-SEC JUN 2029 70:30 INDEX', 'NIFTY SDL JUL 2033 INDEX', 'NIFTY SDL OCT 2026 INDEX', 'NIFTY SDL PLUS AAA PSU BOND APR 2028 75:25 INDEX', 'NIFTY SDL PLUS G-SEC JUNE 2027 40:60 INDEX', 'NIFTY SDL JUL 2028 INDEX', 'NIFTY SDL JUNE 2028 INDEX', 'NIFTY 3 YEAR SDL INDEX', 'NIFTY 5 YEAR SDL INDEX', 'NIFTY BHARAT BOND INDEX - APRIL 2031', 'NIFTY PSU BOND PLUS SDL APR 2026 50:50 INDEX', 'NIFTY 5YR BENCHMARK G-SEC INDEX', 'NIFTY INDIA GOVERNMENT FULLY ACCESSIBLE ROUTE (FAR) SELECT 7 BONDS INDEX (INR)', 'NIFTY INDIA GOVERNMENT FULLY ACCESSIBLE ROUTE (FAR) SELECT 7 BONDS INDEX (USD)', 'NIFTY G-SEC JUN 2027 INDEX', 'NIFTY G-SEC DEC 2030 INDEX', 'NIFTY G-SEC DEC 2026 INDEX', 'NIFTY G-SEC JUL 2031 INDEX', 'NIFTY G-SEC SEP 2027 INDEX']
    selected_INDEX = st.selectbox(
        "Select a NSE INDEX", index, index=0, key="my_selectbox_index"
    )

    if st.button("refersh stock list"):
        if selected_INDEX=="NSE STOCKS":
            stock_list=nsepython.nse_eq_symbols()
            #url="https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=U507CSUWEJLQIUPF"
            #stock_list_dataframe=pd.read_csv(io.BytesIO(requests.get(url).content))
            #stock_list_dataframe['symbol'] = stock_list_dataframe['symbol'].apply(lambda x: str(x) + ".NS")
            #print(stock_list_dataframe)
            stock_list_NSE = [symbol + ".NS" for symbol in stock_list]
            stock_list_dataframe = pd.DataFrame({"symbol": stock_list_NSE})
            stock_list_dataframe.to_csv("./index/nse_equities.csv",index=False)
        if selected_INDEX=="NSE TOP GAINERS":
            stock_list=list(nsepython.nse_get_top_gainers()['symbol'])
            #url="https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=U507CSUWEJLQIUPF"
            #stock_list_dataframe=pd.read_csv(io.BytesIO(requests.get(url).content))
            #stock_list_dataframe['symbol'] = stock_list_dataframe['symbol'].apply(lambda x: str(x) + ".NS")
            #print(stock_list_dataframe)
            
            stock_list_NSE = [symbol + ".NS" for symbol in stock_list]
            stock_list_dataframe = pd.DataFrame({"symbol": stock_list_NSE})
            stock_list_dataframe.to_csv("./index/nse_top_gainers.csv",index=False)
    
    train_until = date(2019, 1, 1)
    selected_date = st.date_input("Train Until: ", train_until)
    train_until = selected_date.strftime("%Y-%m-%d")
    
    if selected_INDEX=="US STOCKS":
        options = get_sp500_tickers()
        st.session_state["data_source"] = "yf"
    elif selected_INDEX=="NSE STOCKS":
        options = get_nse_tickers()
        st.session_state["data_source"] = "yf"
    elif selected_INDEX=="NSE TOP GAINERS":
        options= get_nse_top_gainers_tickers()
        st.session_state["data_source"] = "yf"

    selected_option = st.selectbox(
        "Select a stock", options, index=0, key="my_selectbox"
    )

    # Drop down to select model/strategy
    model_names = [
        "PredictiveSma20CrossoverModel",
        "PredictiveMacdCrossoverModel",
    ]  # Add all your model names
    selected_model_name = st.selectbox("Select a Model", model_names)
    if selected_model_name:
        selected_model_class = globals()[selected_model_name]

    st.write("You selected:", selected_option)

    if st.button("Train Model"):
        # model = PredictiveMacdCrossoverModel(selected_option, train_until)
        model = selected_model_class(selected_option, train_until)
        model_results = model.run_train()
        backtest_strategy(model_results["df_test"])

    with st.expander("Test Model"):
        last_n_days = st.text_input("Last N Days", "30")
        test_date_input_handler()

        # Display the date input with the current date from session state
        selected_end_date = st.date_input(
            "End Date",
            value=st.session_state.curr_date,
            key=st.session_state.update_flag,
        )

        st.write(f"End Date: {st.session_state.curr_date}")
        if st.button("Test Model", key="btn2"):
            # model = PredictiveMacdCrossoverModel(selected_option, train_until)
            model = selected_model_class(selected_option, train_until)
            # Adding 1 day to the end date because yfinance downloads data up to one day before the end date
            df_test = model.run_test(
                selected_option,
                last_n_days,
                (
                    datetime.combine(
                        selected_end_date + timedelta(days=1), datetime.min.time()
                    )
                ),
                data_source=st.session_state["data_source"],
            )
            plot_candlesticks(df_test)
            mlist=[]
            if check_if_today_starts_with_vertical_green_overlay(df_test):
                mlist.append(selected_option)
            print("Asdsadasdasdasda:    ",mlist)
            st.write(mlist) 
            
    with st.expander("Scan all stocks where the model recommends a buy today"):
        last_n_days = st.text_input("Last N Days", "30", key="txt2")
        mlist = []
        if st.button("Scan Stocks", key="btn3"):
            for so in options:
                logger = setup_logger(so)

                try:
                    # model = PredictiveMacdCrossoverModel(so, train_until, data_source=st.session_state['data_source'])
                    model = selected_model_class(
                        so, train_until, data_source=st.session_state["data_source"]
                    )
                    model_results = model.run_train()

                    # Log conditions and decisions
                    logger.info(
                        f"Train Accuracy: {model_results['train_accuracy']}, Test Accuracy: {model_results['test_accuracy']}"
                    )
                    logger.info(
                        f"Train Precision: {model_results['train_precision']}, Test Precision: {model_results['test_precision']}"
                    )

                    if (
                        model_results["train_accuracy"] > 0.6
                        and model_results["test_accuracy"] > 0.6
                        and model_results["test_precision"] > 0.6
                        and model_results["train_precision"] > 0.6
                    ):
                        df_test = model.run_test(
                            so, last_n_days, data_source=st.session_state["data_source"]
                        )
                        if check_if_today_starts_with_vertical_green_overlay(df_test):
                            print("Asdsadasdasdasda:    ",mlist)
                            mlist.append(so)
                            logger.info(f"Buy recommendation for {so}")
                        else:
                            logger.info(f"No recommendation for {so}")
                except Exception as e:
                    logger.error(f"Failed to process {so} due to error: {str(e)}")
        st.write(mlist)


if __name__ == "__main__":
    main()
