import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
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
from plotting_utils import plot_confusion_matrix, get_precision_curve, plot_roc_curve, plot_feature_importances, plot_candlesticks
from utils import get_sp500_tickers, get_nse_tickers
from data_processing import create_feature_cols, check_if_today_starts_with_vertical_green_overlay
import backtrader as bt
# from models import train_model, test_model
from Models.predictive_sma20_crossover_model import PredictiveSma20CrossoverModel
from backtesting import backtest_strategy
from utils import setup_logger

train_until = date(2019, 1, 1)
selected_date = st.date_input("Train Until: ", train_until)
train_until = selected_date.strftime('%Y-%m-%d')

if st.toggle('Indian ticker data'):
    options = get_nse_tickers()
    st.session_state['data_source'] = 'nse'
else:
    options = get_sp500_tickers()
    st.session_state['data_source'] = 'yf'
    
selected_option = st.selectbox("Select a stock", options, index=0, key="my_selectbox")
st.write("You selected:", selected_option)

if st.button("Train Model"):
    model = PredictiveSma20CrossoverModel(selected_option, train_until)
    model_results = model.run_train()

    st.write(f'Training Accuracy: {model_results["train_accuracy"]}')
    st.write(f'Training Precision: {model_results["train_precision"]}')
    st.write('')
    st.write(f'Test Accuracy: {model_results["test_accuracy"]}')
    st.write(f'Test Precision: {model_results["test_precision"]}')

    with st.expander("Show Plots"):
        plot_confusion_matrix(model_results['y_train'], model_results['y_train_pred'], title='Training Data', normalize=False)
        plot_confusion_matrix(model_results['y_train'], model_results['y_train_pred'], title='Training Data - Normalized', normalize=True)

        plot_confusion_matrix(model_results['y_test'], model_results['y_test_pred'], title='Testing Data', normalize=False)
        plot_confusion_matrix(model_results['y_test'], model_results['y_test_pred'], title='Testing Data - Normalized', normalize=True)

        get_precision_curve(model_results['clf'], model_results['x_train'], model_results['y_train'], 'Training - Precision as a Function of Probability')
        get_precision_curve(model_results['clf'], model_results['x_test'], model_results['y_test'], 'Testing - Precision as a Function of Probability')

        plot_roc_curve(model_results['y_train'], model_results['clf'].predict_proba(model_results['x_train'])[:, 1], 'ROC Curve for Training Data')
        plot_roc_curve(model_results['y_test'], model_results['clf'].predict_proba(model_results['x_test'])[:, 1], 'ROC Curve for Test Data')
        plot_feature_importances(model_results['clf'], model_results['x_train'])
    
    backtest_strategy(model_results['df_test'])

def test_date_input_handler():
    """
    Handles the input for the test date, initializes session state variables if not present, and provides buttons for moving the date forward or backward.

    Parameters:
    None

    Returns:
    None
    """
    # Initialize session state variables if not present
    if 'curr_date' not in st.session_state:
        st.session_state.curr_date = datetime.now().date()
    if 'update_flag' not in st.session_state:
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

with st.expander("Test Model"):
    last_n_days = st.text_input("Last N Days", "30")
    test_date_input_handler()

    # Display the date input with the current date from session state
    seld = st.date_input("End Date", value=st.session_state.curr_date, key=st.session_state.update_flag)

    st.write(f"End Date: {st.session_state.curr_date}")
    if st.button("Test Model", key="btn2"):
        model = PredictiveSma20CrossoverModel(selected_option, train_until)
        # Adding 1 day to the end date because yfinance downloads data up to one day before the end date
        df_test = model.run_test(selected_option, last_n_days, (datetime.combine(seld+timedelta(days=1), datetime.min.time())), data_source=st.session_state['data_source'])
        plot_candlesticks(df_test)

with st.expander("Scan all stocks where the model recommends a buy today"):
    last_n_days = st.text_input("Last N Days", "30", key="txt2")
    mlist = []
    if st.button("Scan Stocks", key="btn3"):
        for so in options:
            logger = setup_logger(so)

            try: 
                model = PredictiveSma20CrossoverModel(so, train_until, data_source=st.session_state['data_source'])
                model_results = model.run_train()

                # Log conditions and decisions
                logger.info(f"Train Accuracy: {model_results['train_accuracy']}, Test Accuracy: {model_results['test_accuracy']}")
                logger.info(f"Train Precision: {model_results['train_precision']}, Test Precision: {model_results['test_precision']}")

                if model_results['train_accuracy'] > 0.6 and model_results['test_accuracy'] > 0.6 and model_results['test_precision'] > 0.6 and model_results['train_precision'] > 0.6:
                    df_test = model.run_test(so, last_n_days, data_source=st.session_state['data_source'])
                    if check_if_today_starts_with_vertical_green_overlay(df_test):
                        mlist.append(so)
                        logger.info(f"Buy recommendation for {so}")
                    else:
                        logger.info(f"No recommendation for {so}")
            except Exception as e:
                logger.error(f"Failed to process {so} due to error: {str(e)}")
        st.write(mlist)