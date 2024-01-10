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
from utils import get_sp500_tickers
from data_processing import create_feature_cols, check_if_today_starts_with_vertical_green_overlay
import backtrader as bt
from models import train_model, test_model
from backtesting import backtest_strategy
import logging

def setup_logger(stock_name):
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

options = get_sp500_tickers()

selected_option = st.selectbox("Select an stock", options, index=0, key="my_selectbox")

st.write("You selected:", selected_option)

train_until = date(2019, 1, 1)
selected_date = st.date_input("Train Until: ", train_until)
train_until = selected_date.strftime('%Y-%m-%d')

if st.button("Train Model"):
    model_results = train_model(selected_option, train_until)

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

with st.expander("Test Model"):
    last_n_days = st.text_input("Last N Days", "30")
    if st.button("Test Model", key="btn2"):
        df_test = test_model(selected_option, last_n_days)
        plot_candlesticks(df_test)

with st.expander("Scan all stocks where the model recommends a buy today"):
    last_n_days = st.text_input("Last N Days", "30", key="txt2")
    mlist = []
    if st.button("Scan Stocks", key="btn3"):
        for so in ["GOOG", *options[:2]]:
            logger = setup_logger(so)

            model_results = train_model(so, train_until)

            # Log conditions and decisions
            logger.info(f"Train Accuracy: {model_results['train_accuracy']}, Test Accuracy: {model_results['test_accuracy']}")
            logger.info(f"Train Precision: {model_results['train_precision']}, Test Precision: {model_results['test_precision']}")

            if model_results['train_accuracy'] > 0.6 and model_results['test_accuracy'] > 0.6 and model_results['test_precision'] > 0.6 and model_results['train_precision'] > 0.6:
                df_test = test_model(so, last_n_days)
                if check_if_today_starts_with_vertical_green_overlay(df_test):
                    mlist.append(so)
                    logger.info(f"Buy recommendation for {so}")
                else:
                    logger.info(f"No recommendation for {so}")
        st.write(mlist)