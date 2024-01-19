from sklearn.ensemble import RandomForestClassifier
from cache_utils import save_model_and_training_date, load_model
import pickle
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
from data_processing import create_feature_cols
from utils import my_yf_download, my_nse_download, convert_date_to_string, convert_string_to_date
from .model_utils import BaseModel2
import talib
from pycaret.classification import setup, compare_models, tune_model, finalize_model, predict_model

from sklearn.metrics import classification_report, confusion_matrix
from plotting_utils import plot_confusion_matrix, get_precision_curve, plot_roc_curve, plot_feature_importances, plot_candlesticks

class PredictiveMacdCrossoverModel(BaseModel2):
    """
    A class representing a predictive model for MACD (Moving Average Convergence Divergence) crossover analysis. 
    This model employs default MACD parameter values with 'fastperiod' set to 12, 'slowperiod' set to 26, and 'signalperiod' set to 9. 
    Additionally, it leverages a 14-period Relative Strength Index (RSI) as a feature for prediction. 
    The model is designed to forecast price movements over a 5-day horizon in the future.

    Parameters:
    - selected_option (str): The selected option or choice related to the model.
    - train_until (str): A date until which the model will be trained.
    - data_source (str, optional): The data source for financial data (default is 'yf' for Yahoo Finance).

    Methods:
    - run_train(): Train the predictive model using historical data and evaluate its performance. The method
      prepares training data, trains the model, predicts crossovers, and saves the trained model.
    - prepare_model_train_data(df): Prepare training data for the model, including MACD and RSI calculations,
      target variable definition, and data split.
    - train(df_train): Train the machine learning model using the prepared training data.
    - run_test(selected_option, last_n_days, download_end_date, data_source): Perform testing using live data.
      This method downloads test data, preprocesses it, loads the pre-trained model, and makes predictions.
    - prepare_model_test_data(df_test): Prepare testing data for the model.
    
    Example Usage:
    ```
    model = PredictiveMacdCrossoverModel(selected_option='option1', train_until='2022-01-01')
    train_results = model.run_train()
    test_results = model.run_test(selected_option='option1', last_n_days='30')
    ```

    """
    def __init__(self, selected_option: str, train_until: str, data_source: str = 'yf'):
        super().__init__(selected_option, train_until, data_source)   
        self.model_file_name = f"{self.selected_option}_{self.__class__.__name__}"

    def run_train(self):
        df = self.download_train_data()
        st.write(df.head())
        results = self.prepare_model_train_data(df)
        
        df_train = results['df_train']
        df_test = results['df_test']
        
        final_model = self.train(df_train)

        # Predict on the test data
        predictions = predict_model(final_model, data=df_test)
    
        actual = predictions['target']
        predicted = predictions['prediction_label']
        st.markdown(f"Classification report: \n ```{classification_report(actual, predicted)}```")
        st.write("Confusion matrix: \n", confusion_matrix(actual, predicted))

        df_test['pred_prob'] = predicted.values
        df_test['pred'] = df_test['pred_prob'] == 1
        
        # Save the model
        save_model_and_training_date(self.model_file_name, final_model)

        return {
            'df_test': df_test
        } 
    
    def prepare_model_train_data(self, df):
        df['RSI'] = talib.RSI(df['Close'])
        df['MACD'], df['MACDSignal'], _ = talib.MACD(df['Close'])

        # Create shifted columns for future comparison
        shift_days = 5
        data_future_rsi = df['RSI'].shift(-shift_days)
        data_future_macd = df['MACD'].shift(-shift_days)
        data_future_macdsignal = df['MACDSignal'].shift(-shift_days)
        df = df.dropna()

        date_format = "%Y-%m-%d"
        # Define a target variable based on the predicted crossover
        df['target'] = [1 if (current_macd < current_macd_signal and future_macd > future_macd_signal) or
                            (current_macd > current_macd_signal and future_macd < future_macd_signal)
                        else 0
                        for current_macd, current_macd_signal, future_macd, future_macd_signal in
                        zip(df['MACD'], df['MACDSignal'], data_future_macd, data_future_macdsignal)]
        st.write(df.head())
        df = df.reset_index(drop=True)
        
        df_train = df[df['Date'] < datetime.strptime(self.train_until, date_format)]
        df_test = df[df['Date'] >= datetime.strptime( self.train_until, date_format)]
        
        return {
            'df_train': df_train,
            'df_test': df_test,
        }
    def train(self, df_train):
        clf_experiment = setup(df_train, target='target', session_id=123,
                       normalize=True, verbose=False)
        # Compare different models and select the best one
        best = compare_models()
        tuned_model = tune_model(best)
        final_model = finalize_model(tuned_model)
        return final_model
    
    def run_test(self, selected_option: str, last_n_days: str, download_end_date: datetime=None, data_source: str = 'yf'):
        df2 = self.download_test_data(selected_option, last_n_days, download_end_date, data_source)
        df2 = self.prepare_model_test_data(df2)
        
        clf = load_model(self.model_file_name)
        current_date = datetime.now()

        # Subtract last_n_days days from the current date
        test_till = current_date - timedelta(days=int(last_n_days))
        
        if not download_end_date:
            download_end_date = current_date
            
        df_test = df2[(df2['Date'] > test_till) & (df2['Date'] < download_end_date)].reset_index(drop=True)
        
        # Predict on the test data
        predictions = predict_model(clf, data=df_test)
        predicted = predictions['prediction_label']
        df_test['pred_prob'] = predicted.values
        df_test['pred'] = df_test['pred_prob'] == 1
        
        return df_test
        
    def prepare_model_test_data(self, df_test):
        df_test['RSI'] = talib.RSI(df_test['Close'])
        df_test['MACD'], df_test['MACDSignal'], _ = talib.MACD(df_test['Close'])

        df_test = df_test.dropna()
        return df_test
        