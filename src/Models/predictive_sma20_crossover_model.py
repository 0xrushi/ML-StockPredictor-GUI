from sklearn.ensemble import RandomForestClassifier
from cache_utils import save_model_and_training_date
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
from .model_utils import BaseModel
from plotting_utils import plot_confusion_matrix, get_precision_curve, plot_roc_curve, plot_feature_importances, plot_candlesticks

class PredictiveSma20CrossoverModel(BaseModel):
    """
    A class representing a predictive model for SMA (Simple Moving Average) 20 crossover analysis.

    This class inherits from the BaseModel and is designed to predict whether a financial instrument's price will
    be above or below its 20-day Simple Moving Average (SMA) in the future 5 days.

    Parameters:
    - selected_option (str): The stock_name of the financial instrument.
    - train_until (str): A date until which the model will be trained.
    - data_source (str, optional): The data source for financial data (default is 'yf' for Yahoo Finance or 'nse' for NSE).

    Attributes:
    - model_results (dict): A dictionary containing model training and evaluation results.

    Example Usage:
    ```
    model = PredictiveSma20CrossoverModel(selected_option='AAPL', train_until='2022-01-01')
    model.run_train()
    model.plot()
    ```
    """
    def __init__(self, selected_option: str, train_until: str, data_source: str = 'yf'):
        super().__init__(selected_option, train_until, data_source)
        self.model_file_name = f"{self.selected_option}_{self.__class__.__name__}"
    
    def prepare_model_train_data(self, df):
        df = convert_string_to_date(df, 'Date')  # Assuming this function is defined elsewhere
        df = create_feature_cols(df)  # Assuming this function is defined elsewhere
        df['target'] = df['price_above_ma'].astype(int).shift(-5)  # Example target definition
        df = df.dropna()

        feat_cols = [col for col in df.columns if 'feat' in col]

        x_train = df[df['Date'] < self.train_until][feat_cols]
        y_train = df[df['Date'] < self.train_until]['target']

        x_test = df[df['Date'] >= self.train_until][feat_cols]
        y_test = df[df['Date'] >= self.train_until]['target']
        
        df_train = df[df['Date'] >= self.train_until]
        
        model_results = {
            'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test,
            'feat_cols': feat_cols,
            'df_train': df_train
        }
        
        return model_results
    
    def train(self, x_train, y_train):
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42,
            class_weight='balanced'
        )
        clf.fit(x_train, y_train)
        return clf
    
    def prepare_model_test_data(self, df_test):
        return create_feature_cols(df_test)
    
    def plot(self):
        with st.expander("Show Plots"):
            if self.model_results:
                model_results = self.model_results
                plot_confusion_matrix(model_results['y_train'], model_results['y_train_pred'], title='Training Data', normalize=False)
                plot_confusion_matrix(model_results['y_train'], model_results['y_train_pred'], title='Training Data - Normalized', normalize=True)

                plot_confusion_matrix(model_results['y_test'], model_results['y_test_pred'], title='Testing Data', normalize=False)
                plot_confusion_matrix(model_results['y_test'], model_results['y_test_pred'], title='Testing Data - Normalized', normalize=True)

                get_precision_curve(model_results['clf'], model_results['x_train'], model_results['y_train'], 'Training - Precision as a Function of Probability')
                get_precision_curve(model_results['clf'], model_results['x_test'], model_results['y_test'], 'Testing - Precision as a Function of Probability')

                plot_roc_curve(model_results['y_train'], model_results['clf'].predict_proba(model_results['x_train'])[:, 1], 'ROC Curve for Training Data')
                plot_roc_curve(model_results['y_test'], model_results['clf'].predict_proba(model_results['x_test'])[:, 1], 'ROC Curve for Test Data')
                plot_feature_importances(model_results['clf'], model_results['x_train'])
            else:
                st.write('Model not yet trained')
            