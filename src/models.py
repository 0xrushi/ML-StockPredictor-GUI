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
from utils import my_yf_download

def train_model(selected_option: str, train_until: str) -> dict:
    """
    Trains a machine learning model using the specified ticker data and returns the model and evaluation metrics.

    Parameters:
    - selected_option (str): The ticker data to download and train the model on.
    - train_until (str): The date until which to train the model.

    Returns:
    - dict: A dictionary containing the following key-value pairs:
        - 'train_accuracy' (float): The accuracy of the model on the training set.
        - 'train_precision' (float): The precision of the model on the training set.
        - 'test_accuracy' (float): The accuracy of the model on the test set.
        - 'test_precision' (float): The precision of the model on the test set.
        - 'df_test' (pandas.DataFrame): The test dataset with additional columns for predicted probabilities and predictions.
        - 'y_train_pred' (numpy.ndarray): The predicted labels for the training set.
        - 'y_test_pred' (numpy.ndarray): The predicted labels for the test set.
        - 'x_train' (pandas.DataFrame): The feature matrix for the training set.
        - 'x_test' (pandas.DataFrame): The feature matrix for the test set.
        - 'clf' (RandomForestClassifier): The trained random forest classifier.
        - 'feat_cols' (list): The list of feature columns used in the model.
        - 'y_test' (pandas.Series): The true labels for the test set.
        - 'y_train' (pandas.Series): The true labels for the training set.
    """
    # Download ticker data
    df = my_yf_download(selected_option).reset_index()

    df = create_feature_cols(df)
    df['target'] = df['price_above_ma'].astype(int).shift(-5)
    df = df.dropna()

    feat_cols = [col for col in df.columns if 'feat' in col]

    x_train = df[df['Date'] < train_until][feat_cols]
    y_train = df[df['Date'] < train_until]['target']

    x_test = df[df['Date'] >= train_until][feat_cols]
    y_test = df[df['Date'] >= train_until]['target']

    clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42,
    class_weight='balanced',
    )

    clf.fit(x_train, y_train)

    save_model_and_training_date(selected_option, clf)
    
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)

    df_test = df[df['Date'] >= train_until].reset_index(drop=True)
    df_test['pred_prob'] = clf.predict_proba(x_test)[:, 1]
    df_test['pred'] = df_test['pred_prob'] > 0.5

    return {
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'df_test': df_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'x_train': x_train,
        'x_test': x_test,
        'clf': clf,
        'feat_cols': feat_cols,
        'y_test': y_test,
        'y_train': y_train
    }

def test_model(selected_option: str, last_n_days: str, download_end_date: datetime=None) -> pd.DataFrame:
    """
    Downloads a trained model based on the selected option, and makes predictions on the last n days of data.

    Args:
        selected_option (str): The option selected for downloading the model.
        last_n_days (str): The number of days to make predictions on.
        download_end_date (date, optional): The end date for downloading data. Defaults to the None.

    Returns:
        pandas.DataFrame: A DataFrame containing the predictions for the last n days of data.
    """
    with open(f"models/{selected_option}_model.pkl", "rb") as f:
        clf = pickle.load(f)

    current_date = datetime.now()
    # Subtract last_n_days days from the current date
    test_till = current_date - timedelta(days=int(last_n_days))
    
    # # Format the date as a string
    # test_till = test_till.strftime('%Y-%m-%d')

    if download_end_date is not None:
        df2 = my_yf_download(selected_option, end=download_end_date).reset_index()
    else:
        df2 = my_yf_download(selected_option).reset_index()
        download_end_date = current_date
    df2 = create_feature_cols(df2)

    # show prediction on last last_n_days days
    df_test = df2[(df2['Date'] > test_till) & (df2['Date'] < download_end_date)].reset_index(drop=True)

    df_test['pred_prob'] = clf.predict_proba(df_test[['feat_dist_from_ma_10', 'feat_dist_from_ma_20', 'feat_dist_from_ma_30',
    'feat_dist_from_ma_50', 'feat_dist_from_ma_100', 'feat_dist_from_max_3',
    'feat_dist_from_min_3', 'feat_dist_from_max_5', 'feat_dist_from_min_5',
    'feat_dist_from_max_10', 'feat_dist_from_min_10',
    'feat_dist_from_max_15', 'feat_dist_from_min_15',
    'feat_dist_from_max_20', 'feat_dist_from_min_20',
    'feat_dist_from_max_30', 'feat_dist_from_min_30',
    'feat_dist_from_max_50', 'feat_dist_from_min_50',
    'feat_dist_from_max_100', 'feat_dist_from_min_100', 'feat_price_dist_1',
    'feat_price_dist_2', 'feat_price_dist_3', 'feat_price_dist_4',
    'feat_price_dist_5', 'feat_price_dist_10', 'feat_price_dist_15',
    'feat_price_dist_20', 'feat_price_dist_30', 'feat_price_dist_50',
    'feat_price_dist_100']])[:, 1]
    df_test['pred'] = df_test['pred_prob'] > 0.5

    return df_test