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

def train_model(selected_option, train_until):
    # Download ticker data
    df = yf.download(selected_option).reset_index()

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

def test_model(selected_option, last_n_days):
    with open(f"models/{selected_option}_model.pkl", "rb") as f:
        clf = pickle.load(f)

    current_date = datetime.now()
    # Subtract last_n_days days from the current date
    test_till = current_date - timedelta(days=int(last_n_days))
    # Format the date as a string if necessary
    test_till = test_till.strftime('%Y-%m-%d')

    df2 = yf.download(selected_option).reset_index()
    df2 = create_feature_cols(df2)

    # show prediction on last last_n_days days
    df_test = df2[df2['Date'] > test_till].reset_index(drop=True)
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