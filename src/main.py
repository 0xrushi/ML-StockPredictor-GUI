import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import datetime
import os


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
from plot_utils import plot_confusion_matrix, get_precision_curve, plot_roc_curve, plot_feature_importances, plot_candlesticks

def get_sp500_tickers():
    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0]
    df = pd.read_html(str(table))[0]
    return df['Symbol'].tolist()

options = get_sp500_tickers()

selected_option = st.selectbox("Select an option", options, index=0, key="my_selectbox")

st.write("You selected:", selected_option)

if st.button("Get data"):
    # download ticker data
    df = yf.download(selected_option).reset_index()

    # Feature deriving
    # Distance from the moving averages
    for m in [10, 20, 30, 50, 100]:
        df[f'feat_dist_from_ma_{m}'] = df['Close']/df['Close'].rolling(m).mean()-1

    # Distance from n day max/min
    for m in [3, 5, 10, 15, 20, 30, 50, 100]:
        df[f'feat_dist_from_max_{m}'] = df['Close']/df['High'].rolling(m).max()-1
        df[f'feat_dist_from_min_{m}'] = df['Close']/df['Low'].rolling(m).min()-1

    # Price distance
    for m in [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 100]:
        df[f'feat_price_dist_{m}'] = df['Close']/df['Close'].shift(m)-1

    # Target = if the price above the 20 ma in 5 days time
    df['target_ma'] = df['Close'].rolling(20).mean()
    df['price_above_ma'] = df['Close'] > df['target_ma']
    df['target'] = df['price_above_ma'].astype(int).shift(-5)

    df = df.dropna()

    feat_cols = [col for col in df.columns if 'feat' in col]
    train_until = '2019-01-01'

    x_train = df[df['Date'] < train_until][feat_cols]
    y_train = df[df['Date'] < train_until]['target']

    x_test = df[df['Date'] >= train_until][feat_cols]
    y_test = df[df['Date'] >= train_until]['target']

    # train or load model
    if should_retrain(selected_option):
        clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42,
        class_weight='balanced',
        )

        clf.fit(x_train, y_train)
        save_model_and_training_date(selected_option, clf)
    else:
        with open(f"{selected_option}_model.pkl", "rb") as f:
            clf = pickle.load(f)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    # Calculate accuracy and precision for training data
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)

    # Calculate accuracy and precision for test data
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)

    st.write(f'Training Accuracy: {train_accuracy}')
    st.write(f'Training Precision: {train_precision}')
    st.write('')
    st.write(f'Test Accuracy: {test_accuracy}')
    st.write(f'Test Precision: {test_precision}')

    with st.expander("Show Plots"):
        plot_confusion_matrix(y_train, y_train_pred, title='Training Data', normalize=False)
        plot_confusion_matrix(y_train, y_train_pred, title='Training Data - Normalized', normalize=True)

        plot_confusion_matrix(y_test, y_test_pred, title='Testing Data', normalize=False)
        plot_confusion_matrix(y_test, y_test_pred, title='Testing Data - Normalized', normalize=True)

        get_precision_curve(clf, x_train, y_train, 'Training - Precision as a Function of Probability')
        get_precision_curve(clf, x_test, y_test, 'Testing - Precision as a Function of Probability')

        plot_roc_curve(y_train, clf.predict_proba(x_train)[:, 1], 'ROC Curve for Training Data')
        plot_roc_curve(y_test, clf.predict_proba(x_test)[:, 1], 'ROC Curve for Test Data')
        plot_feature_importances(clf)

    df_test = df[df['Date'] >= train_until].reset_index(drop=True)
    df_test['pred_prob'] = clf.predict_proba(x_test)[:, 1]
    df_test['pred'] = df_test['pred_prob'] > 0.5
    plot_candlesticks(df_test)
