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
from utils import create_feature_cols, get_sp500_tickers

def backtest_strategy(df):
    # Identify the start and end of each green-filled block
    green_blocks = (
        df[df['pred']]
        .groupby((~df['pred']).cumsum())
        .agg(start=('Date', 'first'), end=('Date', 'last'))
    )

    trade_count = 0
    total_profit = 0
    trades = []

    # Iterate over each green block
    for _, block in green_blocks.iterrows():
        start_date = block['start']
        end_date = block['end']

        # Get close prices for the start and end of the block
        buy_price = df.loc[df['Date'] == start_date, 'Close'].iloc[0]
        sell_price = df.loc[df['Date'] == end_date, 'Close'].iloc[0]

        # Calculate profit for this block and add to total profit
        profit = sell_price - buy_price
        total_profit += profit

        trades.append({'Date': start_date, 'Profit': profit})
        trade_count += 1
        st.write(f"Trade {trade_count}: Buy on {start_date} at \${buy_price}, Sell on {end_date} at \${sell_price}, Profit: \${profit}")

    trades_df = pd.DataFrame(trades)

    # Get the total number of trades
    total_trades = len(trades_df)

    # Calculate the number of unique weeks, months, and days
    num_weeks = len(trades_df['Date'].dt.isocalendar().week.unique())
    num_months = len(trades_df['Date'].dt.to_period('M').unique())
    num_days = len(trades_df['Date'].dt.to_period('D').unique())

    # Calculate average trades
    avg_trades_per_week = total_trades / num_weeks
    avg_trades_per_month = total_trades / num_months
    avg_trades_per_day = total_trades / num_days

    st.write(f"Total Profit: ${total_profit}")
    st.write(f"Average Trades per Week: {avg_trades_per_week}")
    st.write(f"Average Trades per Month: {avg_trades_per_month}")
    st.write(f"Average Trades per Day: {avg_trades_per_day}")

options = get_sp500_tickers()

selected_option = st.selectbox("Select an option", options, index=0, key="my_selectbox")

st.write("You selected:", selected_option)

train_until = datetime.date(2019, 1, 1)
selected_date = st.date_input("Train Until: ", train_until)
train_until = selected_date.strftime('%Y-%m-%d')

if st.button("Train Model"):
    # download ticker data
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
        plot_feature_importances(clf, x_train)
    
    df_test = df[df['Date'] >= train_until].reset_index(drop=True)
    df_test['pred_prob'] = clf.predict_proba(x_test)[:, 1]
    df_test['pred'] = df_test['pred_prob'] > 0.5
    backtest_strategy(df_test)

if st.button("Test Model", key="btn2"):

    with open(f"models/{selected_option}_model.pkl", "rb") as f:
        clf = pickle.load(f)

    current_date = datetime.datetime.now()
    # Subtract 30 days from the current date
    test_until = current_date - datetime.timedelta(days=30)
    # Format the date as a string if necessary
    test_until = test_until.strftime('%Y-%m-%d')

    df2 = yf.download(selected_option).reset_index()
    df2 = create_feature_cols(df2)

    # show prediction on last 30 days
    df_test = df2[df2['Date'] > test_until].reset_index(drop=True)
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

    plot_candlesticks(df_test)
