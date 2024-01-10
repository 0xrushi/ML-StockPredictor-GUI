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
from plot_utils import plot_confusion_matrix, get_precision_curve, plot_roc_curve, plot_feature_importances, plot_candlesticks
from utils import create_feature_cols, get_sp500_tickers
import backtrader as bt

# Custom data feed class
class CustomData(bt.feeds.PandasData):
    # Add a 'lines' definition for your custom data line
    lines = ('pred',)

    # add the parameter to the parameters inherited from the base class
    params = (('pred', -1),)

class GreenBlockStrategy(bt.Strategy):
    def __init__(self):
        self.data_pred = self.datas[0].pred  # Assuming 'pred' is part of the data feed
        self.trades = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({'date': self.data.datetime.date(0), 'profit': trade.pnl})


    def next(self):
        # Check if today is the start of a green block
        if not self.position and self.data_pred[0]:
            self.buy(size=1, price=self.data.open[0], exectype=bt.Order.Market)
            self.log(f'BUY EXECUTED, Price: {self.data.open[0]}, Date: {self.data.datetime.date(0)}')

        # Check if today is the end of a green block
        elif self.position and not self.data_pred[0]:
            self.sell(size=1, price=self.data.close[0], exectype=bt.Order.Market)
            self.log(f'SELL EXECUTED, Price: {self.data.close[0]}, Date: {self.data.datetime.date(0)}')

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        st.write(f'{dt.isoformat()} {txt}')


def backtest_strategy(df):
    # Create a Cerebro engine instance
    cerebro = bt.Cerebro()
    initial_cash = 600.0
    cerebro.broker.setcash(initial_cash)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='tradeanalyzer')


    # Load data
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    datafeed = CustomData(dataname=df)

    # Add the data feed
    cerebro.adddata(datafeed)

    # Add the strategy
    cerebro.addstrategy(GreenBlockStrategy)

    # Run the strategy
    results = cerebro.run()
    strat = results[0]

    # Drawdown analysis
    drawdown_info = strat.analyzers.drawdown.get_analysis()
    st.write(f"Max Drawdown: {drawdown_info.max.drawdown}%")

    # Trade analysis
    trade_info = strat.analyzers.tradeanalyzer.get_analysis()
    total_trades = trade_info.total.closed
    trade_dates = [t['date'] for t in strat.trades]
    num_days = (max(trade_dates) - min(trade_dates)).days
    num_weeks = num_days / 7
    num_months = num_days / 30  # Approximation

    avg_trades_per_day = total_trades / num_days
    avg_trades_per_week = total_trades / num_weeks
    avg_trades_per_month = total_trades / num_months

    # Total Profit
    final_portfolio_value = cerebro.broker.getvalue()
    total_profit = final_portfolio_value - initial_cash

    # Total Profit Percentage
    total_profit_percent = (total_profit / initial_cash) * 100

    # Print results

    st.write(f"Average Trades per Day: {avg_trades_per_day}")
    st.write(f"Average Trades per Week: {avg_trades_per_week}")
    st.write(f"Average Trades per Month: {avg_trades_per_month}")
    st.write(f"Total Profit: \${total_profit}")
    st.write(f"Total Profit (%): {total_profit_percent}%")

    # define plot scheme with new additional scheme arguments
    scheme = PlotScheme(decimal_places=5, max_legend_text_width=16)

    figs = cerebro.plot(BacktraderPlotly(show=False, scheme=scheme))


    df['Date'] = df.index
    # Determine the pattern periods to overlay
    df_pattern = (
        df[df['pred']]
        .groupby((~df['pred']).cumsum())
        ['Date']
        .agg(['first', 'last'])
    )

    # Modify the plot with green overlays
    for i, each_run in enumerate(figs):
        for j, each_strategy_fig in enumerate(each_run):
            for idx, row in df_pattern.iterrows():
                each_strategy_fig.add_vrect(
                    x0=row['first'],
                    x1=row['last'],
                    line_width=0,
                    fillcolor='green',
                    opacity=0.2,
                )

            # Save the modified figure
            filename = f'plot_{i}_{j}.html'
            plotly.io.write_html(each_strategy_fig, filename, full_html=True)

            # Generate a link to open the plot in a new tab
            st.markdown(f'[Open Plot {i}-{j}](/{filename})', unsafe_allow_html=True)

def backtest_strategy_old(df):
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

        # Get open prices for the start and close price for the end of the block
        buy_price = df.loc[df['Date'] == start_date, 'Open'].iloc[0]
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

    # Calculate cumulative profit
    trades_df['Cumulative Profit'] = trades_df['Profit'].cumsum()

    # Plotting
    fig, ax1 = plt.subplots()

    # Plot cumulative profit
    ax1.plot(trades_df['Date'], trades_df['Cumulative Profit'], color='green', label='Cumulative Profit')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Profit')
    ax1.legend(loc='upper left')

    st.pyplot(plt)

options = get_sp500_tickers()

selected_option = st.selectbox("Select an stock", options, index=0, key="my_selectbox")

st.write("You selected:", selected_option)

train_until = date(2019, 1, 1)
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

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)

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

with st.expander("Test Model"):
    last_n_days = st.text_input("Last N Days", "30")
    if st.button("Test Model", key="btn2"):

        with open(f"models/{selected_option}_model.pkl", "rb") as f:
            clf = pickle.load(f)

        current_date = datetime.now()
        # Subtract last_n_days days from the current date
        test_until = current_date - timedelta(days=int(last_n_days))
        # Format the date as a string if necessary
        test_until = test_until.strftime('%Y-%m-%d')

        df2 = yf.download(selected_option).reset_index()
        df2 = create_feature_cols(df2)

        # show prediction on last last_n_days days
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
