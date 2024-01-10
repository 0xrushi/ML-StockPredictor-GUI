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
from data_processing import create_feature_cols
import backtrader as bt
from custom_data_feed import CustomData

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