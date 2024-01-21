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
import backtrader as bt
from src.custom_data_feed import CustomData

class CustomSharpeRatio(bt.Analyzer):
    def __init__(self):
        self.portfolio_values = []

    def next(self):
        # Capture the total portfolio value at each time step
        self.portfolio_values.append(self.strategy.broker.getvalue())

    def get_analysis(self):
        # Convert to a numpy array for easy calculations
        portfolio_values = np.array(self.portfolio_values)

        # Calculate daily returns
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Calculate Sharpe Ratio (assuming risk-free rate is 0 for simplicity)
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # 252 trading days

        # Return the result as a dictionary
        return {'sharpe_ratio': sharpe_ratio}

class GreenBlockStrategy(bt.Strategy):
    params = (('profit_target', 0.30),  # 5% profit target
              ('holding_period', 5),    # 5 days holding period
             )
    def __init__(self):
        self.data_pred = self.datas[0].pred  # Assuming 'pred' is part of the data feed
        self.trades = []
        self.buy_order = None
        self.sell_order = None
        self.entry_price = None
        self.entry_date = None

    def notify_trade(self, trade: bt.analyzers.TradeAnalyzer):
        if trade.isclosed:
            # self.trades.append({'date': self.data.datetime.date(0), 'profit': trade.pnl})
            trade_return = None
            buy_price = self.buy_order.executed.price if self.buy_order else None
            sell_price = self.sell_order.executed.price if self.sell_order else None
            if buy_price and sell_price:
                trade_return = (sell_price - buy_price) / buy_price
            self.trades.append({'date': self.data.datetime.date(0),  'profit': trade.pnl, 'return': trade_return})
            # Reset order references
            self.buy_order = None
            self.sell_order = None
            
            
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_order = order
                self.log('BUsY EXECUssTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                        (order.executed.price,
                        order.executed.value,
                        order.executed.comm))

            elif order.issell():
                self.sell_order = order
                self.log('SELL EssXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                        (order.executed.price,
                        order.executed.value,
                        order.executed.comm))



    def next(self):
        # Check if today is the start of a green block
        if not self.position and self.data_pred[0]:
            self.buy_order = self.buy(size=10, price=self.data.open[0], exectype=bt.Order.Market)
            self.entry_price = self.data.open[0]
            self.entry_date = len(self)  # store the current bar index
            self.log(f'BUY EXECUTED, Price: {self.data.open[0]}, Date: {self.data.datetime.date(0)}')

        # # Check if position is open and holding period has not exceeded
        # elif self.position and (len(self) - self.entry_date < self.p.holding_period):
        #     current_price = self.data.close[0]
        #     if self.entry_price:
        #         # Calculate the current profit
        #         profit = (current_price - self.entry_price) / self.entry_price
        #         if profit > self.p.profit_target:
        #             # Profit target met, close the position
        #             self.sell_order = self.sell(size=10, price=current_price, exectype=bt.Order.Market)
        #             self.log(f'TAKE PROFIT SELL EXECUTED, Price: {current_price}, Date: {self.data.datetime.date(0)}')
        #             self.entry_price = None  # Reset entry price

        # Check if today is the end of a green block or holding period exceeded
        elif self.position and (self.data_pred[0] == 0):
            self.sell_order = self.sell(size=10, price=self.data.close[0], exectype=bt.Order.Market)
            self.entry_price = None  # Reset entry price
            self.log(f'SELL EXECUTED, Price: {self.data.close[0]}, Date: {self.data.datetime.date(0)}')
            
    def log(self, txt: str, dt: datetime = None):
        dt = dt or self.datas[0].datetime.date(0)
        # st.write(f'{dt.isoformat()} {txt}')


def backtest_strategy(df: pd.DataFrame):
    """
    Backtests a trading strategy using the provided DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the trading data.

    Returns:
    - None
    """
    # Create a Cerebro engine instance
    cerebro = bt.Cerebro()
    initial_cash = 5000.0
    cerebro.broker.setcash(initial_cash)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(CustomSharpeRatio, _name="custom_sharpe")


    # Load data
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    datafeed = CustomData(dataname=df)

    # Add the data feed
    cerebro.adddata(datafeed)

    cerebro.addstrategy(GreenBlockStrategy)

    # Run the strategy
    results = cerebro.run()
    strat = results[0]

    # Drawdown analysis
    drawdown_info = strat.analyzers.drawdown.get_analysis()
    st.write(f"Max Drawdown: {drawdown_info.max.drawdown}%")
    sharpe_ratio = results[0].analyzers.sharpe_ratio.get_analysis()
    st.write('Sharpe Ratio:', sharpe_ratio['sharperatio'])

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

    final_portfolio_value = cerebro.broker.getvalue()
    total_profit = final_portfolio_value - initial_cash

    total_profit_percent = (total_profit / initial_cash) * 100

    risk_free_rate = 0.02  # Assume risk-free rate of 2%
    
    df_trades = pd.DataFrame(strat.trades)
    df_trades.set_index('date', inplace=True)
    df_trades['cumulative_profit'] = df_trades['profit'].cumsum()
    profitable_trades = df_trades[df_trades['profit'] > 0].shape[0]
    loss_trades = df_trades[df_trades['profit'] < 0].shape[0]
    profits = df_trades[df_trades['profit'] > 0]['profit']
    losses = df_trades[df_trades['profit'] < 0]['profit']
    
    st.markdown(df_trades['profit'])
    
    average_profit = df_trades['profit'].mean()
    std_dev_profit = df_trades['profit'].std()
    
    # Create the plot
    fig, ax = plt.subplots()
    # sns.kdeplot(gross_profits, fill=True, ax=ax)
    ax.plot(df_trades['cumulative_profit'])
    ax.set_title(' Line lot of profit')
    ax.set_xlabel('Date')
    ax.set_ylabel('Profit')
    
    custom_sharpe_ratio = strat.analyzers.custom_sharpe.get_analysis()['sharpe_ratio']
    st.write("Custom Sharpe Ratio:", custom_sharpe_ratio)

    # Display the plot in Streamlit
    st.pyplot(fig)
    
    
    # Create the Profit/Loss Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df_trades['profit'], bins=30, color='skyblue', alpha=0.7)
    plt.title('Profit/Loss Distribution')
    plt.xlabel('Profit/Loss')
    plt.ylabel('Number of Trades')
    plt.axvline(x=0, color='grey', linestyle='--')
    plt.grid(True)
    st.pyplot()
    
    # Profits and losses box plots
    combined_profit_loss = pd.concat([profits.rename('Value'), losses.rename('Value')], axis=0)
    combined_profit_loss = combined_profit_loss.to_frame()
    combined_profit_loss['Type'] = ['Profit' if v > 0 else 'Loss' for v in combined_profit_loss['Value']]
    plt.figure(figsize=(10, 6))
    sns.boxplot(y='Value', x='Type', data=combined_profit_loss, palette=['lightgreen', 'salmon'])
    plt.title('Box Plot of Profits and Losses')
    plt.xlabel('')
    plt.ylabel('Profit/Loss Value')
    plt.grid(True)
    # Show legend
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ['Profits', 'Losses'], title='Type')
    st.pyplot()
    
    st.write(f"Sharpe Ratio: {sharpe_ratio}")
    st.write('Average Profit:', average_profit)
    st.write('Standard Deviation of Profit:', std_dev_profit)
    st.write(f"Average Trades per Day: {avg_trades_per_day}")
    st.write(f"Average Trades per Week: {avg_trades_per_week}")
    st.write(f"Average Trades per Month: {avg_trades_per_month}")
    st.write(f"Total Profitable Trades: {profitable_trades}")
    st.write(f"Total Loss Trades: {loss_trades}")
    st.write(f"Total Profit: \${total_profit}")
    st.write(f"Total Profit (%): {total_profit_percent}%")

    # define plot scheme with new additional scheme arguments
    scheme = PlotScheme(decimal_places=5, max_legend_text_width=16)

    figs = cerebro.plot(BacktraderPlotly(show=False, scheme=scheme))


    df['Date'] = df.index
    df = df.reset_index(drop=True)
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

            filename = f'plot_{i}_{j}.html'
            plotly.io.write_html(each_strategy_fig, filename, full_html=True)

            # Generate a link to open the plot in a new tab
            st.markdown(f'http://localhost:5001/{filename}', unsafe_allow_html=True)
