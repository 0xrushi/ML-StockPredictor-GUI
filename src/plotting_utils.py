import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import datetime
import os
import altair as alt


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
import pickle
import logging
from altair_saver import save

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, normalize: bool = False) -> None:
    if normalize:
        cm = confusion_matrix(y_true, y_pred, normalize='pred')
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False)
    else:
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    st.pyplot(plt)

def get_precision_curve(clf: object, x: np.ndarray, y: np.ndarray, title: str):
    """
    Generate the precision-recall curve for a given classifier.

    Parameters:
        - clf (object): The classifier to use for prediction.
        - x (array-like): The input data.
        - y (array-like): The target labels.
        - title (str): The title of the plot.

    Returns:
        None
    """
    y_scores = clf.predict_proba(x)[:, 1]

    thresholds = np.linspace(0, 1, 100)
    precision = []

    for t in thresholds:
        y_pred_threshold = (y_scores >= t).astype(int)
        precision.append(precision_score(y, y_pred_threshold, zero_division=0))

    # Create the plot with matplotlib
    plt.figure()
    plt.plot(thresholds, precision, 'b.')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True)

    # Display the plot in Streamlit
    st.pyplot(plt)

def plot_roc_curve(y_true, y_scores, title):

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    # Display the plot in Streamlit
    st.pyplot(plt)

    return

def plot_feature_importances(clf, x):
    feature_importances = clf.feature_importances_
    sorted_indices = feature_importances.argsort()[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances in The Classifier")
    plt.bar(range(x.shape[1]), feature_importances[sorted_indices], align='center')
    plt.xticks(range(x.shape[1]), x.columns[sorted_indices], rotation=90)
    plt.ylabel('Importance')
    
    # Display the plot in Streamlit
    st.pyplot(plt)

def plot_candlesticks_plotly(df_test):
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df_test['Date'],
            open=df_test['Open'],
            high=df_test['High'],
            low=df_test['Low'],
            close=df_test['Close'],
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Line(x=df_test['Date'], y=df_test[f'target_ma'], name=f'Target SMA')
    )

    df_pattern = (
        df_test[df_test['pred']]
        .groupby((~df_test['pred']).cumsum())
        ['Date']
        .agg(['first', 'last'])
    )

    last_date_included = False
    for idx, row in df_pattern.iterrows():
        fig.add_vrect(
            x0=row['first'],
            x1=row['last'],
            line_width=0,
            fillcolor='green',
            opacity=0.2,
        )

        logger.debug(f"first : {row['first']} last: {row['last']}")
        if df_test.iloc[-1]['Date'] >= row['first'] and df_test.iloc[-1]['Date'] <= row['last'] and (row['last']!=row['first']):
            last_date_included = True
    
    logger.debug(last_date_included)
            
    # Check if the last entry has pred as True and is not included in any green fill
    if df_test.iloc[-1]['pred'] and not last_date_included:
        fig.add_vline(
            x=df_test.iloc[-1]['Date'], 
            line_width=5, 
            line_color='green'
        )

    fig.update_layout(
        xaxis_rangeslider_visible=True,
        xaxis_title='Date',
        yaxis_title='Price ($)',
        title='Classified Results on SPY',
        width=1000,
        height=700,
    )
    
    st.plotly_chart(fig)
    
def plot_candlesticks(df_test, save_to_file=None):
    '''
    Plot candlesticks chart with interactive tooltips in altair
    '''
    # 'Date' is in datetime format
    df_test['Date'] = pd.to_datetime(df_test['Date'])
    
    # Group by consecutive 'pred=True' and get the first and last date for each group
    df_pattern = (
        df_test[df_test['pred']]
        .groupby((~df_test['pred']).cumsum())
        .agg(start=('Date', 'first'), end=('Date', 'last'))
    ).reset_index(drop=True)

    # Shift the end date by one to include the end day in the highlighted period
    df_pattern['end'] = df_pattern['end'] + pd.Timedelta(days=1)


    # Draws open-close bars with tooltips
    open_close_bar = alt.Chart(df_test).mark_bar().encode(
        alt.X('Date:T', axis=alt.Axis(title='Date')),
        alt.Y('Open:Q', axis=alt.Axis(title='Price', format='$'), scale=alt.Scale(zero=False)),
        alt.Y2('Close:Q'),
        color=alt.condition(
            "datum.Open <= datum.Close",
            alt.value("#06982d"),  # Green bar for rising price
            alt.value("#ae1325")   # Red bar for falling price
        ),
        tooltip=['Date:T', 'Open:Q', 'High:Q', 'Low:Q', 'Close:Q'] 
    )

    # Draws high and low vertical lines with tooltips
    high_low_rule = alt.Chart(df_test).mark_rule().encode(
        alt.Y('Low:Q'),
        alt.Y2('High:Q'),
        alt.X('Date:T'),
        tooltip=['Date:T', 'Open:Q', 'High:Q', 'Low:Q', 'Close:Q'] 
    )

    # Combine the open-close bars and high-low rules with interactive hover
    candlestick_chart = (open_close_bar + high_low_rule).interactive()

    # Green overlay for highlighted periods with df_pattern
    highlight = alt.Chart(df_pattern).mark_rect().encode(
        x='start:T',
        x2='end:T',
        color=alt.value('green'),
        opacity=alt.value(0.2)
    )

    final_chart = alt.layer(candlestick_chart, highlight).properties(
        width=900,  # This will make the width responsive to the container
        height=400  # You can adjust the height as needed
    ).configure_view(
        strokeWidth=0  # This removes the border around the chart
    )

    if save_to_file:
        save(final_chart, save_to_file)
    else:
        st.altair_chart(final_chart)