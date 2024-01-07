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

def plot_confusion_matrix(y_true, y_pred, title, normalize):

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

def get_precision_curve(clf, x, y, title):
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

def plot_feature_importances(clf):
    feature_importances = clf.feature_importances_
    sorted_indices = feature_importances.argsort()[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances in The Classifier")
    plt.bar(range(x_train.shape[1]), feature_importances[sorted_indices], align='center')
    plt.xticks(range(x_train.shape[1]), x_train.columns[sorted_indices], rotation=90)
    plt.ylabel('Importance')
    
    # Display the plot in Streamlit
    st.pyplot(plt)

def plot_candlesticks(df_test):
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

    for idx, row in df_pattern.iterrows():
        fig.add_vrect(
            x0=row['first'],
            x1=row['last'],
            line_width=0,
            fillcolor='green',
            opacity=0.2,
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