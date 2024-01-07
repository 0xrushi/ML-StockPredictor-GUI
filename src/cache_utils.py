import streamlit as st
import pandas as pd
import datetime
import os
import pickle

def save_model_and_training_date(stock, model):
    with open(f"models/{stock}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("cache/training_dates.txt", "a") as file:
        file.write(f"{stock},{datetime.datetime.now()}\n")

def should_retrain(stock):
    model_path = f"models/{stock}_model.pkl"
    if not os.path.exists("cache/training_dates.txt") or not os.path.exists(model_path):
        return True
    
    with open("cache/training_dates.txt", "r") as file:
        for line in file:
            saved_stock, date_str = line.strip().split(',')
            if saved_stock == stock:
                last_train_date = datetime.datetime.fromisoformat(date_str)
                return (datetime.datetime.now() - last_train_date).days > 30
    return True
