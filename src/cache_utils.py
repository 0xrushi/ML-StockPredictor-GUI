import streamlit as st
import pandas as pd
import datetime
import os
import pickle

def save_model_and_training_date(stock, model):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Go one directory up from the script directory
    base_dir = os.path.dirname(script_dir)

    # Define the paths for models and cache directories relative to the base directory
    models_dir = os.path.join(base_dir, "trained_models")
    cache_dir = os.path.join(base_dir, "cache")

    # Define the paths for model file and training dates file
    model_path = os.path.join(models_dir, f"{stock}_model.pkl")
    training_dates_path = os.path.join(cache_dir, "training_dates.txt")

    # Save model
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    except IOError as e:
        print(f"Error saving model for {stock}: {e}")

    # Save training date
    try:
        with open(training_dates_path, "a") as file:
            file.write(f"{stock},{datetime.datetime.now()}\n")
    except IOError as e:
        print(f"Error writing training date for {stock}: {e}")

def load_model(stock):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go one directory up from the script directory
    base_dir = os.path.dirname(script_dir)
    # Define the path for the model directory relative to the base directory
    models_dir = os.path.join(base_dir, "trained_models")
    # Define the path for the model file
    model_path = os.path.join(models_dir, f"{stock}_model.pkl")

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except IOError as e:
        print(f"Error loading model for {stock}: {e}")
        model = None
    return model

def should_retrain(stock):
    model_path = f"../models/{stock}_model.pkl"
    if not os.path.exists("../cache/training_dates.txt") or not os.path.exists(model_path):
        return True
    
    with open("../cache/training_dates.txt", "r") as file:
        for line in file:
            saved_stock, date_str = line.strip().split(',')
            if saved_stock == stock:
                last_train_date = datetime.datetime.fromisoformat(date_str)
                return (datetime.datetime.now() - last_train_date).days > 30
    return True
