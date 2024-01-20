from sklearn.ensemble import RandomForestClassifier
from cache_utils import save_model_and_training_date, load_model
import streamlit as st
from datetime import datetime, timedelta

from .model_utils import BaseModel
import talib
from pycaret.classification import predict_model

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score

from sklearn.preprocessing import StandardScaler
import pandas as pd

profit_target = 0.04

def eval_metric_profit_labels(data):
    profit_labels = []
    for i in range(len(data)):
        if data['Signal'].iloc[i] == 1:
            buy_price = data['Close'].iloc[i]
            target_price = buy_price * (1 + profit_target)

            # Check if target price is met within the next 5 days
            subsequent_prices = data['Close'].iloc[i+1:i+6]
            if any(subsequent_prices >= target_price):
                profit_labels.append(1)
            else:
                profit_labels.append(0)
        else:
            profit_labels.append(0)
    return profit_labels
                            
                            
def create_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates feature columns in the given DataFrame.

    Parameters:
    - df: DataFrame
        The DataFrame to create the feature columns in.

    Returns:
    - df: DataFrame
        The DataFrame with the feature columns added.
    """
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
    return df


class RollingPrecisionRecallModel(BaseModel):
    def __init__(self, selected_option: str, train_until: str, data_source: str = 'yf'):
        super().__init__(selected_option, train_until, data_source)   
        self.model_file_name = f"{self.selected_option}_{self.__class__.__name__}"

    def run_train(self):
        df = self.download_train_data()
        results = self.prepare_model_train_data(df)
        
        df_train = results['df_train']
        df_test = results['df_test']
        
        final_model = self.train(df_train)

        # Predict on the test data
        predictions = predict_model(final_model, data=df_test)
    
        actual = predictions['target']
        predicted = predictions['prediction_label']
        st.markdown(f"Classification report: \n ```{classification_report(actual, predicted)}```")
        st.write("Confusion matrix: \n", confusion_matrix(actual, predicted))

        df_test['pred_prob'] = predicted.values
        df_test['pred'] = df_test['pred_prob'] == 1
        
        # Save the model
        save_model_and_training_date(self.model_file_name, final_model)

        return {
            'df_test': df_test
        } 
    
    def prepare_model_train_data(self, df):
        # Assuming df is your DataFrame with historical stock data
        df = create_feature_cols(df)
        df['Returns'] = df['Close'].pct_change().fillna(0)
        df['Volatility'] = df['Returns'].rolling(5).std()



        # Feature engineering for the primary model
        # (Simple features for demonstration purposes)
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df = df.dropna()

        # Create binary labels: 1 if next day's return is positive, 0 otherwise
        # df['target'] = (df['Returns'].shift(-1) > 0).astype(int)
        # Define the target variable
        df['target'] = df['price_above_ma'].astype(int).shift(-5)
        df.dropna(inplace=True)

        feat_cols = [col for col in df.columns if 'feat' in col]
        # Defining features and labels
        X = df[feat_cols]
        y = df['target']
        
        x_train = df[df['Date'] < self.train_until][feat_cols]
        y_train = df[df['Date'] < self.train_until]['target']

        x_test = df[df['Date'] >= self.train_until][feat_cols]
        y_test = df[df['Date'] >= self.train_until]['target']
        
        df_train = df[df['Date'] >= self.train_until]

        # Scaling features
        scaler = StandardScaler()
        x_train = x_train[feat_cols]
        x_test = x_test[feat_cols]
        X_train_scaled = scaler.fit_transform(x_train)
        X_test_scaled = scaler.transform(x_test)

        x_train.dropna(inplace=True)
        x_train.dropna(inplace=True)
        
        primary_model = self.train(x_train, y_train)
        metrics = self.evaluate(primary_model, x_train, y_train, x_test, y_test)
        
        
        # Define window size for rolling calculation
        window_size = 50  # for example, last 50 predictions

        # Initialize lists to store the rolling metrics
        rolling_precision = []
        rolling_recall = []

        data_scaled = df.copy()
        data_scaled[feat_cols] = scaler.fit_transform(df[feat_cols])

        X = data_scaled[feat_cols]
        y = data_scaled['target']

        predictions_data_scaled = primary_model.predict(X)

        # Loop through the validation dataset in windows
        for i in range(window_size, len(X)):
            # Define the windowed subsets
            y_true_window = y[i-window_size:i]
            y_pred_window = predictions_data_scaled[i-window_size:i]

            # Calculate precision and recall for the current window
            window_precision = precision_score(y_true_window, y_pred_window)
            window_recall = recall_score(y_true_window, y_pred_window)

            # Append to lists
            rolling_precision.append(window_precision)
            rolling_recall.append(window_recall)

        # Now rolling_precision and rolling_recall contain the rolling metrics

        # Convert lists to Series and align with the original DataFrame index
        rolling_precision_series = pd.Series([None]*window_size + rolling_precision, index=X.index)
        rolling_recall_series = pd.Series([None]*window_size + rolling_recall, index=X.index)
        meta_features = X[feat_cols]

        meta_features['primary_predictions'] = predictions_data_scaled
        # Combine rolling metrics with other features
        meta_features['rolling_precision'] = rolling_precision_series
        meta_features['rolling_recall'] = rolling_recall_series

        meta_features = meta_features[['primary_predictions', 'rolling_recall', 'rolling_precision']]
        data = pd.concat([df, meta_features], axis=1)
        
        data['primary_predictions'] = 0
        data['CorrectPrediction'] = (data['primary_predictions'] == data['target'])
        data['MetaLabel'] = data['CorrectPrediction'] &  (data['Returns'] > 0.005)
        
        meta_X = data.dropna()

        # Using chronological order for meta-model
        # meta_X_train, meta_X_test, meta_y_train, meta_y_test = meta_X.drop('MetaLabel', axis=1)[:split], meta_X.drop('MetaLabel', axis=1)[split:], meta_X['MetaLabel'][:split], meta_X['MetaLabel'][split:]
        meta_X_train = meta_X[meta_X['Date'] < self.train_until][feat_cols]
        meta_y_train = meta_X[meta_X['Date'] < self.train_until]['target']

        meta_X_test = meta_X[meta_X['Date'] >= self.train_until][feat_cols]
        meta_y_test = meta_X[meta_X['Date'] >= self.train_until]['target']
        
        meta_df_train = meta_X[meta_X['Date'] >= self.train_until]

        
        
        model_results = {
            'x_train': meta_X_train,
            'x_test': meta_X_test,
            'y_train': meta_y_train,
            'y_test': meta_y_test,
            'feat_cols': feat_cols,
            'df_train': meta_df_train
        }
        
        return model_results
        
    def train(self, x_train, y_train):
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42,
            class_weight='balanced'
        )
        clf.fit(x_train, y_train)
        return clf
    
    def run_train(self):
        df = self.download_train_data()
        results = self.prepare_model_train_data (df)
        
        df = results['df_train']
        x_train = results['x_train']
        y_train = results['y_train']
        x_test = results['x_test']
        y_test = results['y_test']
        feat_cols = results['feat_cols']
    
        

        # Additional processing and returning results
        df_test = df[df['Date'] >= self.train_until]
        meta_model = self.train(x_train, y_train)
        
        metrics = self.evaluate(meta_model, x_train, y_train, x_test, y_test)
        
        meta_model.fit(x_train, y_train)
        primary_predictions = meta_model.predict(x_test)


        df_test['pred_prob'] = primary_predictions
        df_test['pred'] = df_test['pred_prob'] > 0.5
        
        # Save the model
        save_model_and_training_date(self.model_file_name, meta_model)

        results = {
            'train_accuracy': metrics['train_accuracy'],
            'train_precision': metrics['train_precision'],
            'test_accuracy': metrics['test_accuracy'],
            'test_precision': metrics['test_precision'],
            'df_test': df_test,
            'y_train_pred': metrics['y_train_pred'],
            'y_test_pred': metrics['y_test_pred'],
            'x_train': x_train,
            'x_test': x_test,
            'clf': meta_model,
            'feat_cols': feat_cols,
            'y_test': y_test,
            'y_train': y_train
        }
        self.model_results = results
        self.plot()
        return results
    
    def run_test(self, selected_option: str, last_n_days: str, download_end_date: datetime=None, data_source: str = 'yf'):
        df2 = self.download_test_data(selected_option, last_n_days, download_end_date, data_source)
        df2 = self.prepare_model_test_data(df2)
        
        clf = load_model(self.model_file_name)
        current_date = datetime.now()

        # Subtract last_n_days days from the current date
        test_till = current_date - timedelta(days=int(last_n_days))
        
        if not download_end_date:
            download_end_date = current_date
            
        df_test = df2[(df2['Date'] > test_till) & (df2['Date'] < download_end_date)].reset_index(drop=True)
        
        # Predict on the test data
        predictions = predict_model(clf, data=df_test)
        predicted = predictions['prediction_label']
        df_test['pred_prob'] = predicted.values
        df_test['pred'] = df_test['pred_prob'] == 1
        
        return df_test
        
    def prepare_model_test_data(self, df_test):
        df_test['RSI'] = talib.RSI(df_test['Close'])
        df_test['MACD'], df_test['MACDSignal'], _ = talib.MACD(df_test['Close'])

        df_test = df_test.dropna()
        return df_test
    
    def plot(self):
        pass
        