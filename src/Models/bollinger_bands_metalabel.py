from sklearn.ensemble import RandomForestClassifier
from cache_utils import save_model_and_training_date, load_model
import streamlit as st
from datetime import datetime, timedelta

from .model_utils import BaseModel
import talib
from pycaret.classification import predict_model

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler

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
                            


class BollingerBandsMetalabel(BaseModel):
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
        # Calculate Bollinger Bands
        df['Middle Band'] = df['Close'].rolling(window=15).mean()
        df['Upper Band'] = df['Middle Band'] + 2*df['Close'].rolling(window=15).std()
        df['Lower Band'] = df['Middle Band'] - 2*df['Close'].rolling(window=15).std()

        # Generate signals
        df['Signal'] = 0
        df.loc[df['Close'] < df['Lower Band'], 'Signal'] = 1  # Buy signal
        
        # Eval metric based on profitability
        

        df['target'] = eval_metric_profit_labels(df)
        df = df.dropna()

        date_format = "%Y-%m-%d"
        feat_cols = [ 'Signal', 'Middle Band', 'Upper Band', 'Lower Band', 'Close']
        X = df[['Date', 'Signal', 'Middle Band', 'Upper Band', 'Lower Band', 'Close']]
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
        
        
        model_results = {
            'x_train': X_train_scaled,
            'x_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feat_cols': ['Signal', 'Middle Band', 'Upper Band', 'Lower Band', 'Close'],
            'df_train': df_train
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
    
        meta_model = self.train(x_train, y_train)
    
        metrics = self.evaluate(meta_model, x_train, y_train, x_test, y_test)

        # Additional processing and returning results
        df_test = df[df['Date'] >= self.train_until]
        
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
        