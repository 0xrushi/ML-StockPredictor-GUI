from cache_utils import save_model_and_training_date, load_model
import streamlit as st
from datetime import datetime, timedelta

from sklearn.metrics import accuracy_score, precision_score
from utils import my_yf_download, my_nse_download

class BaseModel:
    def __init__(self, selected_option: str, train_until: str, data_source: str = 'yf'):
        self.selected_option = selected_option
        self.train_until = train_until
        self.data_source = data_source
        self.model_results = {}

    def download_train_data(self):
        if self.data_source == 'yf':
            df = my_yf_download(self.selected_option).reset_index(drop=True)
        elif self.data_source == 'nse':
            df = my_nse_download(self.selected_option).reset_index(drop=True)
        return df
    
    def download_test_data(self, selected_option, last_n_days, download_end_date, data_source: str = 'yf'):
        if download_end_date is not None:
            if data_source == 'yf':
                df2 = my_yf_download(selected_option, end=download_end_date).reset_index(drop=True)
            elif data_source == 'nse':
                df2 = my_nse_download(selected_option, end=download_end_date).reset_index(drop=True)
        else:
            if data_source == 'yf':
                df2 = my_yf_download(selected_option).reset_index(drop=True)
            elif data_source == 'nse':
                df2 = my_nse_download(selected_option).reset_index(drop=True)
        return df2

    def prepare_model_train_data(self, df):
        raise NotImplementedError("This method should be overridden in the subclass")
    def prepare_model_test_data(self, df_test):
        raise NotImplementedError("This method should be overridden in the subclass")
    
    def train(self, x_train, y_train):
        raise NotImplementedError("This method should be overridden in the subclass")
    def test(self, selected_option: str, last_n_days: str, download_end_date: datetime=None):
        raise NotImplementedError("This method should be overridden in the subclass")

    def evaluate(self, clf, x_train, y_train, x_test, y_test):
        y_train_pred = clf.predict(x_train)
        y_test_pred = clf.predict(x_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)

        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        
        model_results =  {
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'y_test_pred': y_test_pred,
            'y_train_pred': y_train_pred
        }
        
        st.write(f'Training Accuracy: {model_results["train_accuracy"]}')
        st.write(f'Training Precision: {model_results["train_precision"]}')
        st.write('')
        st.write(f'Test Accuracy: {model_results["test_accuracy"]}')
        st.write(f'Test Precision: {model_results["test_precision"]}')
        
        return model_results

    def run_train(self):
        df = self.download_train_data()
        self.train_until= df['Date'].min() + 0.9 * (df['Date'].max() - df['Date'].min() )
        print("\n\n\n\ntrain_until : ",self.train_until)
        results = self.prepare_model_train_data (df)
        
        df = results['df_train']
        x_train = results['x_train']
        y_train = results['y_train']
        x_test = results['x_test']
        y_test = results['y_test']
        feat_cols = results['feat_cols']
    
        clf = self.train(x_train, y_train)
    
        metrics = self.evaluate(clf, x_train, y_train, x_test, y_test)

        # Additional processing and returning results
        df_test = df[df['Date'] >= self.train_until]

        df_test['pred_prob'] = clf.predict_proba(x_test)[:, 1]
        df_test['pred'] = df_test['pred_prob'] > 0.4
        
        # Save the model
        save_model_and_training_date(self.model_file_name, clf)

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
            'clf': clf,
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

        return df_test
    
    
class BaseModel2:
    def __init__(self, selected_option: str, train_until: str, data_source: str = 'yf'):
        self.selected_option = selected_option
        self.train_until = train_until
        self.data_source = data_source
        self.model_results = {}

    def download_train_data(self):
        if self.data_source == 'yf':
            df = my_yf_download(self.selected_option).reset_index(drop=True)
        elif self.data_source == 'nse':
            df = my_nse_download(self.selected_option).reset_index(drop=True)
        return df
    
    def download_test_data(self, selected_option, last_n_days, download_end_date, data_source: str = 'yf'):
        if download_end_date is not None:
            if data_source == 'yf':
                df2 = my_yf_download(selected_option, end=download_end_date).reset_index(drop=True)
            elif data_source == 'nse':
                df2 = my_nse_download(selected_option, end=download_end_date).reset_index(drop=True)
        else:
            if data_source == 'yf':
                df2 = my_yf_download(selected_option).reset_index(drop=True)
            elif data_source == 'nse':
                df2 = my_nse_download(selected_option).reset_index(drop=True)
        return df2

    def prepare_model_train_data(self, df):
        raise NotImplementedError("This method should be overridden in the subclass")
    def prepare_model_test_data(self, df_test):
        raise NotImplementedError("This method should be overridden in the subclass")

    def run_train(self):
        raise NotImplementedError("This method should be overridden in the subclass")
    def run_test(self, selected_option: str, last_n_days: str, download_end_date: datetime=None, data_source: str = 'yf'):
        raise NotImplementedError("This method should be overridden in the subclass")