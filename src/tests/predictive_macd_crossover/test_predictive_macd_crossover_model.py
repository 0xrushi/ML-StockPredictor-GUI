import unittest
import pandas as pd
from datetime import datetime
import sys
import os
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.predictive_macd_crossover_model import PredictiveMacdCrossoverModel
from unittest.mock import patch
import pandas as pd
from utils import my_yf_download

def mock_my_yf_download(self, *args, **kwargs):
    return pd.read_csv('src/tests/data/AAPL.csv').reset_index(drop=True)

class TestPredictiveMacdCrossoverModel(unittest.TestCase):
    def setUp(self):
        # Initialize with some test parameters
        self.model = PredictiveMacdCrossoverModel('AAPL', '2019-01-01')
        
        self.tempdf = pd.read_csv('src/tests/data/AAPL.csv').reset_index(drop=True)
        self.tempdf['Date'] = pd.to_datetime(self.tempdf['Date'])

    def test_prepare_model_train_data(self):
        # Test the prepare_model_train_data method
        result = self.model.prepare_model_train_data(self.tempdf)
        self.assertIn('RSI', result['df_train'].columns)
        self.assertIn('MACD', result['df_train'].columns)
        self.assertIn('MACDSignal', result['df_train'].columns)
        self.assertIn('target', result['df_train'].columns)

    def test_train(self):
        # Test the train method (may need to mock certain parts)
        train_result = self.model.prepare_model_train_data(self.tempdf)
        df_train = train_result['df_train']
        trained_model = self.model.train(df_train)
        # Depending on the output of train, you can assert certain conditions
        self.assertIsNotNone(trained_model)

    @patch('utils.my_yf_download', side_effect=mock_my_yf_download)
    def test_run_train(self, mock_my_yf_download):
        # Test the run_train method (may need to mock download_train_data and other dependencies)
        # Mock the data fetching and training parts as needed
        result = self.model.run_train()
        self.assertIn('df_test', result)

    @patch.object(PredictiveMacdCrossoverModel, 'download_test_data')
    def test_run_test(self, mock_download_test_data):
        # Test the run_test method (may need to mock download_test_data and other dependencies)
        # Create a mock dataframe for testing
        mock_download_test_data.return_value = self.tempdf
        result = self.model.run_test('AAPL', '30')
        self.assertIn('pred', result.columns)

    def test_prepare_model_test_data(self):
        df_test_processed = self.model.prepare_model_test_data(self.tempdf)
        self.assertIn('RSI', df_test_processed.columns)
        self.assertIn('MACD', df_test_processed.columns)
        self.assertIn('MACDSignal', df_test_processed.columns)

if __name__ == '__main__':
    unittest.main()
