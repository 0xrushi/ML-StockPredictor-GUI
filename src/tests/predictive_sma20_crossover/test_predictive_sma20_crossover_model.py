import unittest
import pandas as pd
from datetime import datetime
import sys
import os
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.predictive_sma20_crossover_model import PredictiveSma20CrossoverModel

# Sample DataFrame to mimic financial data
sample_data = pd.read_csv('src/tests/data/AAPL.csv').reset_index(drop=True)
sample_df = pd.DataFrame(sample_data)

@pytest.fixture
def model():
    return PredictiveSma20CrossoverModel('AAPL', '2021-12-02')

def test_prepare_model_train_data( model):
    # Call the method to be tested
    result = model.prepare_model_train_data(sample_df)

    assert 'target' in result['df_train'].columns
    assert all(col in result['df_train'].columns for col in result['feat_cols'])
    assert not result['x_train'].empty
    assert not result['y_train'].empty
    assert not result['x_test'].empty
    assert not result['y_test'].empty

if __name__ == '__main__':
    unittest.main()
