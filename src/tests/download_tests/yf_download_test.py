import os
import pandas as pd
from unittest.mock import patch, mock_open
import pytest
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import utils

@pytest.fixture
def mock_yf_download():
    # Mock yf.download to return a sample DataFrame
    def _mock_yf_download(*args, **kwargs):
        data = {
            'Date': pd.date_range(start="2021-01-01", periods=3, freq='D'),
            'Open': [100, 101, 102],
            'Close': [110, 111, 112],
        }
        return pd.DataFrame(data)

    with patch('utils.yf.download', side_effect=_mock_yf_download) as mock:
        yield mock

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Date': pd.date_range(start="2020-12-29", periods=3, freq='D'),
        'Open': [95, 96, 97],
        'Close': [105, 106, 107],
    })

def test_my_yf_download_new_download(mock_yf_download, sample_data):
    ticker = "AAPL"
    cache_dir = "src/tests/data/tempdata"
    end_date = "2021-01-03"

    # Mock os.path.exists to control whether the cache file exists
    with patch('utils.os.path.exists', side_effect=lambda path: False):
        # Mock os.makedirs to avoid creating actual directories
        with patch('utils.os.makedirs'):
            # Mock open to avoid actual file operations
            with patch('utils.open', mock_open(), create=True):
                # Call the function
                df = utils.my_yf_download(ticker, cache_dir, end_date)

    assert not df.empty
    assert all(col in df.columns for col in ['Date', 'Open', 'Close'])
    assert df['Date'].dtypes == 'datetime64[ns]'

def test_my_yf_download_existing_cache(mock_yf_download, sample_data):
    ticker = "AAPL"
    cache_dir = "src/tests/data/tempdata"
    end_date = "2021-01-03"
    
    # Prepare a sample cached file content
    sample_csv_content = sample_data.to_csv(index=False)

    # Mock os.path.exists to simulate the existence of a cached file
    with patch('utils.os.path.exists', side_effect=lambda path: True):
        # Mock open to provide the sample cached data
        with patch('utils.open', mock_open(read_data=sample_csv_content), create=True):
            # Call the function
            df = utils.my_yf_download(ticker, cache_dir, end_date)

    assert not df.empty
    assert all(col in df.columns for col in ['Date', 'Open', 'Close'])
    assert df['Date'].dtypes == 'datetime64[ns]'
