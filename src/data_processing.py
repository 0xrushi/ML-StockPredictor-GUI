import pandas as pd

# Feature deriving
# Distance from the moving averages
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

def check_if_today_starts_with_vertical_green_overlay(df_test: pd.DataFrame) -> bool:
    """
    Checks if today starts with a vertical green overlay in the given DataFrame.

    Parameters:
        df_test (DataFrame): The DataFrame to check.

    Returns:
        bool: True if today starts with a vertical green overlay, False otherwise.
    """
    df_pattern = (
        df_test[df_test['pred']]
        .groupby((~df_test['pred']).cumsum())
        ['Date']
        .agg(['first', 'last'])
    )
    
    last_date_included = False
    for idx, row in df_pattern.iterrows():
        if df_test.iloc[-1]['Date'] >= row['first'] and df_test.iloc[-1]['Date'] <= row['last'] and (row['last']!=row['first']):
            last_date_included = True
    
    # Check if the last entry has pred as True and is not included in any green fill
    if df_test.iloc[-1]['pred'] and not last_date_included:
        return True

    return False