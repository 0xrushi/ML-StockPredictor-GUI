import pandas as pd

# Feature deriving
# Distance from the moving averages
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