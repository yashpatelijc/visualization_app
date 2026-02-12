import os

# Set APPDATA to the home directory if it's not already set.
os.environ.setdefault('APPDATA', os.path.expanduser("~"))

from pandasgui import show



import pandas as pd
from pandasgui import show

def calculate_indicators(df):
    """
    Calculates various indicators including moving averages, ATR, range indicators, RSI, MACD, ADX,
    Bollinger Bands, CCI, Trend Strength, as well as new rolling high/low levels and their ATR-adjusted levels.
    
    Args:
        df (pd.DataFrame): Data containing 'Open', 'High', 'Low', 'Close'.
    
    Returns:
        pd.DataFrame: Data with added indicators.
    """
    # Ensure the data is sorted by date
    df = df.sort_values(by='Date')
    
    # Shift OHLC values to avoid lookahead bias
    shifted_data = {
        'High_shifted': df['High'].shift(1),
        'Low_shifted': df['Low'].shift(1),
        'Close_shifted_1': df['Close'].shift(1),
        'Close_shifted_2': df['Close'].shift(2),
    }
    df = pd.concat([df, pd.DataFrame(shifted_data)], axis=1)
    
    # Simple & Exponential Moving Averages
    sma_periods = [3, 5, 10, 12, 15, 20, 26, 30, 50, 100]
    ma_data = {}
    for period in sma_periods:
        ma_data[f'SMA_{period}'] = df['Close_shifted_1'].rolling(window=period).mean()
        ma_data[f'EMA_{period}'] = df['Close_shifted_1'].ewm(span=period, adjust=False).mean()
    df = pd.concat([df, pd.DataFrame(ma_data)], axis=1)
    
    # ATR Calculation (Average True Range)
    df['True_Range'] = df[['High_shifted', 'Low_shifted', 'Close_shifted_2']].apply(
        lambda x: max(x[0] - x[1], abs(x[0] - x[2]), abs(x[1] - x[2])), axis=1
    )
    atr_periods = [10, 20]
    atr_data = {f'ATR_{period}': df['True_Range'].rolling(window=period).mean() for period in atr_periods}
    df = pd.concat([df, pd.DataFrame(atr_data)], axis=1)
    
    # Range-Based Indicators & Ratios
    range_periods = [2, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 80, 90, 100, 120, 150, 160, 180, 200, 240, 300, 320, 360, 400, 500, 540, 720, 900, 1000]
    range_data = {f'Range_{period}': df['High_shifted'].rolling(window=period).max() - df['Low_shifted'].rolling(window=period).min() for period in range_periods}
    df = pd.concat([df, pd.DataFrame(range_data)], axis=1)
    
    range_ratios = {}
    for i in range(len(range_periods)):
        for j in range(i):
            range_ratios[f'Range_{range_periods[i]}_Range_{range_periods[j]}'] = df[f'Range_{range_periods[i]}'] / df[f'Range_{range_periods[j]}']
    df = pd.concat([df, pd.DataFrame(range_ratios)], axis=1)
    
    atr_ratios = {f'Range_{period}_ATR_{atr_period}': df[f'Range_{period}'] / df[f'ATR_{atr_period}'] for period in range_periods for atr_period in [10, 20]}
    df = pd.concat([df, pd.DataFrame(atr_ratios)], axis=1)
    
    # RSI Calculation
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    df['RSI_14'] = compute_rsi(df['Close_shifted_1'])
    
    # Additional Range Differences Divided by ATR_20
    diff_atr_ratios = {
        '(Range_150_Range_60)_ATR_20': (df['Range_150'] - df['Range_60']) / df['ATR_20'],
        '(Range_150_Range_30)_ATR_20': (df['Range_150'] - df['Range_30']) / df['ATR_20'],
        '(Range_120_Range_30)_ATR_20': (df['Range_120'] - df['Range_30']) / df['ATR_20'],
        '(Range_120_Range_60)_ATR_20': (df['Range_120'] - df['Range_60']) / df['ATR_20'],
        '(Range_90_Range_30)_ATR_20': (df['Range_90'] - df['Range_30']) / df['ATR_20'],
        '(Range_360_Range_120)_ATR_20': (df['Range_360'] - df['Range_120']) / df['ATR_20'],
        '(Range_300_Range_100)_ATR_20': (df['Range_300'] - df['Range_100']) / df['ATR_20'],
        '(Range_240_Range_80)_ATR_20': (df['Range_240'] - df['Range_80']) / df['ATR_20'],
        '(Range_60_Range_20)_ATR_20': (df['Range_60'] - df['Range_20']) / df['ATR_20'],
        '(Range_180_Range_60)_ATR_20': (df['Range_180'] - df['Range_60']) / df['ATR_20']
    }
    df = pd.concat([df, pd.DataFrame(diff_atr_ratios)], axis=1)
    
    # MACD Calculation
    df['MACD_Line'] = df['Close_shifted_1'].ewm(span=12, adjust=False).mean() - df['Close_shifted_1'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    
    # ADX Calculation
    def compute_adx(df, period=14):
        plus_dm = df['High_shifted'].diff()
        minus_dm = df['Low_shifted'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        atr = df['True_Range'].rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (abs(minus_dm.rolling(window=period).mean()) / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return dx.rolling(window=period).mean()
    df['ADX_14'] = compute_adx(df)
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close_shifted_1'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['Close_shifted_1'].rolling(window=20).std())
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['Close_shifted_1'].rolling(window=20).std())
    
    # CCI Calculation
    typical_price = (df['High_shifted'] + df['Low_shifted'] + df['Close_shifted_1']) / 3
    mean_deviation = lambda x: (x - x.mean()).abs().mean()
    df['CCI_20'] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * typical_price.rolling(window=20).apply(mean_deviation))
    
    # Corrected Trend Strength Calculation for Multiple Periods
    trend_settings = [(20, 3), (20, 5), (10, 2), (10, 3), (50, 5), (50, 10), (100, 10), (100, 20)]
    for trend_period, high_low_period in trend_settings:
        df[f'Long_Signal_{trend_period}_{high_low_period}'] = (df['Close_shifted_1'] > df['Close_shifted_1'].shift(1).rolling(window=high_low_period).max()).astype(int)
        df[f'Short_Signal_{trend_period}_{high_low_period}'] = (df['Close_shifted_1'] < df['Close_shifted_1'].shift(1).rolling(window=high_low_period).min()).astype(int) * -1
        df[f'Long_Trend_{trend_period}_{high_low_period}'] = df[f'Long_Signal_{trend_period}_{high_low_period}'].rolling(window=trend_period).sum()
        df[f'Short_Trend_{trend_period}_{high_low_period}'] = df[f'Short_Signal_{trend_period}_{high_low_period}'].rolling(window=trend_period).sum()
        df[f'Overall_Trend_{trend_period}_{high_low_period}'] = df[f'Long_Trend_{trend_period}_{high_low_period}'] + df[f'Short_Trend_{trend_period}_{high_low_period}']
    
    # -------------------------------
    # New Indicators: Rolling Highs/Lows and ATR-adjusted Levels
    # -------------------------------
    
    # Define the rolling window sizes for highs and lows
    rolling_periods = [5, 10, 20, 50]
    # Define the ATR multipliers to adjust levels
    atr_multipliers = [1, 1.5, 2, 2.5, 3,3.5,4,5,6,7,8,9]
    
    # Calculate Rolling Highs and Adjusted Highs
    for period in rolling_periods:
        high_col = f'High_prev_{period}'
        df[high_col] = df['High_shifted'].rolling(window=period).max()
        for multiplier in atr_multipliers:
            col_name = f'{high_col}_minus_{multiplier}xATR20'
            df[col_name] = df[high_col] - (multiplier * df['ATR_20'])
    
    # Calculate Rolling Lows and Adjusted Lows
    for period in rolling_periods:
        low_col = f'Low_prev_{period}'
        df[low_col] = df['Low_shifted'].rolling(window=period).min()
        for multiplier in atr_multipliers:
            col_name = f'{low_col}_plus_{multiplier}xATR20'
            df[col_name] = df[low_col] + (multiplier * df['ATR_20'])
        
    # Classify volatility zones based on ATR (unchanged)
    def classify_volatility_zones(df):
        threshold = df['(Range_360_Range_120)_ATR_20'].quantile(0.50)
        df['Volatility_Zone'] = df['(Range_360_Range_120)_ATR_20'].apply(
            lambda x: 'High' if abs(x) < threshold else 'Low'
        )
        return df
    df = classify_volatility_zones(df)
    
    # -------------------------------
    # End of New Indicators
    # -------------------------------
    
    # Compute difference columns for all numeric columns in DataFrame
    def add_difference_columns(df, diff_periods=[2, 5, 10, 20, 50]):
        diff_data = {}
        numeric_cols = df.select_dtypes(include=['number']).columns  # Only process numeric columns
        for col in numeric_cols:
            for period in diff_periods:
                diff_data[f'{col}_DIFF_{period}D'] = df[col] - df[col].shift(period)
        df = pd.concat([df, pd.DataFrame(diff_data)], axis=1)
        return df
    
    # Add differences for all numeric columns
    df = add_difference_columns(df)
    
    # Compute shifted columns for all numeric columns in DataFrame
    def add_shifted_columns(df, diff_periods=[1,2]):
        diff_data = {}
        numeric_cols = df.select_dtypes(include=['number']).columns  # Only process numeric columns
        for col in numeric_cols:
            for period in diff_periods:
                diff_data[f'{col}_Shift_{period}D'] = df[col].shift(period)
        df = pd.concat([df, pd.DataFrame(diff_data)], axis=1)
        return df
    
    # Add differences for all numeric columns
    df = add_shifted_columns(df)
         
    
    # Drop helper columns
    df.drop(columns=['High_shifted', 'Low_shifted', 'Close_shifted_1', 'Close_shifted_2', 'True_Range'], inplace=True)
    
    # Round all numeric columns to 5 decimal places
    df = df.round(5)
    
    return df

# Example usage
if __name__ == "__main__":
    file_path = "C:/Users/yash.patel/Python Projects/backtesting_dashboard/data/NZD_240M.csv"
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = calculate_indicators(df)
    df.to_csv("C:/Users/yash.patel/Python Projects/backtesting_dashboard/data/processed_data.csv", index=False)
    
    # Display DataFrame in an interactive GUI
    show(df)
