# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 10:17:07 2025

@author: yash.patel
"""

import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(file_path):
    """
    Loads data from a CSV file and performs error checks, reporting the exact locations of issues.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned data if no critical errors are found.
    """
    try:
        # Load CSV
        df = pd.read_csv(
            file_path,
            parse_dates=['Date'],
            infer_datetime_format=True,  # pandas will try to pick the right format
            low_memory=False
        )
        # Enforce exact format and catch bad rows if any
        df['Date'] = pd.to_datetime(
            df['Date'],
            format='%m/%d/%Y %H:%M',
            errors='raise'  # or 'coerce' to get NaT on bad parses
        )


        # Convert column names to standard format
        df.columns = df.columns.str.strip().str.title()  # Handles inconsistent column casing
        
        # Required columns
        required_columns = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            missing_info = df[df.isnull().any(axis=1)]
            logging.warning(f"Missing values detected in columns:\n{missing_values[missing_values > 0]}")
            logging.warning(f"Rows with missing values:\n{missing_info}")
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)

        # Ensure `Volume` column is numeric
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        non_numeric_volume = df[df['Volume'].isnull()]
        if not non_numeric_volume.empty:
            logging.warning("Non-numeric values detected in Volume column. Converting to 0.")
            logging.warning(f"Affected rows:\n{non_numeric_volume}")
            df['Volume'].fillna(0, inplace=True)

        # Check for duplicate rows
        duplicate_rows = df[df.duplicated()]
        if not duplicate_rows.empty:
            logging.warning(f"Duplicate rows detected:\n{duplicate_rows}")
            df.drop_duplicates(inplace=True)

        # Check if timestamps are in order
        if not df['Date'].is_monotonic_increasing:
            disorder_index = df[df['Date'] < df['Date'].shift(1)]
            raise ValueError(f"Dates are not in ascending order. Issue found at:\n{disorder_index}")

        logging.info("Data loaded successfully with no critical errors.")
        return df

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except pd.errors.ParserError:
        logging.error("Error parsing CSV file. Please check the format.")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

# Example usage
if __name__ == "__main__":
    file_path = "C:/Users/yash.patel/Python Projects/EUR_daily.csv"
    data = load_data(file_path)
    if data is not None:
        print(data.head())
