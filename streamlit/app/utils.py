# app/utils.py
import pandas as pd
import numpy as np
from scipy import stats

def fetch_data(file_path):
    """Load data from a CSV file and return a DataFrame."""
    df = pd.read_csv(file_path)
    return df

def process_data(df):
    """Process the DataFrame by handling missing values, outliers, and incorrect entries."""
    print("Missing values:")
    print(df.isnull().sum())
    
    df.dropna(inplace=True)  # Example: remove rows with missing values
    df.drop(columns=['Comments'], inplace=True)
    
    z_scores = stats.zscore(df[['GHI', 'DNI', 'DHI']])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df = df[filtered_entries]

    df.drop_duplicates(inplace=True)
    df[df < 0] = np.nan
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    return df