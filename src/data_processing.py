import config
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    df = pd.read_csv(file_path)
    config.data = df
    return df

def read_data():
    return config.data

def clean_data(df):
    df = config.data
    df = df.dropna()  # Drop rows with missing values
    df = df.drop_duplicates()  # Drop duplicate rows
    config.data = df
    return df

def normalize_data(df):
    df = config.data
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=df.columns)
    config.data = df
    return df
