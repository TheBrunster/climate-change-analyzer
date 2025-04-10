import pandas as pd
import numpy as np
from typing import Tuple

class DataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """Load climate data from CSV file."""
        self.data = pd.read_csv(self.file_path)
        return self.data

    def clean_data(self) -> pd.DataFrame:
        """Remove rows with missing values and normalize temperature data."""
        self.data = self.data.dropna()  # Remove rows with missing values
        self.data['temperature'] = (self.data['temperature'] - self.data['temperature'].mean()) / self.data['temperature'].std()  # Normalize temperature
        return self.data

    def get_features_and_target(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split data into features (year, month) and target (temperature)."""
        X = self.data[['year', 'month']].values  # Features: year and month
        y = self.data['temperature'].values  # Target: temperature
        return X, y
