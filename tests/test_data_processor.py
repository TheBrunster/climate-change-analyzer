import unittest
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.data_processor = DataProcessor('test_data.csv')
        self.data_processor.load_data()
        self.data_processor.clean_data()

    def test_clean_data(self):
        cleaned_data = self.data_processor.clean_data()
        self.assertEqual(cleaned_data.isnull().sum().sum(), 0)  # Ensure no missing data

    def test_get_features_and_target(self):
        X, y = self.data_processor.get_features_and_target()
        self.assertEqual(X.shape[1], 2)  # Features: year and month
        self.assertEqual(y.shape[0], X.shape[0])  # Same number of samples

if __name__ == '__main__':
    unittest.main()
