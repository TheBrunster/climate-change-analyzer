import unittest
import numpy as np
from src.algorithms import CustomTemperaturePredictor, custom_clustering, detect_anomalies

class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        self.y = np.array([1, 2, 3, 4, 5])

    def test_temperature_predictor(self):
        model = CustomTemperaturePredictor()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

    def test_custom_clustering(self):
        labels = custom_clustering(self.X, n_clusters=2)
        self.assertEqual(len(labels), len(self.X))

    def test_anomalies_detection(self):
        anomalies = detect_anomalies(self.y)
        self.assertEqual(len(anomalies), len(self.y))

if __name__ == '__main__':
    unittest.main()
