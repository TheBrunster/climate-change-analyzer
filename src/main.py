# main.py

from src.data_processor import DataProcessor
from src.algorithms import CustomTemperaturePredictor, custom_clustering, detect_anomalies
from src.visualizer import Visualizer

def main():
    data_processor = DataProcessor('climate_data.csv')
    data_processor.load_data()
    data_processor.clean_data()
    X, y = data_processor.get_features_and_target()

    # Temperature prediction
    model = CustomTemperaturePredictor()
    model.fit(X, y)
    predictions = model.predict(X)
    Visualizer.plot_temperature_trend(range(len(y)), y, predictions)

    # Clustering
    labels = custom_clustering(X, n_clusters=3)
    Visualizer.plot_clustered_data(X, labels)

    # Anomaly detection
    anomalies = detect_anomalies(y)
    Visualizer.plot_anomalies(y, anomalies)

if __name__ == "__main__":
    main()
