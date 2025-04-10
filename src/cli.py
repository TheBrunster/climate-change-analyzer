import argparse
from src.data_processor import DataProcessor
from src.algorithms import CustomTemperaturePredictor, custom_clustering, detect_anomalies
from src.visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser(description="Climate Change Analysis")
    parser.add_argument('action', choices=['predict', 'cluster', 'anomalies'], help="Action to perform")
    parser.add_argument('file_path', help="Path to the climate data CSV file")
    args = parser.parse_args()

    # Load and preprocess data
    data_processor = DataProcessor(args.file_path)
    data_processor.load_data()
    data_processor.clean_data()
    X, y = data_processor.get_features_and_target()

    if args.action == "predict":
        model = CustomTemperaturePredictor()
        model.fit(X, y)
        predictions = model.predict(X)
        Visualizer.plot_temperature_trend(range(len(y)), y, predictions)

    elif args.action == "cluster":
        labels = custom_clustering(X, n_clusters=3)
        Visualizer.plot_clustered_data(X, labels)

    elif args.action == "anomalies":
        anomalies = detect_anomalies(y)
        Visualizer.plot_anomalies(y, anomalies)

if __name__ == "__main__":
    main()
