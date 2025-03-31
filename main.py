import argparse
import pandas as pd
from src.data_processing import load_data, read_data, clean_data, normalize_data
from src.ml_algorithms import predict_temperature, cluster_data, detect_anomalies
from src.visualization import plot_temperature, plot_heatmap, plot_scatter

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Climate Change Impact Analyzer")
    
    # Data manipulation
    parser.add_argument('--load', type=str, help='Path to the climate data CSV or JSON file', required=True)
    parser.add_argument('--clean', action='store_true', help='Clean the data (remove missing values, duplicates)')
    parser.add_argument('--normalize', action='store_true', help='Normalize the data')
    
    # Machine learning options
    parser.add_argument('--predict', action='store_true', help='Predict temperature using machine learning models')
    parser.add_argument('--cluster', action='store_true', help='Cluster data using machine learning algorithms')
    parser.add_argument('--anomaly', action='store_true', help='Detect anomalies in the climate data')
    
    # Visualization options
    parser.add_argument('--plot-temperature', action='store_true', help='Plot temperature over time')
    parser.add_argument('--plot-heatmap', action='store_true', help='Plot heatmap of correlations')
    parser.add_argument('--plot-scatter', action='store_true', help='Plot scatter plot between two variables')

    # Parse the arguments
    args = parser.parse_args()

    # Load the data from the provided file path
    try:
        df = load_data(args.load)
        print(f"Data loaded from {args.load}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Clean the data if --clean flag is provided
    if args.clean:
        print("Cleaning the data...")
        df = clean_data()   #(df)
    
    # Normalize the data if --normalize flag is provided
    if args.normalize:
        print("Normalizing the data...")
        df = normalize_data()   #(df)

    print("Raw data: ", read_data())
    
    # Predict temperature if --predict flag is provided
    if args.predict:
        print("Predicting temperature using machine learning model...")
        predictions = predict_temperature()   #(df)
        print("Predictions: ", predictions)

    # Perform clustering if --cluster flag is provided
    if args.cluster:
        print("Clustering the data...")
        cluster_results = cluster_data()   #(df)
        print("Cluster Results: ", cluster_results)

    # Detect anomalies if --anomaly flag is provided
    if args.anomaly:
        print("Detecting anomalies in the data...")
        anomalies = detect_anomalies()   #(df)
        print("Anomalies detected: ", anomalies)

    # Plot temperature over time if --plot-temperature flag is provided
    if args.plot_temperature:
        print("Plotting temperature over time...")
        plot_temperature()   #(df)

    # Plot heatmap of correlations if --plot-heatmap flag is provided
    if args.plot_heatmap:
        print("Plotting heatmap of correlations...")
        plot_heatmap()   #(df)

    # Plot scatter plot if --plot-scatter flag is provided
    if args.plot_scatter:
        print("Plotting scatter plot...")
        plot_scatter()   #(df)

if __name__ == "__main__":
    main()
