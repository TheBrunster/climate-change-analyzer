import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

class Visualizer:
    @staticmethod
    def plot_temperature_trend(years: List[int], temperatures: List[float], predictions: List[float]) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(years, temperatures, label='Actual')
        plt.plot(years, predictions, label='Predicted')
        plt.xlabel('Year')
        plt.ylabel('Temperature (normalized)')
        plt.title('Temperature Trend Over Time')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_clustered_data(data: List[Tuple[float, float]], labels: List[int]) -> None:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=[point[0] for point in data], y=[point[1] for point in data], hue=labels, palette="Set1", s=100)
        plt.xlabel('Year')
        plt.ylabel('Temperature (normalized)')
        plt.title('Clustered Data')
        plt.show()

    @staticmethod
    def plot_anomalies(time_series: List[float], anomalies: List[bool]) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(time_series, label='Temperature')
        plt.scatter(range(len(time_series)), time_series, c=anomalies, cmap='coolwarm', label='Anomalies', s=100)
        plt.xlabel('Time')
        plt.ylabel('Temperature (normalized)')
        plt.title('Anomalies Detection')
        plt.legend()
        plt.show()
