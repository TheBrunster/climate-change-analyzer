import config
import matplotlib.pyplot as plt

def plot_temperature(df=config.data):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Temperature'], label='Temperature')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Change Over Time')
    plt.legend()
    plt.show()
