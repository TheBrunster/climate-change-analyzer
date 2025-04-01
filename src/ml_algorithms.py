import config
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def predict_temperature(df=config.data):
    X = df.drop('Temperature', axis=1)  # Features
    y = df['Temperature']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    return predictions

def cluster_data(df=config.data):
    return ["<cluster_data>"]

def detect_anomalies(df=config.data):
    return ["<anomalies>"]
