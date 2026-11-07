# mlflow_train.py - Train 3 models and log each one to MLflow

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Connect to the MLflow server and name the experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("house-price-project")

# Same data and split as train.py
data = pd.read_csv("houses.csv")
X = data[["rooms", "size_sqft"]]
y = data["price_lakh"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear": LinearRegression(),
    "Forest": RandomForestRegressor(random_state=42),
    "Tree": DecisionTreeRegressor(random_state=42),
}

# One MLflow run per model
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        mlflow.log_param("model_type", name)
        mlflow.log_metric("mae", mae)
        print(f"{name}: MAE = {mae:.2f}")