import sys, os, time
import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error

# Path preprocessing class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.automate_Sinta import SklearnPreprocessor

# Set MLflow ke DagsHub
mlflow.set_tracking_uri("https://dagshub.com/Sintasitinuriah/my-first-repo.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = "Sintasitinuriah"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "adac45483360179ca37b590439e42bb1729c415e"

mlflow.set_experiment("Big Mart Sales Prediction")
dagshub.init(repo_owner='Sintasitinuriah', repo_name='my-first-repo', mlflow=True)

# Load data
data = pd.read_csv("preprocessing/cleaned_data.csv")
X = data.drop(['Item_Outlet_Sales','Item_Identifier'], axis=1)
y = np.log(data['Item_Outlet_Sales'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Preprocessor & Pipeline
preprocessor = SklearnPreprocessor(
    num_columns=['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year'],
    ordinal_columns=['Item_Fat_Content','Outlet_Size'],
    nominal_columns=['Item_Type','Outlet_Location_Type','Outlet_Type','Outlet_Identifier'],
    degree=2
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", LinearRegression())
])

# Training & Logging
with mlflow.start_run(run_name="ManualLog - LinearRegression"):
    start = time.time()
    pipeline.fit(X_train, y_train)
    end = time.time()

    y_pred = pipeline.predict(X_test)

    # METRICS
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Tambahan
    explained_var = explained_variance_score(y_test, y_pred)
    max_err = max_error(y_test, y_pred)
    training_time = end - start

    # LOGGING MANUAL
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("degree_poly", 2)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("Explained_Variance", explained_var)
    mlflow.log_metric("Max_Error", max_err)
    mlflow.log_metric("Training_Time", training_time)

    # Logging model
    mlflow.sklearn.log_model(pipeline, "model")

    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    print(f"Explained Variance: {explained_var:.4f}, Max Error: {max_err:.4f}, Training Time: {training_time:.2f}s")