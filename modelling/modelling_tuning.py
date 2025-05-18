import sys
import os
import time
import json
import mlflow
import dagshub
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import eli5
from eli5.sklearn import explain_weights

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    explained_variance_score, max_error
)
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# === Import preprocessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.automate_Sinta import SklearnPreprocessor

# === Load data
data = pd.read_csv("preprocessing/cleaned_data.csv")

X = data.drop(['Item_Outlet_Sales', 'Item_Identifier'], axis=1)
y = np.log(data['Item_Outlet_Sales'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# === Preprocessing pipeline
preprocessor = SklearnPreprocessor(
    num_columns=['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year'],
    ordinal_columns=['Item_Fat_Content', 'Outlet_Size'],
    nominal_columns=['Item_Type', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Identifier'],
    degree=2
)

# === DagsHub + MLflow config (TIDAK DIUBAH)
mlflow.set_tracking_uri("https://dagshub.com/Sintasitinuriah/my-first-repo.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = "Sintasitinuriah"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "adac45483360179ca37b590439e42bb1729c415e"

mlflow.set_experiment("Big Mart Sales Prediction Manual Logging")
dagshub.init(repo_owner='Sintasitinuriah', repo_name='my-first-repo', mlflow=True)

# === Models
models = [
    Pipeline([('preprocessor', preprocessor), ('LinearReg', LinearRegression())]),
    Pipeline([('preprocessor', preprocessor), ('Ridge', Ridge())]),
    Pipeline([('preprocessor', preprocessor), ('RandomForest', RandomForestRegressor())]),
    Pipeline([('preprocessor', preprocessor), ('XGBRegressor', XGBRegressor())])
]

# === Hyperparameters
params = [
    {'preprocessor__degree': [2]},
    {'preprocessor__degree': [2], 'Ridge__alpha': [1.0]},
    {'preprocessor__degree': [2], 'RandomForest__n_estimators': [50], 'RandomForest__max_depth': [10]},
    {'preprocessor__degree': [2], 'XGBRegressor__n_estimators': [50], 'XGBRegressor__learning_rate': [0.1], 'XGBRegressor__max_depth': [3]}
]

model_names = ['Linear_Regression', 'Ridge_Regression', 'Random_Forest', 'XGBoost']

# === Prepare output folder
os.makedirs("model", exist_ok=True)

# === Loop each model
for model, param_grid, model_name in zip(models, params, model_names):
    start_time = time.time()
    grid = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    train_time = time.time() - start_time

    y_valid_pred = best_model.predict(X_valid)

    # === Evaluation
    mae = mean_absolute_error(y_valid, y_valid_pred)
    mse = mean_squared_error(y_valid, y_valid_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_valid, y_valid_pred)
    explained_var = explained_variance_score(y_valid, y_valid_pred)
    max_err = max_error(y_valid, y_valid_pred)

    with mlflow.start_run(run_name=model_name):
        # === Logging Params
        mlflow.log_param("model_name", model_name)
        for param_name, param_val in grid.best_params_.items():
            mlflow.log_param(param_name, param_val)

        # === Logging Metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("explained_variance", explained_var)
        mlflow.log_metric("max_error", max_err)
        mlflow.log_metric("training_time_sec", train_time)

        # === Log Model to MLflow
        mlflow.sklearn.log_model(best_model, "model")

        # === Save estimator.html (if possible)
        try:
            estimator = best_model.named_steps[list(best_model.named_steps)[-1]]

            if isinstance(estimator, (LinearRegression, Ridge)):
                html_exp = eli5.format_as_html(explain_weights(estimator, feature_names=None))
                html_path = f"model/{model_name}_estimator.html"
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_exp)
                mlflow.log_artifact(html_path)
                print(f"[INFO] estimator.html generated for {model_name}")
            else:
                print(f"[INFO] estimator.html not supported for model type: {type(estimator).__name__}")

        except Exception as e:
            print(f"[ERROR] Failed to generate estimator.html for {model_name}: {e}")

        # === Save metrics.info.json
        metrics_path = f"model/{model_name}_metrics.info.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "mae": mae, "mse": mse, "rmse": rmse, "r2": r2,
                "explained_variance": explained_var,
                "max_error": max_err,
                "training_time_sec": train_time
            }, f, indent=4)
        mlflow.log_artifact(metrics_path)

        # === Save residual plot
        residuals = y_valid - y_valid_pred
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_valid, y=residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f"Residual Plot: {model_name}")
        plt.xlabel("Actual")
        plt.ylabel("Residual")
        plot_path = f"model/{model_name}_residuals.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)

    print(f"{model_name} DONE. R2: {r2:.4f}, MAE: {mae:.4f}, Time: {train_time:.2f}s")
