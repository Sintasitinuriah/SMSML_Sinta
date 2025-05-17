import sys
import os
import time
import mlflow
import dagshub
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, max_error
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.automate_Sinta import SklearnPreprocessor

# Load data
data = pd.read_csv("C:/Users/Sinta/Documents/Larskar AI/Submission/Membangun Machine Learning/Eksperimen_SML_Sinta-Siti-Nuriah/preprocessing/cleaned_data.csv")

X = data.drop(['Item_Outlet_Sales','Item_Identifier'], axis = 1)
y = np.log(data['Item_Outlet_Sales'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

preprocessor = SklearnPreprocessor(
    num_columns=['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year'],
    ordinal_columns=['Item_Fat_Content','Outlet_Size'],
    nominal_columns=['Item_Type','Outlet_Location_Type','Outlet_Type','Outlet_Identifier'],
    degree=2
)

# Set experiment
# Set MLflow ke DagsHub
mlflow.set_tracking_uri("https://dagshub.com/Sintasitinuriah/my-first-repo.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = "Sintasitinuriah"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "adac45483360179ca37b590439e42bb1729c415e"

mlflow.set_experiment("Big Mart Sales Prediction")
dagshub.init(repo_owner='Sintasitinuriah', repo_name='my-first-repo', mlflow=True)


# Model dan hyperparameter
models = [
    Pipeline([('preprocessor', preprocessor), ('LinearReg', LinearRegression())]),
    Pipeline([('preprocessor', preprocessor), ('Ridge', Ridge())]),
    Pipeline([('preprocessor', preprocessor), ('RandomForest', RandomForestRegressor())]),
    Pipeline([('preprocessor', preprocessor), ('XGBRegressor', XGBRegressor())])
]

params = [
    # Parameters for Linear Regression
    {'preprocessor__degree': [1,2, 3, 4, 5]},

    # Parameters for Ridge Regression
    {'preprocessor__degree': [1,2, 3, 4, 5],
     'Ridge__alpha': [0.01, 0.1, 1, 10, 100]},

    # Parameters for Random Forest
    {'preprocessor__degree': [1, 3],
     'RandomForest__n_estimators': [50, 100],
     'RandomForest__max_depth': [10, 20]},
    
    # XGBoost
    {'preprocessor__degree': [2, 3, 4, 5],
     'XGBRegressor__n_estimators': [50, 100],
     'XGBRegressor__learning_rate': [0.01, 0.1, 0.2],
     'XGBRegressor__max_depth': [3, 4, 5],
     'XGBRegressor__gamma': [0, 0.1, 0.2]}
]

grid_search = []
results = []
model_names = ['Linear_Regression', 'Ridge_Regression', 'Random_Forest', 'XGBoost']

mlflow.set_experiment("Big Mart Sales Prediction Manual Logging")

for model, param_grid, model_name in zip(models, params, model_names):
    start_time = time.time()
    grid = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    train_time = time.time() - start_time

    y_valid_pred = best_model.predict(X_valid)

    # Metrik dasar
    mae = mean_absolute_error(y_valid, y_valid_pred)
    mse = mean_squared_error(y_valid, y_valid_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_valid, y_valid_pred)

    # Metrik tambahan
    explained_var = explained_variance_score(y_valid, y_valid_pred)
    max_err = max_error(y_valid, y_valid_pred)

    with mlflow.start_run(run_name=model_name):
        # Logging parameter hasil GridSearchCV
        mlflow.log_param("model_name", model_name)
        for param_name, param_val in grid.best_params_.items():
            mlflow.log_param(param_name, param_val)

        # Logging metrik utama
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Logging metrik tambahan
        mlflow.log_metric("explained_variance", explained_var)
        mlflow.log_metric("max_error", max_err)
        mlflow.log_metric("training_time_sec", train_time)

        # Logging model
        mlflow.sklearn.log_model(best_model, "model")

    print(f"{model_name} done. R2: {r2:.4f}, Explained Var: {explained_var:.4f}, Max Error: {max_err:.4f}")