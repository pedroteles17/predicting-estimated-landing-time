# %%
import pandas as pd
import numpy as np
import duckdb
from utils import number_of_flights_expected, calculate_expected_arrival, get_image_clusters
from fancyimpute import IterativeImputer
import xgboost as xgb
import lightgbm as lgb
import catboost
import optuna
from tqdm import tqdm
import pandas._libs.missing as pd_missing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

tqdm.pandas()

# %%
import importlib
import utils
# %%
train = pd.read_parquet("data/modelling/train_df.parquet").assign(
    origin_destiny=lambda x: x["origem"] + "_" + x["destino"]
)

target_encoding_origin_destiny = train.groupby("origin_destiny")[
    "seconds_flying"
].mean()

# Drop the first and last day to avoid missing values
train = (
    train[(train["dt_dep"] >= "2022-06-03") & (train["dt_dep"] <= "2023-05-13")]
    .reset_index(drop=True)
    .assign(
        runway_length=lambda x: x["runway_length"].apply(lambda x: np.mean(x)),
        origin_destiny_encode=lambda x: x["origin_destiny"].map(
            target_encoding_origin_destiny
        ),
        expected_arrival=lambda x: x.apply(calculate_expected_arrival, axis=1),
    )
)

# To speed up the process, we use duckdb
duckdb.sql(
    "CREATE TABLE train_table AS SELECT estimated_departure, expected_arrival, destino FROM train"
)

train = (
    train.assign(
        number_flights_arriving=lambda x: x.progress_apply(
            lambda x: number_of_flights_expected(
                "train_table", x["destino"], x["expected_arrival"], (-30, 30), False
            )
            if not pd.isna(x["expected_arrival"])
            else pd.NA,
            axis=1,
        )
        .replace(pd_missing.NAType(), pd.NA)
        .astype("Int64"),
        number_flights_departing=lambda x: x.progress_apply(
            lambda x: number_of_flights_expected(
                "train_table", x["destino"], x["expected_arrival"], (-30, 30), True
            )
            if not pd.isna(x["expected_arrival"])
            else pd.NA,
            axis=1,
        )
        .replace(pd_missing.NAType(), pd.NA)
        .astype("Int64"),
        # Our target variable is the difference between the expected arrival (conditional mean) and the actual arrival
        excess_seconds_flying=lambda x: x["seconds_flying"]
        - x["origin_destiny_encode"],
    )
    .query("(origin_destiny_encode * 2) > seconds_flying")
    .drop(
        [
            "flightid",
            "origem",
            "destino",
            "origin_destiny",
            "dt_dep",
            "path",
            "unique_metar",
            "time",
            "image_date",
            "estimated_departure",
            "expected_arrival",
            "origin_destiny_encode",
            "seconds_flying",
        ],
        axis=1,
    )
)

X_train, y_train = (
    train.drop(["excess_seconds_flying"], axis=1),
    train["excess_seconds_flying"],
)

#%%
# Hyperparameter optimization

def hyperparameter_tuning_objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    gbm = lgb.LGBMRegressor(**params, importance_type="gain")
    gbm.fit(X_train_val_input, y_train_val)
    
    y_pred = gbm.predict(X_val_input)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    return rmse

## Get validation set
X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2)

## Imputer for train and validation
imputer = IterativeImputer(max_iter=10, random_state=42)
imputer.fit(X_train_val)

X_train_val_input = pd.DataFrame(imputer.transform(X_train_val))
X_train_val_input.columns = X_train_val.columns

X_val_input = pd.DataFrame(imputer.transform(X_val))
X_val_input.columns = X_val.columns

## Start hyperparameter optimization study
study = optuna.create_study(direction='maximize')
study.optimize(hyperparameter_tuning_objective, n_trials=1000)

#del X_train_val, X_val, y_train_val, y_val, X_train_val_input, X_val_input

# %%
X_test = pd.read_parquet("data/modelling/test_df.parquet")

X_test = X_test.assign(
    runway_length=lambda x: x["runway_length"].apply(lambda x: np.mean(x)),
    origin_destiny=lambda x: x["origem"] + "_" + x["destino"],
    origin_destiny_encode=lambda x: x["origin_destiny"].map(
        target_encoding_origin_destiny
    ),
    expected_arrival=lambda x: x.apply(calculate_expected_arrival, axis=1),
)

flight_info = X_test[["flightid", "origem", "destino", "origin_destiny_encode"]]

# To speed up the process, we use duckdb
duckdb.sql(
    "CREATE TABLE test_table AS SELECT estimated_departure, expected_arrival, destino FROM X_test"
)

X_test = X_test.assign(
    number_flights_arriving=lambda x: x.progress_apply(
        lambda x: number_of_flights_expected(
            "test_table", x["destino"], x["expected_arrival"], (-30, 30), False
        )
        if not pd.isna(x["expected_arrival"])
        else pd.NA,
        axis=1,
    )
    .replace(pd_missing.NAType(), pd.NA)
    .astype("Int64"),
    number_flights_departing=lambda x: x.progress_apply(
        lambda x: number_of_flights_expected(
            "test_table", x["destino"], x["expected_arrival"], (-30, 30), True
        )
        if not pd.isna(x["expected_arrival"])
        else pd.NA,
        axis=1,
    )
    .replace(pd_missing.NAType(), pd.NA)
    .astype("Int64"),
).drop(
    [
        "flightid",
        "origem",
        "destino",
        "origin_destiny",
        "dt_dep",
        "path",
        "unique_metar",
        "time",
        "image_date",
        "estimated_departure",
        "expected_arrival",
        "origin_destiny_encode",
    ],
    axis=1,
)

# %%
# Create an instance of the IterativeImputer class
imputer = IterativeImputer(max_iter=10, random_state=42)

# Fit the imputer to your data
X_train = X_train.replace({pd.NA: np.nan})
X_test = X_test.replace({pd.NA: np.nan})

imputer.fit(X_train)

# Transform your data to impute missing values
X_train_input = pd.DataFrame(imputer.transform(X_train))
X_train_input.columns = X_train.columns

X_test_input = pd.DataFrame(imputer.transform(X_test))
X_test_input.columns = X_test.columns

# %%
# Initialize the XGBoost Regressor
#model = lgb.LGBMRegressor(importance_type="gain", random_state=42)
#model = xgb.XGBRegressor(random_state=42)
model = catboost.CatBoostRegressor(random_state=42)

model.fit(X_train_input, y_train)

y_pred = model.predict(X_test_input)

# %%
flight_pred = flight_info.copy()
flight_pred["y_pred"] = y_pred

flight_pred = (
    flight_pred.assign(
        equal_airport=lambda x: pd.to_numeric(x["origem"] != x["destino"]),
        origin_destiny_encode=lambda x: x["origin_destiny_encode"].fillna(0),
        y_pred=lambda x: (x["y_pred"] + x["origin_destiny_encode"]) * x["equal_airport"],
    )
    .drop(["origem", "destino", "equal_airport", "origin_destiny_encode"], axis=1)
    .rename({"flightid": "id", "y_pred": "solution"}, axis=1)
)

flight_pred.to_csv("submission_catboost.csv", index=False)

# %%
