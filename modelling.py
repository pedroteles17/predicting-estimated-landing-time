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

def hyperparameter_tuning_objective(trial: optuna.Trial) -> float:
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "l2_leaf_reg": trial.suggest_float('l2_leaf_reg', 1.0, 8),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        "eval_metric": "RMSE"
    }

    gbm = catboost.CatBoostRegressor(**params, silent=True)

    gbm.fit(X_train_val_input, y_train_val)

    preds = gbm.predict(X_val_input)
    rmse = mean_squared_error(y_val, preds, squared=False)

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
study = optuna.create_study(direction="minimize")
study.optimize(hyperparameter_tuning_objective, n_trials=50)

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

X_test = X_test[X_train.columns]

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
model = catboost.CatBoostRegressor()

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
