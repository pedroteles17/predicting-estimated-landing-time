# %%
import pandas as pd
import numpy as np
import duckdb
from utils import number_of_flights_expected
from fancyimpute import IterativeImputer
import xgboost as xgb
import lightgbm as lgb
import catboost
from tqdm import tqdm

tqdm.pandas()


def calculate_expected_arrival(row):
    if pd.isna(row["estimated_departure"]) or pd.isna(row["origin_destiny_encode"]):
        return pd.NA

    return row["estimated_departure"] + pd.Timedelta(
        seconds=row["origin_destiny_encode"]
    )

# %%
train = pd.read_parquet("data/modelling/train_df.parquet").assign(
    origin_destiny=lambda x: x["origem"] + "_" + x["destino"]
)

target_encoding_origin_destiny = train.groupby("origin_destiny")[
    "seconds_flying"
].mean()

# Add dumies for airport (destiny)
encoded_categories = pd.get_dummies(train['destino'], prefix='destino')
train = pd.concat([train, encoded_categories], axis=1)

# Drop the first and last day to avoid missing values
train = train[(train["dt_dep"] >= "2022-06-03") & (train["dt_dep"] <= "2023-05-13")]\
    .reset_index(drop=True)\
    .assign(
        runway_length = lambda x: x["runway_length"].apply(lambda x: np.mean(x)),
        origin_destiny_encode=lambda x: x["origin_destiny"].map(
            target_encoding_origin_destiny
        ),
        expected_arrival=lambda x: x.apply(calculate_expected_arrival, axis=1)
    )

# To speed up the process, we use duckdb
duckdb.sql("CREATE TABLE train_table AS SELECT * FROM train")

train = train\
    .assign(
        number_flights_arriving = lambda x: x.progress_apply(
            lambda x: number_of_flights_expected("train_table", x["destino"], x["expected_arrival"], (-30, 30), False) if not pd.isna(x["expected_arrival"]) else pd.NA,
            axis=1
        ),
        number_flights_departing = lambda x: x.progress_apply(
            lambda x: number_of_flights_expected("train_table", x["destino"], x["expected_arrival"], (-30, 30), True) if not pd.isna(x["expected_arrival"]) else pd.NA,
            axis=1
        )
    )\
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
        ],
        axis=1,
    )

# Drop extreme flight times
train = train[train["origin_destiny_encode"] * 2 > train["seconds_flying"]]

X_train, y_train = train.drop(["seconds_flying"], axis=1), train["seconds_flying"]

# %%
X_test = pd.read_parquet("data/modelling/test_df.parquet")

flight_info = X_test[["flightid", "origem", "destino"]]

# Add dumies for airport (destiny)
encoded_categories = pd.get_dummies(X_test['destino'], prefix='destino')
X_test = pd.concat([X_test, encoded_categories], axis=1)

X_test = X_test.assign(
    runway_length = lambda x: x["runway_length"].apply(lambda x: np.mean(x)),
    origin_destiny=lambda x: x["origem"] + "_" + x["destino"],
    origin_destiny_encode=lambda x: x["origin_destiny"].map(
        target_encoding_origin_destiny
    ),
    expected_arrival=lambda x: x.apply(calculate_expected_arrival, axis=1)
)

# To speed up the process, we use duckdb
duckdb.sql("CREATE TABLE test_table AS SELECT * FROM X_test")

X_test = X_test\
    .assign(
        number_flights_arriving = lambda x: x.progress_apply(
            lambda x: number_of_flights_expected("test_table", x["destino"], x["expected_arrival"], (-30, 30), False) if not pd.isna(x["expected_arrival"]) else pd.NA,
            axis=1
        ),
        number_flights_departing = lambda x: x.progress_apply(
            lambda x: number_of_flights_expected("test_table", x["destino"], x["expected_arrival"], (-30, 30), True) if not pd.isna(x["expected_arrival"]) else pd.NA,
            axis=1
        )
    )\
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
        ],
        axis=1,
    )

# %%
# Create an instance of the IterativeImputer class
imputer = IterativeImputer(max_iter=10, random_state=42)

# Fit the imputer to your data
imputer.fit(X_train)

# Transform your data to impute missing values
X_train_input = pd.DataFrame(imputer.transform(X_train))
X_train_input.columns = X_train.columns

X_test_input = pd.DataFrame(imputer.transform(X_test))
X_test_input.columns = X_test.columns

# %%
# Initialize the XGBoost Regressor
model = lgb.LGBMRegressor(
    random_state=42, learning_rate=0.1,
    n_estimators=500, reg_alpha=0.05, reg_lambda=0.05,
    importance_type="gain"
)
# model = xgb.XGBRgressor(random_state=42)
#model = catboost.CatBoostRegressor(random_state=42)

# Train the model on the training data
model.fit(X_train_input, y_train)

y_pred = model.predict(X_test_input)

# %%
flight_pred = flight_info.copy()
flight_pred["y_pred"] = y_pred

flight_pred = (
    flight_pred.assign(
        equal_airport=lambda x: pd.to_numeric(x["origem"] != x["destino"]),
        y_pred=lambda x: x["y_pred"] * x["equal_airport"],
    )
    .drop(["origem", "destino", "equal_airport"], axis=1)
    .rename({"flightid": "id", "y_pred": "solution"}, axis=1)
)

flight_pred.to_csv("submission_lightgbm.csv", index=False)

# %%
import pandas._libs.missing as pd_missing
column_names = encoded_categories.columns.tolist()
column_names = ["number_flights_departing", "number_flights_arriving"]
X_test[column_names] = X_test[column_names].replace(pd_missing.NAType(), pd.NA).astype("Int64")
# %%
