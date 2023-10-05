# %%
import pandas as pd
from fancyimpute import IterativeImputer
import xgboost as xgb
import lightgbm as lgb
import catboost


def calculate_expected_arrival(row):
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

# Drop the first and last day to avoid missing values
train = train[(train["dt_dep"] >= "2022-06-03") & (train["dt_dep"] <= "2023-05-13")]

train = train.assign(
    origin_destiny_encode=lambda x: x["origin_destiny"].map(
        target_encoding_origin_destiny
    ),
    expected_arrival=lambda x: x.apply(calculate_expected_arrival, axis=1),
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
    ],
    axis=1,
)

# Drop extreme flight times
train = train[train["origin_destiny_encode"] * 2 > train["seconds_flying"]]

X_train, y_train = train.drop(["seconds_flying"], axis=1), train["seconds_flying"]

train[
    (train["expected_arrival"] > train["expected_arrival"][285087])
    & (
        train["expected_arrival"]
        < train["expected_arrival"][285087] + pd.Timedelta(minutes=15)
    )
]

# %%
X_test = pd.read_parquet("data/modelling/test_df.parquet")

flight_info = X_test[["flightid", "origem", "destino"]]

X_test = X_test.assign(
    origin_destiny=lambda x: x["origem"] + "_" + x["destino"],
    origin_destiny_encode=lambda x: x["origin_destiny"].map(
        target_encoding_origin_destiny
    ),
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
# model = lgb.LGBMRegressor(random_state=42)
# model = xgb.XGBRgressor(random_state=42)
model = catboost.CatBoostRegressor(random_state=42)

# Train the model on the training data
model.fit(X_train_input, y_train)

y_pred = model.predict(X_test_input)

# %%
flight_info["y_pred"] = y_pred

flight_info = (
    flight_info.assign(
        equal_airport=lambda x: pd.to_numeric(x["origem"] != x["destino"]),
        y_pred=lambda x: x["y_pred"] * x["equal_airport"],
    )
    .drop(["origem", "destino", "equal_airport"], axis=1)
    .rename({"flightid": "id", "y_pred": "solution"}, axis=1)
)

flight_info.to_csv("submission_catboost.csv", index=False)

# %%
