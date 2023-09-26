#%%
import pandas as pd
from utils import cyclical_features_to_sin_cos, parse_metars
from tqdm import tqdm  # Progress bar

#%%
clean_data = pd.read_parquet("data/feature_engineering/clean_data.parquet")

clean_data = clean_data\
    .assign(
        tcr = lambda x: x["hora_tcr"].apply(lambda x: 0 if pd.isna(x) else 1), # Troca de Cabeceira Real
        tcp = lambda x: x["troca"].fillna(0), # Troca de Cabeceira Prevista
        esperas = lambda x: x["esperas"].fillna(0),
        dt_arr = lambda x: pd.to_datetime(x["dt_arr"], unit="ms"),
        seconds_flying = lambda x: (x["dt_arr"] - x["dt_dep"]).dt.total_seconds(),
        unique_metar = lambda x: x["metaf"].combine_first(x["metar"]),
        is_forecast = lambda x: x["metaf"].apply(lambda x: 0 if pd.isna(x) else 1),
    )\
    .assign(
        unique_metar = lambda x: x["unique_metar"].apply(lambda x: x.strip("\n").strip("=") if not pd.isna(x) else x),
    )\
    .drop([
        "hora_tcr", "aero_tcr",
        "hora_tcp", "aero_tcp",
        "hora_metar", "aero_metar",
        "hora_metaf", "aero_metaf",
        "hora_esperas", "aero_esperas",
        "hora_ref", "dt_arr",
        "metar", "metaf", "troca"
        ], axis=1
    )

#%%
# Parse METARs
metar_strings = list(clean_data["unique_metar"].dropna().unique())

parsed_metar = parse_metars(metar_strings)\
    .drop([
        "station", "type", "wind", "sky", "METAR", # useless
        "weather", "visual range" # Too many NaNs
        ], axis=1)\
    .rename(columns={"dew point": "dew_point"})
 
clean_data = clean_data\
    .merge(parsed_metar, left_on="unique_metar", right_on="original_metar", how="left")\
    .drop(["original_metar","wind_direction", "wind_speed"], axis=1)

#%%
# Transform timestamps to cyclical features (sin, cos)
clean_data[["minute_sin", "minute_cos"]] = list(clean_data["dt_dep"].apply(lambda x: cyclical_features_to_sin_cos(x.minute, 60)))
clean_data[["hour_sin", "hour_cos"]] = list(clean_data["dt_dep"].apply(lambda x: cyclical_features_to_sin_cos(x.hour, 24)))
clean_data[["weekday_sin", "weekday_cos"]] = list(clean_data["dt_dep"].apply(lambda x: cyclical_features_to_sin_cos(x.weekday(), 7)))

# %%
