#%%
import pandas as pd
from utils import cyclical_features_to_sin_cos, parse_metars, SnapshotRadar
from tqdm import tqdm
import calendar
tqdm.pandas()

train_or_test = "test" # test or train

if train_or_test == "train":
    clean_data = pd.read_parquet("data/feature_engineering/clean_data.parquet")
    output_file = "data/modelling/train_df.parquet"
elif train_or_test == "test":
    clean_data = pd.read_excel("data/original_files/idsc_dataset.xlsx")
    output_file = "data/modelling/test_df.parquet"
else:
    raise ValueError("train_or_test must be 'train' or 'test'")

#%%
metar_scores = pd.read_parquet("data/feature_engineering/metar_llm_scores.parquet")
for column in metar_scores.columns:
    if column != "unique_metar":
        metar_scores.rename(columns={column: "metar_" + column}, inplace=True)

clean_data = clean_data\
    .assign(
        tcr = lambda x: x["hora_tcr"].apply(lambda x: 0 if pd.isna(x) else 1), # Troca de Cabeceira Real
        tcp = lambda x: x["troca"].fillna(0), # Troca de Cabeceira Prevista
        esperas = lambda x: x["esperas"].fillna(0),
        unique_metar = lambda x: x["metaf"].combine_first(x["metar"]),
        is_forecast = lambda x: x["metaf"].apply(lambda x: 0 if pd.isna(x) else 1),
        image_date = lambda x: x["path"].apply(lambda x: x.split("_")[-1].split(".")[0] if not pd.isna(x) else x)
    )\
    .assign(
        unique_metar = lambda x: x["unique_metar"].apply(lambda x: x.strip("\n").strip("=") if not pd.isna(x) else x),
        image_date = lambda x: x["image_date"].apply(lambda x: pd.to_datetime(x, format="%Y%m%d%H%M") if not pd.isna(x) else x),
        dt_dep_aux = lambda x: x["dt_dep"].apply(lambda x: x.strftime('%H:%M:%S')),
        estimated_departure = lambda x: pd.to_datetime(x['image_date'].dt.strftime('%Y-%m-%d') + ' ' + x['dt_dep_aux'].astype(str))
    )\
    .drop([
        "hora_tcr", "aero_tcr",
        "hora_tcp", "aero_tcp",
        "hora_metar", "aero_metar",
        "hora_metaf", "aero_metaf",
        "hora_esperas", "aero_esperas",
        "hora_ref", "metar", 
        "metaf", "troca", "dt_dep_aux"
        ], axis=1
    )\
    .merge(metar_scores, on="unique_metar", how="left")

# Drop column if it exists (this will happen if clean_data is the training data)
if "dt_arr" in clean_data.columns:
    clean_data.drop(["dt_arr"], axis=1, inplace=True)

#%%
# Parse METARs
metar_strings = list(clean_data["unique_metar"].dropna().unique())

parsed_metar = parse_metars(metar_strings)\
    .drop([
        "station", "type", "wind", "sky", "METAR", # useless
        "weather"  # Too many NaNs
        ], axis=1)\
    .rename(columns={"dew point": "dew_point"})

# Drop column if it exists (this will happen if clean_data is the training data)
if "visual range" in parsed_metar.columns:
    parsed_metar.drop(["visual range"], axis=1, inplace=True) # Too many NaNs

clean_data = clean_data\
    .merge(parsed_metar, left_on="unique_metar", right_on="original_metar", how="left")\
    .drop(["original_metar","wind_direction", "wind_speed"], axis=1)

#%%
# Transform timestamps to cyclical features (sin, cos)
clean_data[["minute_sin", "minute_cos"]] = list(clean_data["estimated_departure"].apply(lambda x: cyclical_features_to_sin_cos(x.minute, 60)))
clean_data[["hour_sin", "hour_cos"]] = list(clean_data["estimated_departure"].apply(lambda x: cyclical_features_to_sin_cos(x.hour, 24)))

clean_data[["week_sin", "week_cos"]] = list(clean_data["estimated_departure"]\
    .apply(lambda x: cyclical_features_to_sin_cos(x.week, 7) if not pd.isna(x) else (pd.NA, pd.NA)
    )
)

clean_data[["month_sin", "month_cos"]] = list(clean_data["estimated_departure"]\
    .apply(
        lambda x: cyclical_features_to_sin_cos(x.month, calendar.monthrange(x.year, x.month)[1]) if not pd.isna(x) else (pd.NA, pd.NA)
    )
)

#%%
clean_data["snapshot_radar_distances"] = clean_data\
    .progress_apply(SnapshotRadar.iterate_airport_distances, axis=1)
    
clean_data = clean_data\
    .assign(
        radar_less_500_km = lambda x: x["snapshot_radar_distances"].apply(lambda x: len([i for i in x if i < 500]) if len(x) > 0 else pd.NA),
        radar_less_1000_km = lambda x: x["snapshot_radar_distances"].apply(lambda x: len([i for i in x if i < 1000]) if len(x) > 0 else pd.NA),
        radar_total = lambda x: x["snapshot_radar_distances"].apply(lambda x: len(x) if len(x) > 0 else pd.NA),
    )\
    .drop(["snapshot_radar_distances", "snapshot_radar"], axis=1)

clean_data.to_parquet(output_file)

# %%
