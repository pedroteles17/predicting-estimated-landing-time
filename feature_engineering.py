#%%
import pandas as pd
from utils import cyclical_features_to_sin_cos, parse_metars
import json

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

#%%
import json

with open("data/metar_scores_llm/metar_results_final1.json", "r") as f:
    data1 = json.load(f)

with open("data/metar_scores_llm/metar_results_final2.json", "r") as f:
    data2 = json.load(f)

with open("data/metar_scores_llm/metar_results_final3.json", "r") as f:
    data3 = json.load(f)

with open("data/metar_scores_llm/metar_results_0.json", "r") as f:
     data4 = json.load(f)

final_metar_scores = data1 | data2 | data3 | data4

#%%
metar_scores_list = []
for key in final_metar_scores.keys():
    metar_scores_list.append({"unique_metar": key} | final_metar_scores[key])

metar_scores = pd.DataFrame(metar_scores_list)

#%%
metar_scores = metar_scores\
    .assign(
        overall_score = lambda x: x["overall_score"].combine_first(x["Overall Score"]),
        wind_score = lambda x: x["wind_score"].combine_first(x["Wind"]).combine_first(x["Wind Score"]),
        visibility_score = lambda x: x["visibility_score"].combine_first(x["Visibility"]).combine_first(x["Visibility Score"]),
        cloud_cover_score = lambda x: x["cloud_cover_score"].combine_first(x["Cloud Cover"]).combine_first(x["Cloud Cover Score"]),
        dew_point_spread_score = lambda x: x["dew_point_spread_score"].combine_first(x["Dew Point Spread"]).combine_first(x["Dew Point Spread Score"]),
        altimeter_setting_score = lambda x: x["altimeter_setting_score"].combine_first(x["Altimeter Setting"]).combine_first(x["Altimeter Setting Score"]),
        temperature_score = lambda x: x["temperature_score"].combine_first(x["Temperature Score"]).combine_first(x["Temperature"])
    )\
    .drop([
        "Overall Score", "Wind", "Visibility", "Cloud Cover", 
        "Dew Point Spread", "Altimeter Setting", "Temperature Score",
        "Temperature", "Wind Score", "Visibility Score", "Cloud Cover Score",
        "Dew Point Spread Score", "Altimeter Setting Score"
        ], axis=1)

for column in metar_scores.columns:
    if column != "unique_metar":
        metar_scores[column] = metar_scores[column].replace('None', pd.NA).apply(pd.to_numeric)



# %%
metar_strings = list(final_metar_scores.keys())

import pickle

# Open the file for reading in binary mode
with open("data/feature_engineering/metar_strings.pickle", "rb") as f:
    # Use pickle to load the list from the file
    my_list = pickle.load(f)

len(list(set(my_list) - set(metar_strings)))
# %%
metar_scores.to_parquet("data/feature_engineering/metar_llm_scores.parquet")
# %%
