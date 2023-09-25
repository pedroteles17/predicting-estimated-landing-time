# %%
import pandas as pd
import dotenv
import os
from utils import FetchData, MergeDataSets, FillMissingValues

dotenv.load_dotenv()

api_token = os.getenv("API_TOKEN")

kaggle_test = pd.read_csv("data/original_files/idsc_test.csv")

# %%
start_date = "2022-06-01"  # First Observation: 2022-06-01
end_date = "2023-05-13"  # Last Observation: 2023-05-13

fetcher = FetchData(api_token)

endpoints = [endpoint for endpoint in fetcher.ENDPOINTS if endpoint != "cat-62"]

# Uncomment to fetch data from API
##endpoints_data = {endpoint: fetcher.fetch_endpoint(endpoint, start_date, end_date) for endpoint in endpoints}
endpoints_data = {
    endpoint: pd.read_parquet(f"data/original_files/{endpoint}.parquet")
    for endpoint in endpoints
}

# Uncomment to fetch data from API (its a slow process, ~3 hours)
##date_range = pd.date_range(start=start_date, end=end_date, freq=f'1D')
##cat_62 = [fetcher.fetch_cat_62(date) for date in tqdm(date_range)]
##endpoints_data["cat-62"] = pd.concat(cat_62)
endpoints_data["cat-62"] = pd.read_parquet("data/original_files/cat-62.parquet")

# 'Troca de Cabeceira' (TC) is a runway change. For some reason, the API returns the runway code without the 'SB' prefix.
endpoints_data["tc-prev"]["aero"] = "SB" + endpoints_data["tc-prev"]["aero"]

endpoints_data["tc-real"]["aero"] = "SB" + endpoints_data["tc-real"]["aero"]

# TC-real has some useless columns
endpoints_data["tc-real"] = endpoints_data["tc-real"].drop(
    ["nova_cabeceira", "antiga_cabeceira"], axis=1
)

# The satellite data is returned in a different format than the other endpoints.
endpoints_data["satelite"] = (
    endpoints_data["satelite"].rename(columns={"data": "hora"}).drop("tamanho", axis=1)
)

# %%
for key, value in endpoints_data.items():
    if key == "satelite":
        # Satelite date comes in a different format
        endpoints_data["satelite"]["hora"] = pd.to_datetime(
            endpoints_data["satelite"]["hora"]
        )
        continue

    timestamp_column = (
        "dt_dep" if key == "bimtra" else ("dt_radar" if key == "cat-62" else "hora")
    )

    endpoints_data[key][timestamp_column] = pd.to_datetime(
        endpoints_data[key][timestamp_column], unit="ms"
    )

#%%
# We have some duplicated rows. We will drop them.
endpoints_data["bimtra"] = endpoints_data["bimtra"].drop_duplicates()

## We have some special reports (SPECI) in the data. If there are collisions, we will keep the SPECI report.
### The sorting will ensure that the SPECI reports are the first ones in case of collision.
endpoints_data["metar"] = endpoints_data["metar"]\
    .sort_values(by=["hora", "aero", "metar"], ascending=[True, True, False])\
    .drop_duplicates(subset=["hora", "aero"], keep="first")

## Handling Duplicate Rows in cat-62 Data
## Keep the first occurrence for each flight, prioritizing rows with flight level data.
## 81082ac7e863447586383829cc6aae2e, for example, has very different locaitons for the same time.
endpoints_data["cat-62"] = endpoints_data["cat-62"]\
    .sort_values(by=["dt_radar", "flightid", "flightlevel"], ascending=[True, True, False])\
    .drop_duplicates(subset=["dt_radar", "flightid"], keep="first")

# %%
merger = (
    MergeDataSets(endpoints_data["bimtra"])
    .merge_with_espera(endpoints_data["esperas"])
    .merge_with_metaf(endpoints_data["metaf"])
    .merge_with_metar(endpoints_data["metar"])
    .merge_with_tc_prev(endpoints_data["tc-prev"].copy())
    .merge_with_tc_real(endpoints_data["tc-real"].copy())
    .merge_with_satelite(endpoints_data["satelite"])
    .merge_with_cat_62(endpoints_data["cat-62"].copy())
)

final_df = merger.bimtra_df

final_df = final_df.rename(
    columns={"hora": "hora_esperas", "aero": "aero_esperas", "hora_sat": "hora_ref"}
)

# We ensure that our dataset has the same columns as the Kaggle test set
final_df = final_df[["dt_arr"] + list(kaggle_test.columns)]

#%%
# Fill missing values
missing_values_filler = FillMissingValues(final_df)

missing_values_filler\
    .fill_snapshot_radar(endpoints_data["cat-62"].copy(), minutes_lag=10)\
    .fill_path(endpoints_data["satelite"].copy(), hours_lag=6)\
    .fill_metar(endpoints_data["metar"].copy(), hours_lag=6)

# %%
final_df = missing_values_filler.df.sort_values(by=["dt_dep", "flightid"])

# Save file to be used in the next steps
final_df.to_parquet("data/feature_engineering/clean_data.parquet")

#%%
import json

# Specify the path to your JSON file
json_file_path = "data/metar_scores_llm/metar_results_650.json"

# Open the JSON file for reading
with open(json_file_path, "r") as json_file:
    # Parse the JSON data
    data = json.load(json_file)

# Now, 'data' contains the contents of the JSON file as a Python dictionary
len(data)

# %%
