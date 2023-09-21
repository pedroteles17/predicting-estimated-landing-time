#%%
import pandas as pd
import dotenv
import os
from utils import FetchData, MergeDataSets

dotenv.load_dotenv()

api_token = os.getenv("API_TOKEN")

kaggle_test = pd.read_csv("data/original_files/idsc_test.csv")

#%%
start_date = "2022-06-01" # First Observation: 2022-06-01 
end_date = "2023-05-13" # Last Observation: 2023-05-13

fetcher = FetchData(api_token)

endpoints = [endpoint for endpoint in fetcher.ENDPOINTS if endpoint != "cat-62"]

# Uncomment to fetch data from API
##endpoints_data = {endpoint: fetcher.fetch_endpoint(endpoint, start_date, end_date) for endpoint in endpoints}
endpoints_data = {endpoint: pd.read_parquet(f"data/original_files/{endpoint}.parquet") for endpoint in endpoints}

# Uncomment to fetch data from API (its a slow process, ~3 hours)
##date_range = pd.date_range(start=start_date, end=end_date, freq=f'1D')
##cat_62 = [fetcher.fetch_cat_62(date) for date in tqdm(date_range)]
##endpoints_data["cat-62"] = pd.concat(cat_62)
endpoints_data["cat-62"] = pd.read_parquet("data/original_files/cat-62.parquet")

# 'Troca de Cabeceira' (TC) is a runway change. For some reason, the API returns the runway code without the 'SB' prefix.
endpoints_data["tc-prev"]["aero"] = "SB" + endpoints_data["tc-prev"]["aero"]

endpoints_data["tc-real"]["aero"] = "SB" + endpoints_data["tc-real"]["aero"]

# TC-real has some useless columns
endpoints_data["tc-real"] = endpoints_data["tc-real"].drop(["nova_cabeceira", "antiga_cabeceira"], axis=1)

# The satellite data is returned in a different format than the other endpoints.
endpoints_data["satelite"] = endpoints_data["satelite"]\
    .rename(columns={"data": "hora"})\
    .drop("tamanho", axis=1)

#%%
for key, value in endpoints_data.items():
    if key == "satelite":
        # Satelite date comes in a different format
        endpoints_data["satelite"]["hora"] = pd.to_datetime(endpoints_data["satelite"]["hora"])
        continue

    timestamp_column = "dt_dep" if key == "bimtra" else ("dt_radar" if key == "cat-62" else "hora")

    endpoints_data[key][timestamp_column] = pd.to_datetime(endpoints_data[key][timestamp_column], unit='ms')

#%%
merger = MergeDataSets(endpoints_data["bimtra"])\
    .merge_with_espera(endpoints_data["esperas"])\
    .merge_with_metaf(endpoints_data["metaf"])\
    .merge_with_metar(endpoints_data["metar"])\
    .merge_with_tc_prev(endpoints_data["tc-prev"])\
    .merge_with_tc_real(endpoints_data["tc-real"])\
    .merge_with_satelite(endpoints_data["satelite"])\
    .merge_with_cat_62(endpoints_data["cat-62"])

final_df = merger.bimtra_df

final_df = final_df.rename(
    columns={
        "hora": "hora_esperas",
        "aero": "aero_esperas",
        "hora_sat": "hora_ref"
    }
)

# We ensure that our dataset has the same columns as the Kaggle test set
final_df = final_df[["dt_arr"] + list(kaggle_test.columns)]

# Save file to be used in the next steps
final_df.to_parquet("data/feature_engineering/clean_data.parquet")

#%%
