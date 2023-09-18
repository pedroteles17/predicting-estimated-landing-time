#%%
import pandas as pd
import openai
import dotenv
import os
from utils import FetchData, MetarExtender, MergeDataSets
from shapely.geometry import MultiPoint
from shapely.wkt import loads

dotenv.load_dotenv()

api_token = os.getenv("API_TOKEN")

openai.api_key = os.getenv("OPENAI_API_KEY")

kaggle_test = pd.read_csv("idsc_test.csv")

#%%
start_date = "2022-06-01" # First Observation: 2022-06-01 
end_date = "2023-05-13" # Last Observation: 2023-05-13

fetcher = FetchData(api_token)

endpoints = [endpoint for endpoint in fetcher.ENDPOINTS if endpoint != "cat-62"]

# Uncomment to fetch data from API
##endpoints_data = {endpoint: fetcher.fetch_endpoint(endpoint, start_date, end_date) for endpoint in endpoints}
endpoints_data = {endpoint: pd.read_parquet(f"data/{endpoint}.parquet") for endpoint in endpoints}

# 'Troca de Cabeceira' (TC) is a runway change. For some reason, the API returns the runway code without the 'SB' prefix.
endpoints_data["tc-prev"]["aero"] = "SB" + endpoints_data["tc-prev"]["aero"]

endpoints_data["tc-real"]["aero"] = "SB" + endpoints_data["tc-real"]["aero"]

# TC-real has some useless columns
endpoints_data["tc-real"] = endpoints_data["tc-real"].drop(["nova_cabeceira", "antiga_cabeceira"], axis=1)
endpoints_data["tc-real"]["troca_real"] = 1

# The satellite data is returned in a different format than the other endpoints.
endpoints_data["satelite"] = endpoints_data["satelite"]\
    .rename(columns={"data": "hora"})\
    .drop("tamanho", axis=1)

endpoints_data["satelite"]["hora"] = pd.to_datetime(endpoints_data["satelite"]["hora"])

#%%
for key, value in endpoints_data.items():
    if key == "satelite":
        continue

    timestamp_column = "dt_dep" if key == "bimtra" else "hora"

    endpoints_data[key][timestamp_column] = pd.to_datetime(endpoints_data[key][timestamp_column], unit='ms')

#%%
merger = MergeDataSets(endpoints_data["bimtra"])\
    .merge_with_espera(endpoints_data["esperas"])\
    .merge_with_metaf(endpoints_data["metaf"])\
    .merge_with_metar(endpoints_data["metar"])\
    .merge_with_tc_prev(endpoints_data["tc-prev"])\
    .merge_with_tc_real(endpoints_data["tc-real"])\
    .merge_with_satelite(endpoints_data["satelite"])

final_df = merger.bimtra_df

final_df["troca_real"] = final_df["troca_real"].fillna(0)

final_df["snapshot_radar"] = None

final_df = final_df.rename(
    columns={
        "hora": "hora_esperas",
        "aero": "aero_esperas",
        "hora_sat": "hora_ref"
    }
)

# %%
# We ensure that our dataset has the same columns as the Kaggle test set
final_df = final_df[["dt_arr"] + list(kaggle_test.columns)]

# %%
metar_strings = list(set(list(final_df["metar"]) + list(final_df["metaf"])))

metar_strings = [metar.strip("\n").strip("=") for metar in metar_strings if not pd.isna(metar)]
# %%
llm_prompt = """
You are tasked with analyzing METAR reports for aviation purposes. 
Your goal is to provide a comprehensive assessment of flying conditions on a 
scale from 0 to 100, where 0 represents extremely hazardous conditions, 
and 100 indicates perfect flying conditions. The assessment should be based on 
key meteorological parameters. You are required to return this assessment in 
JSON format, consisting of an overall score and individual scores for the following 
categories: Wind, Visibility, Cloud Cover, Temperature/Dew Point Spread, and Altimeter 
Setting. If there is insufficient data available for any specific category, 
please return "None. You must return only the JSON, and nothing else.
"""


response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role":"system", "content":llm_prompt},
        {"role":"user", "content":metar_strings[0]}
    ],
    temperature=0
)

# %%
import json
json.loads(response.choices[0].to_dict_recursive()["message"]["content"])
# %%
