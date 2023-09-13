#%%
import pandas as pd
import dotenv
import os
from utils import FetchData, MetarExtender

dotenv.load_dotenv()

api_token = os.getenv("API_TOKEN")

#%%
start_date = "2022-06-01" # First Observation: 2022-06-01 
end_date = "2023-05-13" # Last Observation: 2023-05-13

fetcher = FetchData(api_token)

endpoints = [endpoint for endpoint in fetcher.ENDPOINTS if endpoint != "cat-62"]

# Uncomment to fetch data from API
##endpoints_data = {endpoint: fetcher.fetch_endpoint(endpoint, start_date, end_date) for endpoint in endpoints}

endpoints_data = {endpoint: pd.read_parquet(f"data/{endpoint}.parquet") for endpoint in endpoints}

# %%
endpoints_data["metar"]["metar"] = endpoints_data["metar"]["metar"].apply(lambda x: x.strip("="))

parsed_metar = []
for index, row in endpoints_data["metar"].iterrows():
    metar_str = row["metar"]

    metar_modifications = {}

    report_elements = metar_str.split(" ")

    if report_elements[-2] == "WS":
        metar_str = " ".join(report_elements[:-2]).strip()
        metar_modifications["ws"] = report_elements[-1]

    if report_elements[1] == "COR":
        metar_str = metar_str.replace("COR ", "")
        metar_modifications["cor"] = True

    if "/////////" in metar_str:
        metar_str = metar_str.replace("/////////", "")
        metar_modifications["missing_data"] = True

    try:
        obs = MetarExtender(metar_str)
        metar_dict = obs.get_metar_dict()
    except:
        print(f"Error parsing METAR: {index, metar_str}")
        continue

    metar_dict["hora"] = row["hora"]
    metar_dict["aero"] = row["aero"]

    parsed_metar.append(metar_dict | metar_modifications)

# %%
MetarExtender("METAR SBSV 261900Z 12009KT CAVOK Q1015").get_metar_dict()
# %%
