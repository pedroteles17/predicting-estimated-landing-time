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
endpoints_data["metaf"]["metaf"] = endpoints_data["metaf"]["metaf"].apply(lambda x: x.strip("\n").strip("="))

parsed_metar = []
for index, row in endpoints_data["metaf"].iterrows():
    metar_str, modification_dict = MetarExtender.clean_metar_string(row["metaf"])

    try:
        obs = MetarExtender(metar_str)
        metar_dict = obs.get_metar_dict()
    except:
        print(f"Error parsing METAR: {index, metar_str}")
        continue

    metar_dict["hora"] = row["hora"]
    metar_dict["aero"] = row["aero"]
    metar_dict.update(modification_dict)

    parsed_metar.append(metar_dict)
