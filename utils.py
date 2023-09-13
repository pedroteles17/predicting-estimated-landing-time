#%%
import requests
import pandas as pd
import re
import time
from metar import Metar

class FetchData:
    ENDPOINTS = ["bimtra", "cat-62", "esperas", "metaf", "metar", "satelite", "tc-prev", "tc-real"]

    def __init__(self, api_token):
        self.api_token = api_token

    @staticmethod
    def _is_valid_date_format(pattern: str, date: str) -> bool:
        return bool(re.match(pattern, date))

    def fetch_endpoint(self, endpoint: str, start_date: str, end_date: str) -> pd.DataFrame:
        time.sleep(10)
        if endpoint not in self.ENDPOINTS:
            raise ValueError(f"Endpoint not recognized. Must be one of {', '.join(self.ENDPOINTS)}")

        date_format = r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}$' if endpoint == "cat-62" else r'^\d{4}-\d{2}-\d{2}$'

        if not self._is_valid_date_format(date_format, start_date) or not self._is_valid_date_format(date_format, end_date):
            raise ValueError(f"Start and end dates must be in the right format ({date_format}).")

        url = f"http://montreal.icea.decea.mil.br:5002/api/v1/{endpoint}?token={self.api_token}&idate={start_date}&fdate={end_date}"
        response = requests.get(url)
        return pd.DataFrame(response.json())

class MetarExtender(Metar.Metar):
    def __init__(self, metar_str):
        super().__init__(metar_str)

    @staticmethod
    def _clean_metar_string(metar_str):
        metar_str = metar_str.replace("\n     ", ";")

        if metar_str.split(" ")[1] == "COR":
            metar_str = metar_str[6:]

    def get_metar_dict(self):
        # sky_condition has a different format (\n     ) so we need to replace it
        metar_lines = self.string().replace("\n     ", ";").split("\n")

        metar_dict = {}
        for line in metar_lines:
            key, value = line.split(":", 1)
            metar_dict[key.strip()] = value.strip()

        return metar_dict