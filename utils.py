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
        self.metar_str = metar_str

    @staticmethod
    def _metaf_to_metar(metaf_str):
        if metaf_str.split(" ")[0] == "METAF":
            return metaf_str.replace("METAF", "METAR"), True
        
        return metaf_str, False

    @staticmethod
    def _remove_cor(metar_str):
        if metar_str.split(" ")[1] == "COR":
            return metar_str.replace("COR ", ""), True

        return metar_str, False

    @staticmethod
    def _remove_missing_data(metar_str):
        has_missing_data = False

        # Regex to match Rxx///// or RxxL///// or RxxK/////
        pattern = r'\bR\d{2}[A-Z]?/////[^ ]*\s*'

        if re.search(pattern, metar_str):
            metar_str = re.sub(pattern, '', metar_str)
            has_missing_data = True

        if "/////////" in metar_str:
            metar_str = metar_str.replace("/////////", "")
            has_missing_data = True

        return metar_str, has_missing_data

    @staticmethod
    def _remove_ws(metar_str):
        if metar_str.split(" ")[-2] == "WS":
            metar_str = " ".join(metar_str.split(" ")[:-2]).strip()
            ws_value = metar_str.split(" ")[-1]
            return metar_str, ws_value

        return metar_str, None

    @staticmethod
    def clean_metar_string(metar_str):
        metar_str, cor = MetarExtender._remove_cor(metar_str)
        metar_str, missing_data = MetarExtender._remove_missing_data(metar_str)
        metar_str, ws_value = MetarExtender._remove_ws(metar_str)
        metar_str, is_metaf = MetarExtender._metaf_to_metar(metar_str)

        modification_dict = {
            "correction": cor,
            "missing_data": missing_data,
            "wind_shear": ws_value,
            "is_forecast": is_metaf
        }

        return metar_str, modification_dict

    @staticmethod
    def _remove_sky_extra_spaces(metar_str):
        return metar_str.replace("\n     ", "")

    def get_metar_dict(self):
        metar_lines = self._remove_sky_extra_spaces(self.string()).split("\n")

        metar_dict = {}
        for line in metar_lines:
            key, value = line.split(":", 1)
            metar_dict[key.strip()] = value.strip()

        return metar_dict
