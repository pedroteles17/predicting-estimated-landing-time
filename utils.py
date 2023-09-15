#%%
import requests
import pandas as pd
import re
import time
from metar import Metar

class MergeDataSets:
    AIRPORT_COLUMN = "aero"
    HOUR_COLUMN = "hora"

    def __init__(self, bimtra_df: pd.DataFrame):
        self.bimtra_df = bimtra_df

    def _merge_dataframes(self, left_on_cols, right_df, suffixes):
        # Merge two dataframes and return the result
        return self.bimtra_df.merge(
            right_df, left_on=left_on_cols, right_on=[self.AIRPORT_COLUMN, self.HOUR_COLUMN], how="left", suffixes=suffixes
        )

    def _floor_departure_date(self):
        self.bimtra_df["dt_dep_floor"] = self.bimtra_df["dt_dep"].dt.floor('H')

    def _ceil_departure_date(self):
        self.bimtra_df["dt_dep_ceiling"] = self.bimtra_df["dt_dep"].dt.ceil('H')

    def merge_with_espera(self, wait_df: pd.DataFrame) -> 'MergeDataSets':
        # To merge, we must floor the departure date to the hour and lag it by one hour
        self._floor_departure_date()
        self.bimtra_df["dt_dep_floor"] = self.bimtra_df["dt_dep_floor"] - pd.to_timedelta(1, unit='h')

        self.bimtra_df = self._merge_dataframes(
            left_on_cols=["destino", "dt_dep_floor"],
            right_df=wait_df,
            suffixes=("", "_esperas")
        ).drop("dt_dep_floor", axis=1)
        
        return self

    def merge_with_metaf(self, metaf_df: pd.DataFrame) -> 'MergeDataSets':
        self._ceil_departure_date()

        self.bimtra_df = self._merge_dataframes(
            left_on_cols=["destino", "dt_dep_ceiling"],
            right_df=metaf_df,
            suffixes=("", "_metaf")
        ).drop("dt_dep_ceiling", axis=1)
        
        return self

    def merge_with_metar(self, metar_df: pd.DataFrame) -> 'MergeDataSets':
        self._floor_departure_date()

        self.bimtra_df = self._merge_dataframes(
            left_on_cols=["destino", "dt_dep_floor"],
            right_df=metar_df,
            suffixes=("", "_metar")
        ).drop("dt_dep_floor", axis=1)
        
        return self

    def merge_with_tc_prev(self, tc_prev: pd.DataFrame) -> 'MergeDataSets':
        self._ceil_departure_date()

        self.bimtra_df = self._merge_dataframes(
            left_on_cols=["destino", "dt_dep_ceiling"],
            right_df=tc_prev,
            suffixes=("", "_tcp")
        ).drop("dt_dep_ceiling", axis=1)
        
        return self

    def merge_with_tc_real(self, tc_real: pd.DataFrame) -> 'MergeDataSets':
        self._floor_departure_date()

        tc_real["hora"] = tc_real["hora"].dt.floor('H')

        self.bimtra_df = self._merge_dataframes(
            left_on_cols=["destino", "dt_dep_floor"],
            right_df=tc_real,
            suffixes=("", "_tcr")
        ).drop("dt_dep_floor", axis=1)
        
        return self

    def merge_with_satelite(self, satelite: pd.DataFrame) -> 'MergeDataSets':
        self._floor_departure_date()

        self.bimtra_df = self.bimtra_df.merge(
            satelite, left_on="dt_dep_floor", right_on="hora", how="left", suffixes=("", "_sat")
        ).drop("dt_dep_floor", axis=1)
        
        return self

class FetchData:
    ENDPOINTS = ["bimtra", "cat-62", "esperas", "metaf", "metar", "satelite", "tc-prev", "tc-real"]
    BASE_URL = "http://montreal.icea.decea.mil.br:5002/api/v1/"
    VALID_DATE_FORMATS = {
        "cat-62": r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}$",
        "other": r"^\d{4}-\d{2}-\d{2}$",
    }

    def __init__(self, api_token):
        self.api_token = api_token

    @staticmethod
    def _is_valid_date_format(pattern: str, date: str) -> bool:
        return bool(re.match(pattern, date))

    def fetch_endpoint(self, endpoint: str, start_date: str, end_date: str) -> pd.DataFrame:
        time.sleep(10)
        if endpoint not in self.ENDPOINTS:
            raise ValueError(f"Endpoint not recognized. Must be one of {', '.join(self.ENDPOINTS)}")

        date_format = self.VALID_DATE_FORMATS["cat-62"] if endpoint == "cat-62" else self.VALID_DATE_FORMATS["other"]

        if not self._is_valid_date_format(date_format, start_date) or not self._is_valid_date_format(date_format, end_date):
            raise ValueError(f"Start and end dates must be in the right format ({date_format}).")

        url = f"{self.BASE_URL}{endpoint}?token={self.api_token}&idate={start_date}&fdate={end_date}"
        response = requests.get(url)
        response.raise_for_status()
        return pd.DataFrame(response.json())

    def fetch_cat_62(self, start_date: str, end_date: str) -> pd.DataFrame:
        date_range = pd.date_range(start=start_date, end=end_date, freq=f'2D')

        dfs = []
        for date in date_range:
            start_date = date.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            end_date = (date + pd.Timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            try:
                dfs.append(self.fetch_endpoint("cat-62", start_date, end_date))
            except Exception as e:
                print(f"Error fetching data for {date}: {e}")

        return pd.concat(dfs, ignore_index=True)
        

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
