# %%
import requests
import pandas as pd
import duckdb
import numpy as np
import re  # Regex
import time
import os
from metar import Metar  # METAR parsing library
import openai  # OpenAI API
import json
import aiofiles  # Async file writing
import asyncio  # Async requests
from tqdm import tqdm  # Progress bar
from geopy.distance import great_circle
from typing import Tuple
import pyproj

from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA



# class definitions for image clustering
class KerasFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, batch_size=32):
        model = VGG16()
        self.model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        self.batch_size = batch_size

    def fit(self, X):
        return self

    def transform(self, X):
        n_samples = len(X)
        n_batches = int(np.ceil(n_samples / self.batch_size))
        features = []

        for idx in range(n_batches):
            start = idx * self.batch_size
            end = (idx + 1) * self.batch_size
            batch_images = X[start:end]
            batch_features = self.process_batch(batch_images)
            features.extend(batch_features)

        return list(zip(X, np.vstack(features)))

    def process_batch(self, batch_images):
        batch_data = [preprocess_input(np.array(load_img(img, target_size=(224, 224)))) for img in batch_images]
        batch_data = np.array(batch_data)
        return self.model.predict(batch_data)


class DimReducer(TransformerMixin, BaseEstimator):

    def __init__(self, n_components=100, random_state=1):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)

    def fit(self, X, y=None):
        _, features = zip(*X)
        self.pca.fit(features)
        return self

    def transform(self, X):
        filenames, features = zip(*X)
        reduced_features = self.pca.transform(features)
        return list(zip(filenames, reduced_features))


class ClusterKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters)

    def fit(self, X, y=None):
        filenames, features = zip(*X)
        self.kmeans.fit(features)
        self.labels_ = self.kmeans.labels_

        self.groups = {}
        for filename, label in zip(filenames, self.labels_):
            if label not in self.groups:
                self.groups[label] = []
            self.groups[label].append(filename)

        return self


def get_image_clusters(img_path_list: str):

    pipeline = Pipeline([
      ("feature_extractor", KerasFeatureExtractor(batch_size=32)),
      ("reducer", DimReducer()),
      ("clustering", ClusterKMeans())
    ])
    pipeline.fit(img_path_list)

    groups = pipeline.named_steps["clustering"].groups
    dict_groups = {}
    for group, group_items in groups.items():
        for item in group_items:
            dict_groups[item] = group

    return dict_groups



def calculate_expected_arrival(row):
    if pd.isna(row["estimated_departure"]) or pd.isna(row["origin_destiny_encode"]):
        return pd.NA

    return row["estimated_departure"] + pd.Timedelta(
        seconds=row["origin_destiny_encode"]
    )


def number_of_flights_expected(
    table_name: str,
    airport: str,
    date: pd._libs.tslibs.timestamps.Timestamp,
    minutes_lag: Tuple[int, int],
    departure: bool,
):
    column = "estimated_departure" if departure else "expected_arrival"

    start_time = date + pd.Timedelta(minutes=minutes_lag[0])
    end_time = date + pd.Timedelta(minutes=minutes_lag[1])

    duckdb_query = f"""
    SELECT COUNT(*) FROM {table_name}
    WHERE {column} > '{start_time}' 
        AND {column} < '{end_time}'
            AND destino = '{airport}'
    """

    row_count = duckdb.sql(duckdb_query).fetchall()[0][0]

    return row_count

class GeoSpatial:
    @staticmethod
    def cardinal_direction_to_degrees(cardinal_direction: str) -> float:
        cardinal_directions = [
            "N",
            "NNE",
            "NE",
            "ENE",
            "E", 
            "ESE",
            "SE",
            "SSE",
            "S",
            "SSW",
            "SW",
            "WSW",
            "W",
            "WNW",
            "NW",
            "NNW",
        ]

        if cardinal_direction not in cardinal_directions:
            return None   

        return cardinal_directions.index(cardinal_direction) * 22.5
    
    @staticmethod
    def direction_between_points(point_1: Tuple[float, float], point_2: Tuple[float, float]) -> float:
        # point_1 and point_2 are tuples with (lat, lon)
        lat_1, lon_1 = point_1
        lat_2, lon_2 = point_2
        
        geodesic = pyproj.Geod(ellps='WGS84')
        fwd_azimuth, _, _ = geodesic.inv(lon_1, lat_1, lon_2, lat_2)

        return fwd_azimuth

class BrazilianHolidays:
    HOLIDAYS = [
        "2022-06-16",
        "2022-09-07",
        "2022-10-12",
        "2022-11-02",
        "2022-11-15",
        "2022-12-25",
        "2023-01-01",
        "2023-02-28",
        "2023-03-01",
        "2023-04-07",
        "2023-04-21",
        "2023-05-01",
        "2023-06-15",
        "2023-09-07",
        "2023-10-12",
        "2023-11-02",
        "2023-11-15",
        "2023-12-25",
    ]

    def __init__(self):
        self.holidays = pd.to_datetime(self.HOLIDAYS)

    def days_to_holiday(self, date: pd._libs.tslibs.timestamps.Timestamp) -> int:
        return min(abs(self.holidays - date))


class BrazilianAirports:
    AIRPORT_INFO = {
        "SBSP": {
            "name": "Congonhas",
            "city": "São Paulo",
            "state": "SP",
            "lat": -23.626110076904297,
            "lon": -46.65638732910156,
            "elevation": 802,
            "runway_number": 2,
            "runway_length": [1660, 1195],
        },
        "SBCT": {
            "name": "Afonso Pena",
            "city": "Curitiba",
            "state": "PR",
            "lat": -25.531700134277344,
            "lon": -49.17580032348633,
            "elevation": 911,
            "runway_number": 2,
            "runway_length": [2218, 1798],
        },
        "SBPA": {
            "name": "Salgado Filho",
            "city": "Porto Alegre",
            "state": "RS",
            "lat": -29.994400024414062,
            "lon": -51.1713981628418,
            "elevation": 3,
            "runway_number": 1,
            "runway_length": [3200],
        },
        "SBSV": {
            "name": "Deputado Luís Eduardo Magalhães",
            "city": "Salvador",
            "state": "BA",
            "lat": -12.908611297607422,
            "lon": -38.3224983215332,
            "elevation": 20,
            "runway_number": 2,
            "runway_length": [2763, 1518],
        },
        "SBGR": {
            "name": "Guarulhos",
            "city": "São Paulo",
            "state": "SP",
            "lat": -23.435556411743164,
            "lon": -46.47305679321289,
            "elevation": 750,
            "runway_number": 2,
            "runway_length": [3000, 3620],
        },
        "SBCF": {
            "name": "Tancredo Neves",
            "city": "Belo Horizonte",
            "state": "MG",
            "lat": -19.62444305419922,
            "lon": -43.97194290161133,
            "elevation": 827,
            "runway_number": 1,
            "runway_length": [3600],
        },
        "SBBR": {
            "name": "Presidente Juscelino Kubitschek",
            "city": "Brasília",
            "state": "DF",
            "lat": -15.86916732788086,
            "lon": -47.920833587646484,
            "elevation": 1066,
            "runway_number": 2,
            "runway_length": [3050, 3150],
        },
        "SBRF": {
            "name": "Guararapes - Gilberto Freyre",
            "city": "Recife",
            "state": "PE",
            "lat": -8.126389503479004,
            "lon": -34.92361068725586,
            "elevation": 10,
            "runway_number": 1,
            "runway_length": [2751],
        },
        "SBRJ": {
            "name": "Santos Dumont",
            "city": "Rio de Janeiro",
            "state": "RJ",
            "lat": -22.9102783203125,
            "lon": -43.16310119628906,
            "elevation": 3,
            "runway_number": 2,
            "runway_length": [1260, 1323],
        },
        "SBGL": {
            "name": "Galeão - Antônio Carlos Jobim",
            "city": "Rio de Janeiro",
            "state": "RJ",
            "lat": -22.809999465942383,
            "lon": -43.25055694580078,
            "elevation": 9,
            "runway_number": 2,
            "runway_length": [2930, 4000],
        },
        "SBKP": {
            "name": "Viracopos",
            "city": "Campinas",
            "state": "SP",
            "lat": -23.007400512695312,
            "lon": -47.134498596191406,
            "elevation": 661,
            "runway_number": 1,
            "runway_length": [3150],
        },
        "SBFL": {
            "name": "Hercílio Luz",
            "city": "Florianópolis",
            "state": "SC",
            "lat": -27.670278549194336,
            "lon": -48.5525016784668,
            "elevation": 5,
            "runway_number": 2,
            "runway_length": [2400, 1180],
        },
    }

    def __init__(self):
        self.airport_info = self.AIRPORT_INFO

    def calculate_distance(self, airport_1: str, airport_2: str) -> float:
        airport_1 = self.airport_info[airport_1]
        airport_2 = self.airport_info[airport_2]
        return great_circle(
            (airport_1["lat"], airport_1["lon"]), (airport_2["lat"], airport_2["lon"])
        ).kilometers

    def get_runway_info(self, airport: str) -> dict:
        selected_keys = ["runway_number", "runway_length", "elevation"]
        return {key: self.airport_info[airport][key] for key in selected_keys}


def cyclical_features_to_sin_cos(time_list, max_val: int) -> pd.DataFrame:
    sin = np.sin(2 * np.pi * time_list / max_val)
    cos = np.cos(2 * np.pi * time_list / max_val)
    return sin, cos


class FillMissingValues:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def fill_metar(self, metar: pd.DataFrame, hours_lag: int) -> "FillMissingValues":
        for _, row in tqdm(
            self.df.iterrows(), total=self.df.shape[0], desc="Filling metar"
        ):
            if pd.isna(row["metar"]):
                end_date = row["dt_dep"].floor("H")
                start_date = end_date - pd.Timedelta(hours=hours_lag)

                relevant_metar = metar[
                    (metar["hora"] >= start_date)
                    & (metar["hora"] <= end_date)
                    & (metar["aero"] == row["destino"])
                ]

                if not relevant_metar.empty:
                    relevant_metar = relevant_metar.sort_values(
                        by="hora", ascending=False
                    ).reset_index(drop=True)
                    self.df.at[row.name, "metar"] = relevant_metar["metar"][0]
                    self.df.at[row.name, "hora_metar"] = relevant_metar["hora"][0]
                    self.df.at[row.name, "aero_metar"] = relevant_metar["aero"][0]

        return self

    def fill_snapshot_radar(
        self, cat_62: pd.DataFrame, minutes_lag: int
    ) -> "FillMissingValues":
        for _, row in tqdm(
            self.df.iterrows(), total=self.df.shape[0], desc="Filling snapshot_radar"
        ):
            if pd.isna(row["snapshot_radar"]):
                start_date = row["dt_dep"] - pd.Timedelta(minutes=minutes_lag)
                relevant_cat_62 = cat_62[
                    (cat_62["dt_radar"] >= start_date)
                    & (cat_62["dt_radar"] <= row["dt_dep"])
                ]

                # The same flight can have multiple radar reports, so we need to select the most recent one
                relevant_cat_62 = relevant_cat_62.sort_values(
                    by=["flightid", "dt_radar"], ascending=[True, False]
                ).drop_duplicates(subset=["flightid"], keep="first")

                if not relevant_cat_62.empty:
                    multipoints = MergeDataSets.create_multipoint(relevant_cat_62)
                    self.df.at[row.name, "snapshot_radar"] = multipoints

        return self

    def fill_path(self, satelite: pd.DataFrame, hours_lag: int) -> "FillMissingValues":
        for _, row in tqdm(
            self.df.iterrows(), total=self.df.shape[0], desc="Filling path"
        ):
            if pd.isna(row["path"]):
                end_date = row["dt_dep"].floor("H")
                start_date = end_date - pd.Timedelta(hours=hours_lag)

                relevant_satelite = satelite[
                    (satelite["hora"] >= start_date) & (satelite["hora"] <= end_date)
                ]

                if not relevant_satelite.empty:
                    relevant_satelite = relevant_satelite.sort_values(
                        by="hora", ascending=False
                    ).reset_index(drop=True)
                    self.df.at[row.name, "path"] = relevant_satelite["path"][0]
                    self.df.at[row.name, "hora_ref"] = relevant_satelite["hora"][0]

        return self


class OpenAIAsync:
    # We did some prompt engineering to get the best results from OpenAI
    METAR_PROMPT = """
    Analyze METAR reports for aviation and rate flying conditions from 0 (hazardous) to 100 
    (perfect) based on key meteorological parameters. Return the assessment in JSON format 
    with an overall score and individual scores for: Wind, Visibility, Cloud Cover, 
    Dew Point Spread, Altimeter Setting, Temperature. If data is insufficient 
    for any category, return 'None'. Only the JSON should be returned.
    """.replace(
        "\n", " "
    )

    def __init__(self, api_key):
        openai.api_key = api_key
        self.results_dict = {}

    @staticmethod
    def clean_openai_response(response):
        try:
            choices = response.choices[0]
            content = choices.message["content"]
            return json.loads(content)
        except:
            print("Error when cleaning: ", response)
            return None

    @staticmethod
    async def save_to_file(data, filename):
        try:
            async with aiofiles.open(filename, "w") as file:
                await file.write(json.dumps(data, indent=4))
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving data to {filename}: {str(e)}")

    @staticmethod
    def delete_old_file(directory, filename):
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {str(e)}")

    async def main(self, metar_reports, output_directory, file_name):
        tasks = [self.fetch_scores_for_metar(metar) for metar in metar_reports]

        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Fetching scores from LLM",
        ):
            await task

            await asyncio.sleep(0.5)

        # Now, results_dict contains METAR data
        print("\nFinished fetching data.")

        # Save the results to a JSON file asynchronously
        await self.save_to_file(self.results_dict, f"{output_directory}/{file_name}")

    async def fetch_scores_for_metar(self, metar):
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.METAR_PROMPT},
                    {"role": "user", "content": metar},
                ],
                temperature=0.2,
                top_p=0.8,
            )
            metar_data = self.clean_openai_response(response)
            self.results_dict[metar] = metar_data
        except Exception as e:
            print("Error:", e)


class MergeDataSets:
    AIRPORT_COLUMN = "aero"
    HOUR_COLUMN = "hora"

    def __init__(self, bimtra_df: pd.DataFrame):
        self.bimtra_df = bimtra_df.copy()

    def _merge_dataframes(self, left_on_cols, right_df, suffixes):
        # Merge two dataframes and return the result
        return self.bimtra_df.merge(
            right_df,
            left_on=left_on_cols,
            right_on=[self.AIRPORT_COLUMN, self.HOUR_COLUMN],
            how="left",
            suffixes=suffixes,
        )

    def _floor_departure_date(self):
        self.bimtra_df["dt_dep_floor"] = self.bimtra_df["dt_dep"].dt.floor("H")

    def _ceil_departure_date(self):
        self.bimtra_df["dt_dep_ceiling"] = self.bimtra_df["dt_dep"].dt.ceil("H")

    @staticmethod
    def create_multipoint(df):
        points = [f"({lat} {lon})" for lat, lon in zip(df["lat"], df["lon"])]
        return f"MULTIPOINT ({', '.join(points)})"

    def merge_with_cat_62(self, cat_62_df: pd.DataFrame) -> "MergeDataSets":
        self.bimtra_df["dt_dep_floor"] = self.bimtra_df["dt_dep"].dt.floor("min")

        cat_62_df["dt_radar_ceil"] = cat_62_df["dt_radar"].dt.ceil("min")

        # The same flight can have multiple radar reports, so we need to select the most recent one
        cat_62_df = cat_62_df.sort_values(
            by=["dt_radar_ceil", "flightid", "dt_radar"], ascending=[True, True, False]
        ).drop_duplicates(subset=["dt_radar_ceil", "flightid"], keep="first")

        multipoints = cat_62_df.groupby("dt_radar_ceil").apply(self.create_multipoint)

        multipoints = (
            pd.DataFrame(multipoints)
            .reset_index()
            .rename(columns={0: "snapshot_radar"})
        )

        self.bimtra_df = self.bimtra_df.merge(
            multipoints,
            left_on="dt_dep_floor",
            right_on="dt_radar_ceil",
            how="left",
            suffixes=("", "_cat"),
        ).drop(["dt_dep_floor", "dt_radar_ceil"], axis=1)

        return self

    def merge_with_espera(self, wait_df: pd.DataFrame) -> "MergeDataSets":
        # To merge, we must floor the departure date to the hour and lag it by one hour
        self._floor_departure_date()
        self.bimtra_df["dt_dep_floor"] = self.bimtra_df[
            "dt_dep_floor"
        ] - pd.to_timedelta(1, unit="h")

        self.bimtra_df = self._merge_dataframes(
            left_on_cols=["destino", "dt_dep_floor"],
            right_df=wait_df,
            suffixes=("", "_esperas"),
        ).drop("dt_dep_floor", axis=1)

        return self

    def merge_with_metaf(self, metaf_df: pd.DataFrame) -> "MergeDataSets":
        self._ceil_departure_date()

        self.bimtra_df = self._merge_dataframes(
            left_on_cols=["destino", "dt_dep_ceiling"],
            right_df=metaf_df,
            suffixes=("", "_metaf"),
        ).drop("dt_dep_ceiling", axis=1)

        return self

    def merge_with_metar(self, metar_df: pd.DataFrame) -> "MergeDataSets":
        self._floor_departure_date()

        self.bimtra_df = self._merge_dataframes(
            left_on_cols=["destino", "dt_dep_floor"],
            right_df=metar_df,
            suffixes=("", "_metar"),
        ).drop("dt_dep_floor", axis=1)

        return self

    def merge_with_tc_prev(self, tc_prev: pd.DataFrame) -> "MergeDataSets":
        self._ceil_departure_date()

        self.bimtra_df = self._merge_dataframes(
            left_on_cols=["destino", "dt_dep_ceiling"],
            right_df=tc_prev,
            suffixes=("", "_tcp"),
        ).drop("dt_dep_ceiling", axis=1)

        return self

    def merge_with_tc_real(self, tc_real: pd.DataFrame) -> "MergeDataSets":
        self._floor_departure_date()

        tc_real["hora"] = tc_real["hora"].dt.floor("H")

        # Drop duplicates by airport and hour floor
        tc_real = tc_real.drop_duplicates(subset=["aero", "hora"])

        self.bimtra_df = self._merge_dataframes(
            left_on_cols=["destino", "dt_dep_floor"],
            right_df=tc_real,
            suffixes=("", "_tcr"),
        ).drop("dt_dep_floor", axis=1)

        return self

    def merge_with_satelite(self, satelite: pd.DataFrame) -> "MergeDataSets":
        self._floor_departure_date()

        self.bimtra_df = self.bimtra_df.merge(
            satelite,
            left_on="dt_dep_floor",
            right_on="hora",
            how="left",
            suffixes=("", "_sat"),
        ).drop("dt_dep_floor", axis=1)

        return self


class FetchData:
    ENDPOINTS = [
        "bimtra",
        "cat-62",
        "esperas",
        "metaf",
        "metar",
        "satelite",
        "tc-prev",
        "tc-real",
    ]
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

    def fetch_endpoint(
        self, endpoint: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        time.sleep(10)
        if endpoint not in self.ENDPOINTS:
            raise ValueError(
                f"Endpoint not recognized. Must be one of {', '.join(self.ENDPOINTS)}"
            )

        date_format = (
            self.VALID_DATE_FORMATS["cat-62"]
            if endpoint == "cat-62"
            else self.VALID_DATE_FORMATS["other"]
        )

        if not self._is_valid_date_format(
            date_format, start_date
        ) or not self._is_valid_date_format(date_format, end_date):
            raise ValueError(
                f"Start and end dates must be in the right format ({date_format})."
            )

        url = f"{self.BASE_URL}{endpoint}?token={self.api_token}&idate={start_date}&fdate={end_date}"
        response = requests.get(url)
        response.raise_for_status()
        return pd.DataFrame(response.json())

    def fetch_cat_62(self, date: pd._libs.tslibs.timestamps.Timestamp) -> pd.DataFrame:
        # API has data each minute
        start_date = (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        end_date = date.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            df = self.fetch_endpoint("cat-62", start_date, end_date)
        except Exception as e:
            print(f"Error fetching data for {date}: {e}")
            return None

        return df


def parse_metars(metar_strings):
    parsed_metar = []
    for metar_str in tqdm(
        metar_strings, total=len(metar_strings), desc="Parsing METARs"
    ):
        clean_metar_str, _ = MetarExtender.clean_metar_string(metar_str)

        try:
            obs = MetarExtender(clean_metar_str)
            metar_dict = MetarExtender.clean_metar_dict(obs.get_metar_dict())
        except:
            try:
                # If first try throws an error, try to remove the third word (METAR time)
                word = clean_metar_str.split(" ")
                clean_metar_str = " ".join(word[:2] + word[3:])
                obs = MetarExtender(clean_metar_str)
                metar_dict = MetarExtender.clean_metar_dict(obs.get_metar_dict())
            except Exception as e:
                print(f"Error parsing METAR: {metar_str} - {e}")
                continue

        metar_dict = metar_dict | {"original_metar": metar_str}

        parsed_metar.append(metar_dict)

    return pd.DataFrame(parsed_metar)


class MetarExtender(Metar.Metar):
    def __init__(self, metar_str):
        super().__init__(metar_str)

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
        pattern = r"\bR\d{2}[A-Z]?/////[^ ]*\s*"

        if re.search(pattern, metar_str):
            metar_str = re.sub(pattern, "", metar_str)
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
            "is_forecast": is_metaf,
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

    @staticmethod
    def clean_metar_dict(metar_dict):
        for key in ["temperature", "dew point", "pressure"]:
            if key in metar_dict:
                metar_dict[key] = float(metar_dict[key].split(" ")[0])

        if "time" in metar_dict:
            # Example of metar_dict["time"]: Fri Sep 15 06:00:00 2023
            metar_dict["time"] = pd.to_datetime(
                metar_dict["time"], format="%a %b %d %H:%M:%S %Y"
            )

        if "wind" in metar_dict:
            # Example of metar_dict["wind"]: E at 31 knots
            if re.match(r"\b[NEWS]+\b at \d+ knots", metar_dict["wind"]):
                metar_dict["wind_direction"] = metar_dict["wind"].split(" ")[0]
                metar_dict["wind_speed"] = int(metar_dict["wind"].split(" ")[2])
            # Example of metar_dict["wind"]: E to SSE at 10 knots
            elif re.match(r"\b[NEWS]+ to [NEWS]+ at \d+ knots\b", metar_dict["wind"]):
                metar_dict["wind_direction"] = metar_dict["wind"].split(" ")[2]
                metar_dict["wind_speed"] = int(metar_dict["wind"].split(" ")[-2])
            elif re.match(r"variable at \d+ knots", metar_dict["wind"]):
                metar_dict["wind_direction"] = "variable"
                metar_dict["wind_speed"] = int(metar_dict["wind"].split(" ")[-2])
            elif metar_dict["wind"] == "calm":
                metar_dict["wind_direction"] = "calm"
                metar_dict["wind_speed"] = 0

        if "visibility" in metar_dict:
            # Consider only the first visibility value, remove 'greater than' and get the value at first position
            metar_dict["visibility"] = (
                metar_dict["visibility"]
                .split(";")[0]
                .replace("greater than ", "")
                .split(" ")[0]
            )
            metar_dict["visibility"] = int(metar_dict["visibility"])

        return metar_dict