# %%
import requests
import pandas as pd
import re  # Regex
import time
import os
from metar import Metar  # METAR parsing library
import openai  # OpenAI API
import json
import aiofiles  # Async file writing
import asyncio  # Async requests
from tqdm import tqdm  # Progress bar
from shapely.geometry import Point, MultiPoint  # Geospatial data


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
    Dew Point Spread, Altimeter Setting, Precipitation, Temperature. If data is insufficient 
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

    async def main(self, metar_reports, output_directory, save_interval):
        tasks = [self.fetch_scores_for_metar(metar) for metar in metar_reports]

        print("Progress:")
        counter = 0
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Fetching scores from LLM",
        ):
            await task
            counter += 1
            if counter % save_interval == 0:
                # Save the results to a JSON file asynchronously every X iterations
                await self.save_to_file(
                    self.results_dict,
                    f"{output_directory}/metar_results_{counter}.json",
                )
                # Delete the file from 3 iterations ago. This is to avoid filling up the disk
                self.delete_old_file(
                    output_directory,
                    f"metar_results_{counter - (save_interval*3)}.json",
                )

            await asyncio.sleep(0.5)

        # Now, results_dict contains METAR data
        print("\nFinished fetching data.")

        # Save the results to a JSON file asynchronously
        await self.save_to_file(
            self.results_dict, f"{output_directory}/metar_results.json"
        )

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
