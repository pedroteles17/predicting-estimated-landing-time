#%%
import pandas as pd
from tqdm import tqdm
from utils import MergeDataSets

#%%
final_df = pd.read_parquet("data/feature_engineering/clean_data.parquet")

#cat_62 = pd.read_parquet("data/original_files/cat-62.parquet")

#cat_62["dt_radar"] = pd.to_datetime(cat_62["dt_radar"], unit="ms")

satelite = pd.read_parquet("data/original_files/satelite.parquet")

satelite["data"] = pd.to_datetime(satelite["data"])

#%%

# %%
