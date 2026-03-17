import pandas as pd
import joblib

# Load the cleaned dataset for study
def load_house_data():
    return pd.read_csv("outputs/datasets/collection/house_prices_records.csv")

# Load the specific 4 houses inherited by the client (Raw Data)
def load_inherited_house_data():
    return pd.read_csv("inputs/datasets/raw/house-price/inherited_houses.csv")

# Load the engineered inherited houses data
def load_engineered_inherited_house_data():
    return pd.read_csv("outputs/datasets/collection/inherited_houses_engineered.csv")

# Helper to load serialized objects (models, column lists, etc.)
def load_pkl_asset(file_path):
    return joblib.load(filename=file_path)