import pandas as pd
import joblib

# Load the cleaned dataset for study
def load_house_data():
    # Load the cleaned dataset from outputs (whitelisted in .gitignore for production)
    df = pd.read_csv("outputs/datasets/cleaned/cleaned_house_prices_records.csv")
    
    # Engineer 'HouseAge' feature required for the EDA study on Page 2
    if 'YearBuilt' in df.columns:
        df['HouseAge'] = 2010 - df['YearBuilt']
        
    return df

# Load the specific 4 houses inherited by the client (Raw Data)
def load_inherited_house_data():
    return pd.read_csv("inputs/datasets/raw/house-price/inherited_houses.csv")

# Load the engineered inherited houses data
def load_engineered_inherited_house_data():
    return pd.read_csv("outputs/datasets/collection/inherited_houses_engineered.csv")

# Helper to load serialized objects (models, column lists, etc.)
def load_pkl_asset(file_path):
    return joblib.load(filename=file_path)