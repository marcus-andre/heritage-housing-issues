import streamlit as st
import pandas as pd
import joblib

@st.cache_data
def load_house_data():
    """
    Loads the cleaned housing dataset for Exploratory Data Analysis (EDA).
    Applies caching to prevent redundant disk reads during Streamlit interactions.
    """
    df = pd.read_csv("outputs/datasets/cleaned/cleaned_house_prices_records.csv")
    
    # Feature Engineering: Create 'HouseAge' required for the EDA study on Page 2
    if 'YearBuilt' in df.columns:
        df['HouseAge'] = 2010 - df['YearBuilt']
        
    return df

@st.cache_data
def load_inherited_house_data():
    """ Loads the raw dataset containing the 4 inherited houses. """
    return pd.read_csv("inputs/datasets/raw/house-price/inherited_houses.csv")

@st.cache_data
def load_engineered_inherited_house_data():
    """ Loads the pre-processed (engineered) inherited houses data. """
    return pd.read_csv("outputs/datasets/collection/inherited_houses_engineered.csv")

@st.cache_resource
def load_pkl_asset(file_path):
    """ Helper function to load serialized machine learning assets (Pipelines, Scalers). """
    return joblib.load(filename=file_path)