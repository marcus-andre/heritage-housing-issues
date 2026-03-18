import streamlit as st
import numpy as np
import pandas as pd
from src.data_management import load_inherited_house_data, load_engineered_inherited_house_data, load_pkl_asset

def page_3_inherited_houses_body():
    """
    Renders the UI for predicting inherited house prices and custom properties.
    Loads the trained Machine Learning Pipeline and applies it to the
    engineered dataset to predict the target variable (SalePrice).
    """
    st.write("### 🏠 Inherited Houses Price Prediction")
    
    st.info(
        f"**Business Requirement 2:**\n"
        f"The client is interested in predicting the sale price of her four inherited houses."
    )

    # Load raw data for display
    inherited_df_raw = load_inherited_house_data()
    
    with st.expander("🔎 Inspect Features of Inherited Houses"):
        st.write("* These are the raw attributes of the 4 houses inherited by the client.")
        st.dataframe(inherited_df_raw)

    # Define file paths
    version = 'v1'
    pipeline_path = f"outputs/ml_pipeline/predict_housing/{version}/linear_regression_pipeline.pkl"
    columns_path = f"outputs/ml_pipeline/predict_housing/{version}/train_columns.pkl"

    # Load ML pipeline, feature list, and engineered data
    pipeline = load_pkl_asset(pipeline_path)
    price_features = load_pkl_asset(columns_path)
    inherited_df_engineered = load_engineered_inherited_house_data()

    if st.button("Predict Sale Prices"):
        try:
            # Enforce exact feature order to match the training dataset.
            # Using list indexing instead of .filter() guarantees reordering
            # and prevents 'Silent Fails' or Scikit-Learn feature name errors.
            inherited_df_filtered = inherited_df_engineered[price_features]
            
            # Predict sale prices using the ML Pipeline (returns logarithmic values, e.g., ~12.2)
            log_predictions = pipeline.predict(inherited_df_filtered)
            
            # Reverse log transformation to get real US Dollars
            real_predictions = np.expm1(log_predictions)
            
            # Append predictions to the display dataframe
            inherited_df_raw['Predicted_SalePrice'] = real_predictions.round(2)
            
            st.write("---")
            st.success("#### 💰 Portfolio Prediction Results")
            
            # Calculate and display the total portfolio value
            total_value = inherited_df_raw['Predicted_SalePrice'].sum()
            st.metric(label="Total Portfolio Value", value=f"${total_value:,.2f}")
            
            st.dataframe(inherited_df_raw[['Predicted_SalePrice']])
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.warning("Ensure the engineered dataset exists and matches the expected features.")
            
    st.write("---")
    
    # Custom Property Prediction Section Header
    st.write("### 🛠️ Predict Custom Property Price")
    st.info(
        f"This tool allows you to predict the sale price of any custom property in Ames, Iowa. "
        f"By adjusting the widgets below, your input is processed through the exact same **Machine Learning Pipeline** "
        f"— including Feature Engineering and Scaling — used for the inherited houses, ensuring accurate and reliable results."
    )
    
    with st.expander("📖 How to use this prediction tool"):
        st.write(
            """
            **Step-by-Step Guide:**
            1. **Adjust the Widgets:** Use the number inputs and dropdown menus below to define the characteristics of the property. 
               * *Note:* The input constraints (minimum and maximum values) are strictly set based on the historical training data. This ensures the model's predictions remain reliable and prevents out-of-bounds errors.
            2. **Understand the Ratings:** 
               * `OverallQual` and `OverallCond` are rated from 1 (Very Poor) to 10 (Very Excellent).
               * Categorical dropdowns use standard abbreviations: **Ex** (Excellent), **Gd** (Good), **TA** (Typical/Average), **Fa** (Fair), **Po** (Poor).
               * `None` indicates the absence of a feature (e.g., no basement or no garage).
            3. **Run the Prediction:** Once you are satisfied with the property profile, click the **'Predict Custom House Price'** button below the widgets.
            4. **Review the Output:** The application will instantly process your live data through the ML pipeline and display the estimated sale price in US Dollars.
            """
        )

    # Render the interactive input widgets and capture the live data
    X_live = DrawInputsWidgets()
    
    # Button to predict the custom live data
    if st.button("Predict Custom House Price"):
        try:
            # Process the live data mimicking Notebook 06 steps
            X_live_processed = process_live_data(X_live, price_features)
            
            # Predict using the ML Pipeline
            log_prediction = pipeline.predict(X_live_processed)
            
            # Reverse log transformation
            real_prediction = np.expm1(log_prediction[0])
            
            st.success(f"#### Predicted Custom Property Price: ${real_prediction:,.2f}")
        except Exception as e:
            st.error(f"Error during custom prediction: {e}")


def DrawInputsWidgets():
    """
    Creates interactive Streamlit input widgets for housing features.
    Organizes the UI into columns and captures user input for real-time prediction.
    Returns a Pandas DataFrame containing the user's selected values, structured
    to be eventually processed by the Machine Learning Pipeline.
    """
    st.write("### Property Features Input")
    st.write("Adjust the features below to define a custom property profile.")
    
    # Divide the layout into 4 columns to optimize screen space
    col1, col2, col3, col4 = st.columns(4)

    # 1st Column: Floor & Living Area
    with col1:
        st.write("**Floor & Living Area**")
        GrLivArea = st.number_input("GrLivArea (sq ft)", min_value=334, max_value=5642, value=1500)
        FirstFlrSF = st.number_input("1stFlrSF (sq ft)", min_value=334, max_value=4692, value=1000)
        SecondFlrSF = st.number_input("2ndFlrSF (sq ft)", min_value=0, max_value=2065, value=0)
        BedroomAbvGr = st.number_input("BedroomAbvGr", min_value=0, max_value=8, value=3)
        KitchenQual = st.selectbox("KitchenQual", ['Ex', 'Gd', 'TA', 'Fa', 'Po'], index=2) # TA as default
        OverallQual = st.number_input("OverallQual (1-10)", min_value=1, max_value=10, value=5)

    # 2nd Column: Basement Features
    with col2:
        st.write("**Basement Features**")
        TotalBsmtSF = st.number_input("TotalBsmtSF (sq ft)", min_value=0, max_value=6110, value=1000)
        BsmtFinSF1 = st.number_input("BsmtFinSF1 (sq ft)", min_value=0, max_value=5644, value=500)
        BsmtUnfSF = st.number_input("BsmtUnfSF (sq ft)", min_value=0, max_value=2336, value=500)
        BsmtFinType1 = st.selectbox("BsmtFinType1", ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'None'], index=5)
        BsmtExposure = st.selectbox("BsmtExposure", ['Gd', 'Av', 'Mn', 'No', 'None'], index=3)

    # 3rd Column: Lot & Outdoor Features
    with col3:
        st.write("**Lot & Outdoor**")
        LotArea = st.number_input("LotArea (sq ft)", min_value=1300, max_value=215245, value=10000)
        LotFrontage = st.number_input("LotFrontage (ft)", min_value=21, max_value=313, value=70)
        WoodDeckSF = st.number_input("WoodDeckSF (sq ft)", min_value=0, max_value=736, value=0)
        OpenPorchSF = st.number_input("OpenPorchSF (sq ft)", min_value=0, max_value=547, value=0)
        EnclosedPorch = st.number_input("EnclosedPorch (sq ft)", min_value=0, max_value=286, value=0)
        MasVnrArea = st.number_input("MasVnrArea (sq ft)", min_value=0, max_value=1600, value=0)

    # 4th Column: Garage & History
    with col4:
        st.write("**Garage & History**")
        GarageArea = st.number_input("GarageArea (sq ft)", min_value=0, max_value=1418, value=400)
        GarageFinish = st.selectbox("GarageFinish", ['Fin', 'RFn', 'Unf', 'None'], index=2)
        GarageYrBlt = st.number_input("GarageYrBlt", min_value=1900, max_value=2010, value=1990)
        OverallCond = st.number_input("OverallCond (1-10)", min_value=1, max_value=10, value=5)
        YearBuilt = st.number_input("YearBuilt", min_value=1872, max_value=2010, value=1990)
        YearRemodAdd = st.number_input("YearRemodAdd", min_value=1950, max_value=2010, value=1990)

    # Compile data into a single-row DataFrame, matching EXACT original dataset order
    X_live = pd.DataFrame({
        '1stFlrSF': [FirstFlrSF],
        '2ndFlrSF': [SecondFlrSF],
        'BedroomAbvGr': [BedroomAbvGr],
        'BsmtExposure': [BsmtExposure],
        'BsmtFinSF1': [BsmtFinSF1],
        'BsmtFinType1': [BsmtFinType1],
        'BsmtUnfSF': [BsmtUnfSF],
        'EnclosedPorch': [EnclosedPorch],
        'GarageArea': [GarageArea],
        'GarageFinish': [GarageFinish],
        'GarageYrBlt': [GarageYrBlt],
        'GrLivArea': [GrLivArea],
        'KitchenQual': [KitchenQual],
        'LotArea': [LotArea],
        'LotFrontage': [LotFrontage],
        'MasVnrArea': [MasVnrArea],
        'OpenPorchSF': [OpenPorchSF],
        'OverallCond': [OverallCond],
        'OverallQual': [OverallQual],
        'TotalBsmtSF': [TotalBsmtSF],
        'WoodDeckSF': [WoodDeckSF],
        'YearBuilt': [YearBuilt],
        'YearRemodAdd': [YearRemodAdd]
    })

    return X_live


def process_live_data(X_live, price_features):
    """
    Applies Data Cleaning, Feature Engineering, Log Transformations, 
    and Scaling to the live user data.
    Prevents Data Leakage by using only the fitted scaler's .transform() method.
    """
    df = X_live.copy()
    
    # 1. Drop columns that were excluded during training
    cols_to_drop = ['EnclosedPorch', 'WoodDeckSF']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # 2. Ordinal Encoding
    mapping_quality = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    mapping_exposure = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    mapping_fin_type = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    mapping_garage = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
    
    df['KitchenQual'] = df['KitchenQual'].map(mapping_quality)
    df['BsmtExposure'] = df['BsmtExposure'].map(mapping_exposure)
    df['BsmtFinType1'] = df['BsmtFinType1'].map(mapping_fin_type)
    df['GarageFinish'] = df['GarageFinish'].map(mapping_garage)
    
    # 3. Age Features (Reference year 2010 to match training)
    reference_year = 2010
    df['HouseAge'] = reference_year - df['YearBuilt']
    df['RemodelAge'] = reference_year - df['YearRemodAdd']
    df['GarageAge'] = reference_year - df['GarageYrBlt']
    
    columns_to_drop_years = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
    df = df.drop(columns=columns_to_drop_years, axis=1)
    
    # 4. Log Transformations
    skewed_features = [
        'LotArea', 'MasVnrArea', 'LotFrontage', 'OpenPorchSF',
        'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea',
        'BsmtExposure', 'BsmtUnfSF', '2ndFlrSF'
    ]
    for feature in skewed_features:
        if feature in df.columns:
            df[feature] = np.log1p(df[feature])
            
    # 5. Feature Scaling (Using the fitted scaler to prevent Data Leakage)
    scaler_path = "outputs/ml_pipeline/predict_housing/v1/scaler.pkl"
    scaler = load_pkl_asset(scaler_path)
    
    features_to_scale = df.select_dtypes(include=['int64', 'float64']).columns
    df[features_to_scale] = scaler.transform(df[features_to_scale])
    
    # 6. Enforce exact matching of train columns to prevent 'Silent Fails'
    return df[price_features]
