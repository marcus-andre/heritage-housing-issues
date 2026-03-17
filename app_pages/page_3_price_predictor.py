import streamlit as st
from src.data_management import load_inherited_house_data, load_pkl_asset

def page_3_inherited_houses_body():
    
    # Load the specific 4 houses inherited by the client
    inherited_df = load_inherited_house_data()
    
    # Define model version and paths
    version = 'v1'
    pipeline_path = f"outputs/ml_pipeline/predict_housing/{version}/linear_regression_pipeline.pkl"
    columns_path = f"outputs/ml_pipeline/predict_housing/{version}/train_columns.pkl"

    # Load the winning Linear Regression pipeline and the required feature list
    pipeline = load_pkl_asset(pipeline_path)
    price_features = load_pkl_asset(columns_path)

    st.write("### 🏠 Inherited Houses Price Prediction")
    
    st.info(
        f"**Business Requirement 2:**\n"
        f"The client is interested in predicting the sale price of her four inherited houses."
    )

    st.write("#### Features of the inherited houses")
    st.dataframe(inherited_df)

    if st.button("Predict Sale Prices"):
        # 1. Handling Missing Values (Simple fix to prevent crashes)
        # If your model doesn't accept NaNs, we fill them with 0 or the median
        inherited_df = inherited_df.fillna(0)

        # 2. FEATURE ENGINEERING: Creating the missing age columns
        # These are usually expected by the model if you created them in the EDA
        current_year = 2010 # Base year of the Ames dataset
        inherited_df['HouseAge'] = current_year - inherited_df['YearBuilt']
        inherited_df['RemodelAge'] = current_year - inherited_df['YearRemodAdd']
        inherited_df['GarageAge'] = current_year - inherited_df['GarageYrBlt']

        # 3. Ensure we only pass columns the model knows
        # price_features contains the list of features from your training
        inherited_df_filtered = inherited_df.filter(price_features)
        
        # 4. Predict
        try:
            predictions = pipeline.predict(inherited_df_filtered)
            inherited_df['Predicted_SalePrice'] = predictions.round(2)
            
            st.write("---")
            st.success("#### Prediction Results")
            st.dataframe(inherited_df[['Predicted_SalePrice']])
            
            total_value = inherited_df['Predicted_SalePrice'].sum()
            st.metric(label="Total Portfolio Value", value=f"${total_value:,.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.warning("This usually happens when columns are missing or in the wrong format.")