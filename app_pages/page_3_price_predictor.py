import streamlit as st
import numpy as np
from src.data_management import load_inherited_house_data, load_engineered_inherited_house_data, load_pkl_asset

def page_3_inherited_houses_body():
    st.write("### 🏠 Inherited Houses Price Prediction")
    
    st.info(
        f"**Business Requirement 2:**\n"
        f"The client is interested in predicting the sale price of her four inherited houses."
    )

    # Load raw data for display
    inherited_df_raw = load_inherited_house_data()
    
    st.write("#### Features of the inherited houses (Raw Data)")
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
            # Filter features to match the training dataset order
            inherited_df_filtered = inherited_df_engineered.filter(price_features)
            
            # Predict sale prices (returns logarithmic values, e.g., ~12.2)
            log_predictions = pipeline.predict(inherited_df_filtered)
            
            # Reverse log transformation to get real US Dollars
            real_predictions = np.expm1(log_predictions)
            
            # Append predictions to the display dataframe
            inherited_df_raw['Predicted_SalePrice'] = real_predictions.round(2)
            
            st.write("---")
            st.success("#### Prediction Results")
            st.dataframe(inherited_df_raw[['Predicted_SalePrice']])
            
            # Calculate and display the total portfolio value
            total_value = inherited_df_raw['Predicted_SalePrice'].sum()
            st.metric(label="Total Portfolio Value", value=f"${total_value:,.2f}")
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.warning("Ensure the engineered dataset exists and matches the expected features.")