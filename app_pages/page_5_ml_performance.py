import streamlit as st
import pandas as pd
import joblib
import os

def page_5_ml_performance_body():
    st.write("### 📊 ML Performance Metrics")
    
    st.info(
        f"**Business Requirement 2:**\n"
        f"The client is interested in predicting the house sale prices from her 4 inherited houses, "
        f"and any other house in Ames, Iowa.\n\n"
        f"To satisfy this requirement, we trained a Machine Learning Pipeline using regression algorithms."
    )

    st.write("---")

    # Define the paths for the assets generated in the Jupyter Notebook
    version = 'v1'
    pipeline_path = f"outputs/ml_pipeline/predict_housing/{version}/linear_regression_pipeline.pkl"
    feat_importance_image = f"outputs/ml_pipeline/predict_housing/{version}/feature_importance.png"
    regression_evaluation_image = f"outputs/ml_pipeline/predict_housing/{version}/regression_evaluation_plots.png"

    # 0. Model Comparison
    st.write("### 🥇 Model Comparison")
    st.write(
        f"During the modeling phase, we evaluated different algorithms. As shown below, the simpler "
        f"**Linear Regression** unexpectedly outperformed the more complex ensemble models, delivering "
        f"the highest $R^2$ score and the lowest Mean Absolute Error (MAE)."
    )
    
    comparison_df = pd.DataFrame({
        "Model": ["Linear Regression (Baseline)", "Random Forest (Optimized)", "Decision Tree"],
        "R2 Score": [0.878, 0.839, 0.744],
        "MAE (Real $)": ["$18,394", "$20,007", "$26,659"]
    })
    st.table(comparison_df)

    st.write("---")

    # 1. Pipeline Summary
    st.write("### 🛠️ ML Pipeline Steps")
    st.write(
        f"Our modeling journey revealed that a **Linear Regression** model outperformed more complex "
        f"ensembles (like Random Forest) after thorough data cleaning and feature engineering. "
        f"Below is the exact pipeline used to transform data and make predictions."
    )
    
    try:
        if os.path.exists(pipeline_path):
            pipeline = joblib.load(pipeline_path)
            with st.expander("🔍 Inspect Machine Learning Pipeline"):
                st.write(pipeline)
        else:
            st.warning("Pipeline file not found. Please run the modeling notebook to generate it.")
    except Exception as e:
        st.error(f"Pipeline could not be loaded: {e}")

    st.write("---")

    # 2. Feature Coefficients (Importance)
    st.write("### 🌟 Feature Coefficients (Importance)")
    st.write(
        f"Since we are using a Linear Regression model, the most predictive features are represented "
        f"by their coefficients. This indicates how much the target variable (SalePrice) is expected "
        f"to change per unit change in the feature."
    )
    try:
        st.image(feat_importance_image, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Feature importance image not found at `{feat_importance_image}`.")

    st.write("---")

    # 3. Model Performance
    st.write("### 📈 Pipeline Performance")
    st.write(
        f"The model's performance is evaluated using standard regression metrics. "
        f"Our target was to achieve an $R^2$ score of **at least 0.75** on the test set."
    )

    st.success(
        f"**🎯 Model Evaluation Metrics (Linear Regression)**\n\n"
        f"**Train Set:**\n"
        f"* $R^2$ Score: **0.760**\n"
        f"* Mean Absolute Error (MAE): **$17,498.12**\n\n"
        f"**Test Set:**\n"
        f"* $R^2$ Score: **0.878**\n"
        f"* Mean Absolute Error (MAE): **$18,394.25**\n\n"
        f"*(Note: The model satisfies the project requirement of achieving an $R^2$ score of at least 0.75)*"
    )

    try:
        st.image(regression_evaluation_image, caption="Actual vs. Predicted Sale Prices", use_container_width=True)
    except Exception as e:
        st.info(f"*(Optional)* Model evaluation plots not found at `{regression_evaluation_image}`.")