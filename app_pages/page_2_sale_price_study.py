import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_management import load_house_data

def page_2_sale_price_study_body():
    
    # Load the house records data
    df = load_house_data()

    # Variables with the highest correlation to SalePrice
    vars_to_study = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'HouseAge', '1stFlrSF']

    st.write("### 📊 House Sale Price Study")
    
    # Restate the Business Requirement
    st.info(
        f"**Business Requirement 1:**\n"
        f"The client is interested in discovering how house attributes correlate with sale prices. "
        f"Therefore, the client expects data visualizations of the correlated variables against the sale price."
    )

    # 1. Checkbox to inspect the raw house records
    if st.checkbox("Inspect House Records Data"):
        st.write(f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

        # Nested Checkbox for the Data Dictionary
        if st.checkbox("Show Data Dictionary and Sample Records"):
            st.write("---")
            st.write("**Data Dictionary (What do these columns mean?)**")
            
            # Split dictionary into two columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                * **1stFlrSF**: First Floor square feet.
                * **2ndFlrSF**: Second Floor square feet.
                * **BedroomAbvGr**: Bedrooms above grade.
                * **BsmtExposure**: Walkout or garden level walls.
                * **BsmtFinType1**: Rating of basement area.
                * **BsmtFinSF1**: Type 1 finished square feet.
                * **BsmtUnfSF**: Unfinished basement area.
                * **TotalBsmtSF**: Total basement area.
                * **GarageArea**: Size of garage.
                * **GarageFinish**: Interior finish of garage.
                * **HouseAge**: Age of the house (Replaced YearBuilt).
                """)
                
            with col2:
                st.markdown("""
                * **GrLivArea**: Above grade living area.
                * **KitchenQual**: Kitchen quality.
                * **LotArea**: Lot size.
                * **LotFrontage**: Linear feet of street connected.
                * **MasVnrArea**: Masonry veneer area.
                * **EnclosedPorch** / **OpenPorchSF**: Porch areas.
                * **OverallCond**: Rates the overall condition.
                * **OverallQual**: Rates the material and finish.
                * **WoodDeckSF**: Wood deck area.
                * **YearRemodAdd**: Remodel date.
                * **SalePrice**: Final sale price (Target).
                """)
            
            st.write("---")
            st.write("### Dataset Sample")
            st.dataframe(df.head(10))

            # Caption for the table
            st.caption(
                f"**Table 1: Ames Housing Data Sample.**\n"
                f"Displays the first 10 records of the pre-processed dataset. "
                f"Each row represents a distinct property, and the columns contain the respective features evaluated, "
                f"including the target variable (SalePrice)."
            )

    st.write("---")

    # 2. Checkbox for the Correlation Study (Bar Plot)
    if st.checkbox("Correlation Study"):
        st.write("### Analysis of Variable Correlation")
        st.write(
            f"The following variables were found to have the strongest impact on Sale Price: \n"
            f"**{vars_to_study}**"
        )
        
        if st.checkbox("Plot Correlation vs Sale Price"):
            # Calculate Spearman correlation against SalePrice, sort descending
            corr_spearman = df.corr(method='spearman', numeric_only=True)['SalePrice'].sort_values(ascending=False)
            
            # Drop SalePrice itself from the list (correlation of 1.0) and get top 10
            corr_spearman = corr_spearman.drop('SalePrice').head(10)

            # Replicate the bar plot from the EDA Jupyter Notebook
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=corr_spearman.values, y=corr_spearman.index, ax=ax, color='#1f77b4')
            ax.set_title("Key Numerical Value Drivers: Spearman Correlation vs SalePrice")
            ax.set_xlabel("Spearman Correlation Coefficient")
            st.pyplot(fig)
            plt.close(fig) # Prevent memory leak
            
            st.info(
                f"**Insights:**\n"
                f"* The bar plot clearly shows which features have the strongest positive correlation with the final price.\n"
                f"* Variables like **OverallQual** and **GrLivArea** dominate the top of the chart, confirming their importance."
            )

    # 3. EDA Section (Visualizing the findings)
    if st.checkbox("Exploratory Data Analysis (EDA)"):
        st.write("### Visualizing Relationship: Feature vs Sale Price")
        st.write("The plots below illustrate how the most important features relate to the final property value.")
        
        target_var = 'SalePrice'
        
        # Plotting each variable against the target
        for col in vars_to_study:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df, x=col, y=target_var, alpha=0.5, ax=ax)
            ax.set_title(f"{col} vs {target_var}")
            st.pyplot(fig)
            
            # Caption for each plot
            if col in ['GarageArea', 'TotalBsmtSF']:
                st.caption(
                    f"**Figure: {col} vs {target_var}**.\n"
                    f"The vertical cluster of points represents houses completely lacking this feature (real value = 0). "
                    f"The negative position on the X-axis is an artifact of the data standardization process."
                )
            elif col == 'OverallQual':
                st.caption(
                    f"**Figure: {col} vs {target_var}**.\n"
                    f"Demonstrates a distinct 'step' pattern where each incremental increase in the quality rating "
                    f"corresponds to a visible jump in the baseline sale price."
                )
            elif col == 'HouseAge':
                st.caption(
                    f"**Figure: {col} vs {target_var}**.\n"
                    f"Demonstrates a **negative correlation**: as the house age increases (moving to the right), "
                    f"the sale price generally decreases. Newer houses (lower age) command the highest prices."
                )
            else:
                st.caption(
                    f"**Figure: {col} vs {target_var}**.\n"
                    f"Demonstrates a positive correlation: as the feature's value increases, the property's sale price generally rises."
                )
                
            st.write("---")
            plt.close(fig) # For Streamlit performance
            
        st.success(
            f"**Key Findings Summary:**\n"
            f"* A clear positive trend is visible across all studied features.\n"
            f"* Overall Quality has the most distinct pricing tiers.\n"
            f"* Properties completely lacking a basement or garage form clear zero-inflation clusters but still maintain baseline values determined by their other attributes."
        )