import streamlit as st
import plotly.express as px
from src.data_management import load_house_data

def page_4_project_hypotheses_body():
    """
    Renders the Project Hypotheses page.
    Displays the 3 project hypotheses formulated during the planning phase
    and details how they were validated through Exploratory Data Analysis (EDA).
    """
    # Load data for visualizations
    df = load_house_data()

    st.write("### 💡 Project Hypotheses and Validation")

    st.info(
        f"During the project planning phase, we formulated **three main hypotheses** regarding the "
        f"Ames housing market. This page details these hypotheses and how they were scientifically "
        f"validated using data analytics and statistical correlation studies."
    )

    st.write("---")

    # Hypothesis 1
    st.write("### Hypothesis 1: Property Size vs. Sale Price")
    st.write(
        f"* **We suspect that:** The overall size of a property (specifically `GrLivArea` and `TotalBsmtSF`) "
        f"has a strong positive linear correlation with the Target Variable (`SalePrice`)."
    )
    st.success(
        f"**Validation: Confirmed ✅**\n\n"
        f"* Through Exploratory Data Analysis (EDA) on the 'Sale Price Study' page, we validated this using "
        f"Pearson and Spearman correlation matrices, alongside interactive scatter plots.\n"
        f"* **Conclusion:** Both `GrLivArea` (Above ground living area) and `TotalBsmtSF` (Basement area) showed "
        f"a clear upward trend, proving that larger properties command higher market values."
    )

    # Visual Evidence for Hypothesis 1
    with st.expander("📈 View Visual Evidence"):
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.scatter(df, x='GrLivArea', y='SalePrice', title="Above Ground Area vs Price", opacity=0.5), use_container_width=True)
        with col2:
            st.plotly_chart(px.scatter(df, x='TotalBsmtSF', y='SalePrice', title="Basement Area vs Price", opacity=0.5), use_container_width=True)

    st.write("---")

    # Hypothesis 2
    st.write("### Hypothesis 2: Property Quality vs. Sale Price")
    st.write(
        f"* **We suspect that:** The `OverallQual` (which rates the overall material and finish of the house) "
        f"is the most predictive categorical/ordinal feature for the Target Variable."
    )
    st.success(
        f"**Validation: Confirmed ✅**\n\n"
        f"* We validated this hypothesis using Spearman correlation and Box Plots.\n"
        f"* **Conclusion:** The Box Plot analysis revealed distinct, non-overlapping pricing tiers. "
        f"Every incremental increase in the `OverallQual` rating corresponds to a visible and significant jump "
        f"in the baseline sale price, making it one of the strongest drivers of value in the dataset."
    )

    # Visual Evidence for Hypothesis 2
    with st.expander("📈 View Visual Evidence"):
        fig2 = px.box(df, x='OverallQual', y='SalePrice', color='OverallQual', title="Overall Quality vs Sale Price")
        st.plotly_chart(fig2, use_container_width=True)

    st.write("---")

    # Hypothesis 3
    st.write("### Hypothesis 3: Property Age vs. Sale Price")
    st.write(
        f"* **We suspect that:** Newer houses (`YearBuilt`) or recently remodeled houses (`YearRemodAdd`) "
        f"hold significantly higher market value (Target Variable) than older, unrenovated ones."
    )
    st.success(
        f"**Validation: Confirmed ✅**\n\n"
        f"* We utilized **Feature Engineering** to create a `HouseAge` variable and validated the hypothesis using scatter plots.\n"
        f"* **Conclusion:** The EDA showed a **negative correlation** between house age and sale price. "
        f"As the house age increases, the sale price generally decreases. Newer properties, or those recently "
        f"remodeled, consistently cluster at the higher end of the pricing spectrum."
    )

    # Visual Evidence for Hypothesis 3
    with st.expander("📈 View Visual Evidence"):
        fig3 = px.scatter(df, x='HouseAge', y='SalePrice', title="House Age vs Sale Price", opacity=0.5, trendline=None)
        st.plotly_chart(fig3, use_container_width=True)