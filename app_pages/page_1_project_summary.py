import streamlit as st

def page_1_project_summary_body():
    st.write("### 🏠 Quick Project Summary")

    # Explanatory text about the Dataset
    st.info(
        f"**Project Dataset**\n\n"
        f"The dataset represents housing records from Ames, Iowa. "
        f"It contains nearly 1500 houses' profiles with various attributes (such as Floor Area, "
        f"Basement, Garage, Kitchen, Lot, Year Built, and Overall Quality) and their respective sale prices.\n\n"
        f"**Project Terms & Jargon:**\n"
        f"* **SalePrice:** The target variable. The exact price a house sold for in USD.\n"
        f"* **Features:** The attributes/characteristics of a property that influence its value."
    )

    # Link to the GitHub repository
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/marcus-andre/heritage-housing-issues)."
    )

    st.write("---")

    # Business Requirements extracted directly from the Code Institute Handbook
    st.success(
        f"**Business Requirements**\n\n"
        f"**1.** The client is interested in discovering how house attributes correlate with sale prices. "
        f"Therefore, the client expects data visualizations of the correlated variables against the sale price.\n\n"
        f"**2.** The client is interested in predicting the house sale prices from her 4 inherited houses, and any other house in Ames, Iowa."
    )