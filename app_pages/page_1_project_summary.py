import streamlit as st

def page_1_project_summary_body():
    """
    Renders the Project Summary page.
    Provides an overview of the dataset, terminology, and business requirements.
    """
    st.write("### 🏠 Quick Project Summary")

    # Explanatory text about the Dataset
    st.info(
        f"**Project Dataset**\n\n"
        f"The dataset represents housing records from Ames, Iowa. "
        f"It contains nearly 1500 houses' profiles with various attributes (such as Floor Area, "
        f"Basement, Garage, Kitchen, Lot, Year Built, and Overall Quality) and their respective sale prices."
    )

    with st.expander("📖 Project Terms & Jargon"):
        st.markdown(
            """
            * **SalePrice:** The **Target Variable**. The exact price a house sold for in USD.
            * **Features:** The attributes/characteristics of a property that influence its value.
            * **Machine Learning Pipeline:** A sequence of data processing steps (Cleaning, Feature Engineering, Scaling) and modeling used to predict the target.
            """
        )

    # Link to the GitHub repository
    st.write(
        f"🔗 For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/marcus-andre/heritage-housing-issues)."
    )

    st.write("---")

    # Business Requirements extracted directly from the Code Institute Handbook
    st.write("### 💼 Business Requirements")
    st.success(
        f"**1.** The client is interested in discovering how house attributes correlate with sale prices. "
        f"Therefore, the client expects data visualizations of the correlated variables against the sale price.\n\n"
        f"**2.** The client is interested in predicting the house sale prices from her 4 inherited houses, and any other house in Ames, Iowa."
    )