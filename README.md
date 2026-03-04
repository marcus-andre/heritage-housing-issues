# Heritage Housing Issues - Predictive Analytics

## 1. Business Requirements
As a Data Analyst for Code Institute Consulting, I was requested by a client to provide actionable insights into the housing market of Ames, Iowa. The client has inherited four properties and requires a data-driven approach to maximize her inheritance value.

* **Business Requirement 1:** The client is interested in discovering how house attributes (features) correlate with the sale price. The client expects data visualizations of the correlated variables against the sale price to easily interpret the insights.
* **Business Requirement 2:** The client is interested in predicting the house sale price from her four inherited houses, and any other house in Ames, Iowa, to make informed real estate decisions.

## 2. Project Hypotheses and Validation
To address the business requirements, the following three hypotheses were formulated and will be validated through Exploratory Data Analysis (EDA):

* **Hypothesis 1:** We suspect that the overall size of a property (e.g., `GrLivArea`, `TotalBsmtSF`) has a strong positive linear correlation with the `SalePrice`. 
    * *Validation:* We will validate this by plotting a Pearson correlation matrix and creating scatter plots to visualize the trend.
* **Hypothesis 2:** We suspect that the `OverallQual` (Rates the overall material and finish of the house) is the most predictive categorical/ordinal feature for the target variable.
    * *Validation:* We will validate this using Spearman correlation and box plots to show price distribution across different quality tiers.
* **Hypothesis 3:** We suspect that newer houses (`YearBuilt`) or recently remodeled houses (`YearRemodAdd`) hold significantly higher market value than older, unrenovated ones.
    * *Validation:* We will validate this using scatter plots with regression lines to observe the price decay over time.

## 3. The Rationale to Map Business Requirements to ML Tasks
* **BR1 (Data Visualization):** We will implement a correlation study (Pearson and Spearman) to identify the most relevant variables. We will build an interactive dashboard page displaying at least 4 different types of plots (e.g., scatter, box, heatmap, bar) to represent these data stories visually.
* **BR2 (Predictive Modeling):** We will develop a Supervised Machine Learning model. Specifically, a Regression task to predict a continuous target (`SalePrice`). We will perform extensive Hyperparameter Optimization to ensure maximum predictive accuracy.

## 4. ML Business Case
* **Problem Definition:** We want an ML model to predict the `SalePrice` of houses in Ames, Iowa. It is a Supervised, uni-dimensional Regression model.
* **Ideal Outcome:** Provide the client with a reliable, interactive dashboard to estimate property values instantly.
* **Success Metrics:** The agreed performance goal is an $R^2$ score of 0.75 or higher on both the train and test sets.
* **Model Output:** A continuous numerical value representing the predicted property price in USD.
* **Training Data:** The data was sourced from a public Kaggle dataset (Ames Housing Dataset) containing nearly 1500 house records with various independent variables (features) and the final sale price (target).

## 5. Dashboard Design (Streamlit UI)
The application will consist of the following pages:
* **Page 1: Quick Project Summary:** Describes the project dataset, the client's requirements, and general information about the Ames housing market.
* **Page 2: House Sale Price Study:** Addresses BR1. It contains dynamic data visualizations (histograms, scatter plots, correlation heatmaps) and a checkbox to display the raw data.
* **Page 3: Price Predictor:** Addresses BR2. It displays the predicted price for the 4 inherited houses and contains an interactive set of widgets for the user to input custom property features and receive a real-time price prediction.
* **Page 4: Project Hypotheses:** Displays the 3 business hypotheses and the conclusions derived from the EDA.
* **Page 5: ML Performance Metrics:** A technical page detailing the ML pipeline, feature importance, and model evaluation metrics ($R^2$ and Actual vs. Predicted plots).