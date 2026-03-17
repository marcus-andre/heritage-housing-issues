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

### 4.1 Architectural Decision: Explicit Feature Selection Impact
During the modeling phase, an architectural experiment was conducted to evaluate the impact of explicit automated feature selection (`SelectFromModel`) across different algorithms (Linear Regression and Random Forest).

* **The Experiment:** We tested both algorithms with and without `SelectFromModel` acting as a strict filter.
* **The Conclusion (The Winning Architecture):** The experiment conclusively proved that forcing explicit feature selection actually *degraded* the predictive performance of both the Linear Regression and the Random Forest models. Because rigorous feature selection and scaling were already performed manually during the Exploratory Data Analysis (EDA) phase, the input dataset was already highly optimized. Both algorithms achieved their peak performance (Linear Regression achieving an $R^2$ of 0.878) by utilizing all provided features and relying on their internal mathematical weighting to naturally discard any remaining noise. Therefore, the simpler, unfiltered pipeline was chosen for the final production deployment.

## 5. Dashboard Design (Streamlit UI)
The application will consist of the following pages:
* **Page 1: Quick Project Summary:** Describes the project dataset, the client's requirements, and general information about the Ames housing market.
* **Page 2: House Sale Price Study:** Addresses BR1. It contains dynamic data visualizations (histograms, scatter plots, correlation heatmaps) and a checkbox to display the raw data.
* **Page 3: Price Predictor:** Addresses BR2. It displays the predicted price for the 4 inherited houses and contains an interactive set of widgets for the user to input custom property features and receive a real-time price prediction.
* **Page 4: Project Hypotheses:** Displays the 3 business hypotheses and the conclusions derived from the EDA.
* **Page 5: ML Performance Metrics:** A technical page detailing the ML pipeline, feature importance, and model evaluation metrics ($R^2$ and Actual vs. Predicted plots).

## 6. Technologies and Libraries Used
The following tools and libraries were utilized within the virtual environment to develop this project:

* **numpy (1.26.1):** Used for efficient multi-dimensional array operations and mathematical calculations (e.g., calculating logarithmic inversions using `np.expm1`).
* **pandas (2.1.1):** Essential for data manipulation, aggregation, and DataFrame creation during the Data Cleaning and Feature Engineering phases.
* **matplotlib (3.8.0) & seaborn (0.13.2):** Used for static data visualization, creating correlation heatmaps, scatter plots, and distribution graphs during EDA. In the Streamlit dashboard, explicit figure closure was implemented to ensure optimal server memory management.
* **plotly (5.17.0):** Used to build interactive charts and plots for the Streamlit dashboard.
* **streamlit (1.40.2):** The core framework used to develop and deploy the interactive web dashboard interface.
* **kaggle (1.5.16):** Used to securely authenticate and download the Ames Housing dataset directly from the Kaggle API.
* **scikit-learn (1.3.2):** The primary machine learning library used for building pipelines, feature scaling (`StandardScaler`), model training (`LinearRegression`, `RandomForestRegressor`), hyperparameter optimization (`GridSearchCV`), and calculating performance metrics ($R^2$, MAE).

## 7. 🐛 Fixed Bugs & Issues

**Issue 1: Feature Order Mismatch (Silent Fail Prevention)**
* **Bug:** Machine learning models require the exact same feature order during live prediction as they had during training. If the Streamlit dashboard passes user input in a different sequence, the model will output completely incorrect predictions without throwing any visible system errors (a highly dangerous silent fail).
* **Fix:** During the modeling phase (`05_Modeling.ipynb`), I extracted and serialized the training columns order into a `train_columns.pkl` file. The Streamlit app will load this file and use the `.filter()` command to enforce the correct column sequence on the live data before passing it to the prediction pipeline.

**Issue 2: Notebook Sequential Execution State**
* **Bug:** During development, I noticed that loading pre-trained models from the disk to save time broke the logical flow for a fresh "Run All" execution. If an evaluator ran the notebook from top to bottom, it would create state conflicts between the freshly trained memory variables and the loaded disk files.
* **Fix:** Refactored the `05_Modeling.ipynb` architecture to ensure that the newly trained models are passed purely in-memory directly to the comparison step. This guarantees that evaluators can execute the notebook end-to-end seamlessly, establishing a perfect continuous integration flow.

**Issue 3: Incorrect Model Serialization for Production**
* **Bug:** During the final step of the modeling phase (`05_Modeling.ipynb`), the `RandomForestRegressor` was accidentally serialized (`.pkl`) instead of the winning `LinearRegression` model. This would result in the Streamlit dashboard using a sub-optimal model with a lower $R^2$ score (0.749 instead of 0.878) for live predictions.
* **Fix:** I identified the mismatch between the documented winning model and the serialized variable. I corrected the output code to explicitly call `joblib.dump()` on the `lin_reg_pipe` variable and renamed the output file to `linear_regression_pipeline.pkl` to prevent deployment errors.

**Issue 4: Matplotlib Memory Leak in Dashboard Loops**
* **Bug:** While rendering multiple scatter plots within a `for` loop on the Sale Price Study page, the application triggered a `PyplotGlobalUseWarning`. This happened because Matplotlib was keeping all figure objects in the global state, leading to excessive memory consumption and eventual server instability.
* **Fix:** I implemented `plt.close(fig)` immediately after the `st.pyplot(fig)` call within the plotting loop.
* **Reasoning:** Unlike a standard Jupyter Notebook where plots are cleared after display, a Streamlit web server requires explicit resource cleanup. Closing the figure object manually ensures that memory is released back to the system after each plot is rendered, preventing leaks and maintaining high performance during user interaction.