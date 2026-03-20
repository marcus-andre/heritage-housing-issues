# Heritage Housing Issues - Predictive Analytics

## Dataset Content
* The dataset is sourced from **Kaggle**. We created a fictitious user story where predictive analytics can be applied in a real-world workplace scenario.
* The dataset contains nearly 1500 rows and represents housing records from Ames, Iowa. It includes various property features (such as Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and their respective sale prices for houses built between 1872 and 2010.

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
* **BR2 (Predictive Modeling):** We will develop a Supervised Machine Learning model. Specifically, a Regression task to predict a continuous target (`SalePrice`). We will perform extensive Hyperparameter Optimization using `GridSearchCV`—testing a robust grid of multiple hyperparameters (e.g., at least 6 different parameters with 3 distinct values each for the baseline model algorithms)—to ensure maximum predictive accuracy.

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

## 5. Dashboard Design (Streamlit App UI)
Following the core dashboard design principles, the application was crafted to ensure each page has a clear **Use Case** and **Course of Action**, explicitly answering the business requirements. To guarantee a professional User Experience (UX), the dashboard is logically split to accommodate two distinct types of users: **non-technical** (business stakeholders) and **technical** (data practitioners).

### Non-Technical Pages (Business Stakeholders)
The first four pages are tailored for the client and business executives to easily navigate the project's insights and utilize the predictive tools:
* **Page 1: Quick Project Summary:** Describes the project dataset, the client's requirements, and general information about the Ames housing market.
* **Page 2: House Sale Price Study:** Addresses BR1. It contains dynamic data visualizations (histograms, scatter plots, correlation heatmaps) and a checkbox to display the raw data.
* **Page 3: Price Predictor:** Addresses BR2. It displays the predicted price for the 4 inherited houses and contains an interactive set of widgets for the user to input custom property features and receive a real-time price prediction.
* **Page 4: Project Hypotheses:** Displays the 3 business hypotheses, how we validated them, and the conclusions derived from the EDA.

### Technical Pages (Data Practitioners)
The final page is tailored for data practitioners who are interested in learning about the model's architecture and performance:
* **Page 5: ML Performance Metrics:** A technical page detailing the ML pipeline, feature importance, and model evaluation metrics ($R^2$ and Actual vs. Predicted plots) related to the Regression task.

## 6. Technologies and Libraries Used
The following tools and libraries were utilized within the virtual environment to develop this project:

* **numpy (1.26.1):** Used for efficient multi-dimensional array operations and mathematical calculations (e.g., calculating logarithmic inversions using `np.expm1`).
* **pandas (2.1.1):** Essential for data manipulation, aggregation, and DataFrame creation during the Data Cleaning and Feature Engineering phases.
* **matplotlib (3.8.0) & seaborn (0.13.2):** Used for static data visualization, creating correlation heatmaps, scatter plots, and distribution graphs during EDA. In the Streamlit dashboard, explicit figure closure was implemented to ensure optimal server memory management.
* **plotly (5.17.0):** Used to build interactive charts and plots for the Streamlit dashboard.
* **streamlit (1.40.2):** The core framework used to develop and deploy the interactive web dashboard interface.
* **kaggle (1.5.16):** Used to securely authenticate and download the Ames Housing dataset directly from the Kaggle API.
* **scikit-learn (1.3.2):** The primary machine learning library used for building pipelines, feature scaling (`StandardScaler`), model training (`LinearRegression`, `RandomForestRegressor`), hyperparameter optimization (`GridSearchCV`), and calculating performance metrics ($R^2$, MAE).

### 6.1 Architectural Decision: Static vs Interactive Visualizations
A deliberate architectural decision was made to mix static (`matplotlib`/`seaborn`) and dynamic (`plotly`) plotting libraries within the Streamlit dashboard to balance visual clarity, user engagement, and server performance:
* **Plotly (Dynamic/Interactive):** Used for Scatter Plots, Histograms, and Box Plots on the Sale Price Study page. Plotly provides an interactive HTML/JS frontend, allowing the user to hover over specific data points, pan, and zoom. This strictly satisfies the requirement for "interactive visualisations" to maximize user engagement, while safely offloading graphic memory management to the client's web browser.
* **Matplotlib & Seaborn (Static):** Used for complex statistical visualizations like the Correlation Heatmaps and Bar Plots. While Seaborn excels at rendering detailed statistical matrices, Matplotlib inherently retains figure objects in the Python global state. This creates a known risk of server memory leaks when rendering repeatedly in a Streamlit environment. To mitigate this and ensure system stability, explicit resource cleanup (`plt.close(fig)`) was rigorously implemented immediately after rendering every static plot.

### 6.2 Architectural Decision: Pre-rendered ML Performance Plots
During the transition from the Jupyter Notebook environment to the live Streamlit production dashboard, a deliberate MLOps decision was made to export the model's performance plots (Feature Importance and Regression Evaluation scatter plots) as static images (`.png`) rather than rendering them dynamically on the fly.
* **Performance & Server Memory Optimization:** Rendering complex scatter plots with thousands of data points dynamically would require the Streamlit server to load the entire dataset, perform train/test splits, reverse logarithmic transformations, and run predictions upon every page load. This consumes significant RAM and processing power, creating a high risk of application crashes (Out of Memory errors) in lightweight or free-tier deployment environments.
* **Reproducibility & Consistency:** By saving the visual artifacts at the exact moment of model training in the notebook, we guarantee that the dashboard displays a consistent, immutable snapshot of the model's approved performance. This follows industry MLOps best practices by strictly separating the **Research/Training** environment from the **Production/Deployment** environment.

### 6.3 Architectural Decision: Streamlit Data and Asset Caching
Streamlit's execution model runs the entire script from top to bottom upon every user interaction (e.g., clicking a button or changing a widget). 
* **The Risk:** Reading large CSV datasets and heavy serialized Machine Learning objects (`.pkl` pipelines and scalers) from the disk on every interaction would cause severe UI lag and excessive memory consumption, leading to potential server crashes.
* **The Solution:** We explicitly implemented Streamlit's caching decorators within the `src/data_management.py` module. We applied `@st.cache_data` to functions loading Pandas DataFrames and `@st.cache_resource` to functions loading non-mutating global ML models via `joblib`. This ensures these assets are loaded from the disk into RAM exactly once. Subsequent user interactions retrieve the data instantly from memory, achieving a highly responsive and stable production dashboard.

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

**Issue 4: Matplotlib Memory Leak in Dashboard**
* **Bug:** While rendering Matplotlib and Seaborn plots (such as the Correlation Bar Plot and Heatmaps) on the Sale Price Study page, the application is prone to trigger a `PyplotGlobalUseWarning`. This happens because Matplotlib keeps figure objects in the global state, leading to excessive memory consumption and eventual server instability upon multiple user interactions.
* **Fix:** I implemented `plt.close(fig)` immediately after every `st.pyplot(fig)` call.
* **Reasoning:** Unlike a standard Jupyter Notebook where plots are cleared after display, a Streamlit web server requires explicit resource cleanup. Closing the figure object manually ensures that memory is released back to the system after each plot is rendered, preventing leaks and maintaining high performance during user interaction.

**Issue 5: Scikit-Learn Feature Order Mismatch on Custom Prediction**
* **Bug:** When testing the "Predict Custom Property Price" tool, the ML pipeline crashed with a `ValueError` stating that the feature names must be in the exact same order as they were during the model's `fit()` phase. The dictionary creating the live Pandas DataFrame had slightly disordered keys, and Pandas' `.filter()` method does not reorder columns, it only drops unselected ones.
* **Fix:** I reorganized the dictionary keys within the `DrawInputsWidgets()` function to strictly reflect the exact column sequence of the original dataset. Furthermore, to make the pipeline bulletproof, I replaced the `.filter(price_features)` call with explicit list indexing `df[price_features]`, which guarantees that the DataFrame columns are forcefully reordered to match the trained pipeline's expectations before prediction.

## 8. Development Workflow (CRISP-DM)
This project was developed following the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology:
1. **Business Understanding:** Defining the client's needs (BR1 and BR2).
2. **Data Understanding:** Performing initial EDA to discover patterns and formulate hypotheses.
3. **Data Preparation:** Cleaning missing values, engineering new features (e.g., `HouseAge`), and encoding categorical variables.
4. **Modeling:** Training various algorithms (Decision Trees, Random Forests, Linear Regression) and hyperparameter tuning.
5. **Evaluation:** Assessing models against the success metric ($R^2 \ge 0.75$) and selecting the best performer.
6. **Deployment:** Building an interactive Streamlit dashboard and deploying it via Heroku.

## 9. Deployment

### Heroku Deployment
The project was deployed to Heroku using the following steps:

1. Log in to your Heroku account and create a new App.
2. Navigate to the **Deploy** tab, select **GitHub** as the deployment method, and connect your authorization.
3. Search for the project repository (`heritage-housing-issues`) and click **Connect**.
4. Before deploying, ensure the following core configuration files are present in the repository root:
   * `setup.sh`: Contains the Streamlit server configuration requirements.
   * `Procfile`: Tells Heroku how to run the application (`web: sh setup.sh && streamlit run app.py`).
   * `runtime.txt`: Explicitly sets the Python environment (e.g., `python-3.8.18`), which significantly reduces environment conflicts between development and production.
   * `requirements.txt`: Contains all the external packages and dependencies required to run the application.
   * `.slugignore`: Used to ignore large or sensitive files/folders (like the `jupyter_notebooks/` directory) that the application doesn't need once deployed, maintaining a lightweight server.
5. Scroll to the bottom of the deploy page, select the `main` branch, and click **Deploy Branch**.
6. Keep an eye on the build log to ensure there are no errors.
7. Once the build process runs smoothly and completes, click **Open App** to view the live dashboard.

**Live App Link:** [INSERIR O LINK DO SEU APP AQUI]

### Forking the GitHub Repository
To experiment with the code without affecting the main branch:
1. Log in to GitHub and locate the repository.
2. At the top right of the Repository page, click the **Fork** button.
3. A copy of the original repository will be created in your GitHub account.

### Local Clone
To clone the repository to your local machine:
1. Under the repository name on GitHub, click the **Code** button.
2. Choose your preferred cloning method (HTTPS, SSH, or GitHub CLI) and copy the provided URL.
3. Open your terminal or command prompt.
4. Change the current working directory to the location where you want the cloned directory to be made.
5. Type `git clone`, followed by the URL you copied in Step 2.
   ```bash
   git clone https://github.com/marcus-andre/heritage-housing-issues.git
   ```
6. Press Enter. Your local clone will be created.
7. Install the required dependencies using: `pip install -r requirements.txt`.

## 10. Credits
* The Ames Housing dataset was sourced from Kaggle.
* The project template and Streamlit multipage architecture were provided by Code Institute.
* Guidance on mitigating Streamlit memory leaks with Matplotlib was sourced from standard Python/Streamlit community best practices.