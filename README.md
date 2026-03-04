# Heritage Housing Issues - Ames, Iowa

## Business Requirements
The client is interested in discovering how house attributes correlate with the sale price.
1. The client expects data visualizations of the correlated variables against the sale price.
2. The client is interested in predicting the house sale price for her four inherited houses and any other house in Ames, Iowa.

## Project Hypothesis and Validation
* We suspect that the overall size of the house (Gross Living Area) and its overall quality have the strongest positive correlation with the sale price.
* **Validation:** We will perform a correlation study and visualize relationships using Seaborn (pairplots and scatterplots).

## ML Business Case
* **Service:** Regression model to predict house sale prices.
* **Success Metric:** R2 score of 0.75 or higher on both train and test sets.
* **Output:** Continuous numerical value (USD) for the predicted price.