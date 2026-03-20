import streamlit as st
from app_pages.multi_page import MultiPage

# 1. Import page functions 
from app_pages.page_1_project_summary import page_1_project_summary_body
from app_pages.page_2_sale_price_study import page_2_sale_price_study_body
from app_pages.page_3_price_predictor import page_3_inherited_houses_body
from app_pages.page_4_project_hypotheses import page_4_project_hypotheses_body
from app_pages.page_5_ml_performance import page_5_ml_performance_body

# Instantiate the app
app = MultiPage(app_name="Heritage Housing Analytics")

# 2. Add pages to the dashboard menu
# The motor will automatically create a button for each line below
app.app_page("1. Quick Project Summary", page_1_project_summary_body)
app.app_page("2. House Sale Price Study", page_2_sale_price_study_body)
app.app_page("3. Inherited Houses Price Prediction", page_3_inherited_houses_body)
app.app_page("4. Project Hypotheses", page_4_project_hypotheses_body)
app.app_page("5. ML Performance Metrics", page_5_ml_performance_body)

# Run the app
app.run()