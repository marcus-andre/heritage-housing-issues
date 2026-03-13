import streamlit as st
from app_pages.multi_page import MultiPage

# Import page functions
from app_pages.page_1_project_summary import page_1_project_summary_body
from app_pages.page_2_sale_price_study import page_2_sale_price_study_body
from app_pages.page_3_price_predictor import page_3_price_predictor_body

# Instantiate the app
app = MultiPage(app_name="Heritage Housing Analytics")

# Add pages to the dashboard menu using the official Code Institute method
app.app_page("1. Quick Project Summary", page_1_project_summary_body)
app.app_page("2. House Sale Price Study", page_2_sale_price_study_body)
app.app_page("3. Price Predictor", page_3_price_predictor_body)

# Run the app
app.run()