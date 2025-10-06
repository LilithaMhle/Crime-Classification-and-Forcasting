# Crime Forecasting and Classification in South Africa

# Overview
This project focuses on analyzing South African crime data and economic indicators to classify crime hotspots and forecast future crime trends. Using machine learning models such as Random Forest for classification and ARIMA for time series forecasting, the notebook demonstrates a complete workflow: from data collection, cleaning, and exploratory analysis, to feature engineering, modeling, evaluation, and drone simulation for hotspot monitoring.

# Datasets
The analysis uses two datasets from Kaggle:
1. South African Reserve Bank Dataset – https://www.kaggle.com/datasets/lylebegbie/south-african-reserve-bank-dataset?utm_source=chatgpt.com
   - Contains time-series economic indicators like interest rates and inflation.
2. Crime Statistics for South Africa – https://www.kaggle.com/datasets/slwessels/crime-statistics-for-south-africa?utm_source=chatgpt.com
   - Includes crime cases by type, province, and year across South Africa.

# Deliverables
- Crime Statistics dataset (CSV)
- South African Reserve Bank dataset (CSV)
- Jupyter Notebook with full analysis, feature engineering, modeling, and visualization
- Streamlit dashboard application for interactive exploration

# Streamlit Tabs
The Streamlit app contains the following tabs:
1. Home – Overview of datasets and project
2. EDA – Exploratory Data Analysis visualizations
3. Classification – Random Forest crime type classification results with accuracy and confusion matrix
4. Forecasting – ARIMA crime trend forecasting with test MSE and visualizations
5. Summaries – Project summary and technical/non-technical explanation
6. Settings – Theme mode selection (Light/Dark)

