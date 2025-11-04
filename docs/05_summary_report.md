PROJECT: Build an ML Pipeline for Short-Term Rental Prices  
Author: Nino Delgado  
Date: November 2025  

This project builds a complete machine learning pipeline using MLflow for reproducible experimentation and modular workflow management. The dataset used was cleaned, validated, and modeled to predict short-term rental prices based on Airbnb listings in New York City.

The pipeline was executed successfully across the following stages:
1. Data Cleaning (`basic_cleaning`) – Removed null values, standardized columns, and produced a clean CSV artifact.  
2. Data Validation (`data_check`) – Verified integrity and schema consistency of the clean dataset.  
3. Exploratory Data Analysis (`eda`) – Generated visualizations including correlation heatmaps (`04_eda_visual.png`).  
4. Model Training (`train_random_forest`) – Trained and evaluated a Random Forest Regressor using MLflow tracking.  
5. Model Evaluation (`quick_eval.py`) – Produced validation metrics confirming model performance.

Final model metrics (from quick evaluation):
- Mean Absolute Error (MAE): 33.53  
- Root Mean Squared Error (RMSE): 47.10  
- R² Score: 0.57  

Evidence of each pipeline step and its results can be found in the `docs/` folder.  
All MLflow runs, artifacts, and configurations are included in this project directory.
