# NYC Airbnb Price Prediction Pipeline

## Project Overview
This project builds and automates a full machine learning pipeline to predict Airbnb listing prices in New York City.  
The workflow follows the modular structure from Udacity’s **Build an ML Pipeline for Production** project, using MLflow to track experiments and manage model stages.

The pipeline performs data validation, cleaning, exploratory data analysis (EDA), model training, model registration, and testing.  
The model uses features from Airbnb listings such as location, room type, and availability to estimate nightly prices.

---

## Pipeline Steps

### 1. **data_check**
Validates the input dataset to ensure that:
- Row count is within a reasonable range.  
- Price values fall between configured `min_price` and `max_price` thresholds (10–350).  
- Data schema matches expected columns.  

Command:
```bash
mlflow run . -P steps=data_check --env-manager=local
