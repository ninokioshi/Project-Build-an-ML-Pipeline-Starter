# NYC Airbnb Price Prediction Pipeline

## Project Overview
This project builds and automates a complete machine learning pipeline to predict Airbnb listing prices in New York City.

The pipeline follows the modular structure from **Udacity’s Build an ML Pipeline for Production** project, using **MLflow** for experiment tracking and **Weights & Biases (W&B)** for artifact management and versioning.

It performs the following key stages:
- Data download and validation  
- Data cleaning (including NYC boundary filtering)  
- Data splitting into training, validation, and test sets  
- Model training, evaluation, and versioning  

The model estimates nightly prices using features such as location, room type, and availability.

---

## Pipeline Steps

### 1. `download`
Downloads and logs the raw Airbnb dataset as a W&B artifact.

**Command:**
```bash
mlflow run . -P steps=download --env-manager=local
