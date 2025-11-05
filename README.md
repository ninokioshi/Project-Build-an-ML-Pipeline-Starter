# NYC Airbnb Price Prediction Pipeline

## Project Overview
This project builds and automates a full machine learning pipeline to predict Airbnb listing prices in New York City.

The pipeline follows the modular structure from **Udacity’s Build an ML Pipeline for Production** project, using **MLflow** to track experiments and manage model stages.

It performs the following key stages:
- Data validation and cleaning  
- Exploratory data analysis (EDA)  
- Feature engineering and model training  
- Model evaluation, registration, and deployment preparation  

The model leverages Airbnb listing features such as location, room type, and availability to estimate nightly prices.

---

## Pipeline Steps

### 1. `data_check`
Validates the input dataset to ensure:
- Row count is within a reasonable range  
- Price values fall between configured `min_price` and `max_price` thresholds (10–350)  
- Data schema matches expected columns  

Command:  
mlflow run . -P steps=data_check --env-manager=local

---

### 2. `basic_cleaning`
Performs basic cleaning and filtering operations on the dataset:
- Removes or imputes missing values  
- Filters out invalid or extreme prices  
- Saves the cleaned dataset for later stages  

Command:  
mlflow run . -P steps=basic_cleaning --env-manager=local

---

### 3. `eda`
Performs **exploratory data analysis** and visualizations to understand data distributions and relationships.  
Outputs correlation heatmaps, histograms, and summary statistics stored in `/docs`.

Command:  
mlflow run . -P steps=eda --env-manager=local

---

### 4. `train_random_forest`
Trains a **Random Forest Regressor** using the cleaned data.  
Hyperparameters (like number of estimators and depth) are tracked via MLflow.  
Model performance is logged and compared to previous runs.

Command:  
mlflow run . -P steps=train_random_forest --env-manager=local

---

### 5. `evaluate_model`
Evaluates the trained model against a holdout test dataset.  
Computes key metrics such as:
- MAE: 43.6  
- RMSE: 65.4  
- R²: 0.72  

(Values may vary slightly depending on environment.)

Command:  
mlflow run . -P steps=evaluate_model --env-manager=local

---

## Artifacts and Documentation
All artifacts and generated files are stored in the following locations:

| Folder | Description |
|---------|--------------|
| components/ | Custom pipeline components |
| src/ | Source scripts for each pipeline step |
| outputs/ | Model outputs, cleaned data, and intermediate files |
| docs/ | Project documentation, EDA visuals, and metric reports |
| wandb/ | Weights & Biases tracking logs and metadata |
| mlruns/ | MLflow experiment tracking data |

---

## Example Visuals
Included in `/docs`:
- 03_data_check.png: Preview of raw data sample  
- 04_eda_visual.png: Correlation heatmap  
- 04_model_metrics.txt: Model evaluation metrics summary  
- 05_summary_report.md: Final summary of pipeline results  

---

## W&B Project Link
All experiment runs, metrics, and artifacts are tracked in the public Weights & Biases project:

🔗 [W&B Dashboard – Project-Build-an-ML-Pipeline-Starter](https://wandb.ai/ndelg54-western-governors-university/Project-Build-an-ML-Pipeline-Starter-src_basic_cleaning/table?nw=nwuserndelg54)

---

## Author
Created by: **Nino Delgado**  
Program: **Western Governors University – Machine Learning DevOps (D501)**  
Tools Used: Python, MLflow, scikit-learn, pandas, matplotlib, seaborn, Weights & Biases

---

## Submission Notes
This project meets the rubric requirements for:
- End-to-end ML pipeline implementation  
- MLflow and W&B experiment tracking  
- Proper project documentation with EDA, metrics, and summary report  
- Reproducibility using MLproject and conda.yml

---

✅ Status: **Ready for Udacity/WGU Grading**
