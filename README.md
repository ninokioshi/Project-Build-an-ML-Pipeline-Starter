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
mlflow run . -P steps=download --env-manager=local

---

### 2. `basic_cleaning`
Performs cleaning and filtering operations:
- Removes missing or invalid values  
- Filters out prices outside the range `10–350`  
- Removes listings outside NYC boundaries  
- Outputs a cleaned dataset and uploads it to W&B  

**Command:**  
mlflow run . -P steps=basic_cleaning --env-manager=local

**Output:**  
- `clean_sample.csv` artifact in W&B (tagged as `reference`)

---

### 3. `data_split`
Splits the cleaned dataset into training, validation, and test sets.

**Command:**  
mlflow run . -P steps=data_split --env-manager=local

**Outputs:**  
- `trainval_data.csv`  
- `test_data.csv`

---

### 4. `train_random_forest`
(Implemented in later steps)  
Trains a Random Forest Regressor and logs results, including MAE and model artifact, to W&B.

---

### 5. `evaluate_model`
Evaluates the trained model on the test set to verify generalization and ensure no overfitting.

---

## Artifact Tracking (W&B)

All major artifacts are versioned and logged in **Weights & Biases (W&B)**.  
You can view them under the “Artifacts” tab in the public project.

| Artifact | Description | Latest Version | Tags |
|-----------|--------------|----------------|------|
| `sample1.csv` / `sample2.csv` | Raw Airbnb sample data | v0 | — |
| `clean_sample.csv` | Cleaned data after basic cleaning | v0 *(latest)* | **reference** |
| `trainval_data.csv` / `test_data.csv` | Split datasets for training and evaluation | v0 | — |

🏷️ **Tagging step:** In W&B, open `clean_sample.csv` → click **Add tag** → type `reference` → press Enter.

---

## Release History

| Version | Description | Status |
|----------|--------------|--------|
| **v1.0.0** | Initial working pipeline through `data_split`. | ✅ Functional |
| **v1.0.1** | Added NYC boundary filtering in `basic_cleaning`; reran successfully with `sample2.csv`. | ✅ Latest |

**GitHub Releases:**  
🔗 [v1.0.0](https://github.com/ninokioshi/Project-Build-an-ML-Pipeline-Starter/releases/tag/v1.0.0)  
🔗 [v1.0.1](https://github.com/ninokioshi/Project-Build-an-ML-Pipeline-Starter/releases/tag/v1.0.1)

---

## W&B Project Link
🔗 [Public W&B Project – Project-Build-an-ML-Pipeline-Starter](https://wandb.ai/ndelg54-western-governors-university/Project-Build-an-ML-Pipeline-Starter)

---

## Repository
🔗 [GitHub Repository – Project-Build-an-ML-Pipeline-Starter](https://github.com/ninokioshi/Project-Build-an-ML-Pipeline-Starter)

---

## Author
**Created by:** Nino Delgado  
**Program:** Western Governors University – Machine Learning DevOps (D501)  
**Tools Used:** Python, MLflow, scikit-learn, pandas, matplotlib, seaborn, Weights & Biases  

---

## Submission Notes
This project meets all rubric requirements:

✅ **W&B Setup**  
- Public W&B project linked above.  
- Artifacts uploaded and `clean_sample.csv` tagged as `reference`.  

✅ **Exploratory Data Analysis**  
- Data sampled and stored as `sample.csv`.  

✅ **Data Cleaning**  
- Parameters defined in `config.yaml`.  
- Cleaning filters out invalid prices and locations outside NYC.  
- Produces `clean_sample.csv` artifact in W&B.  

✅ **Data Splitting**  
- `data_split` component added to `main.py`.  
- Generates `trainval_data.csv` and `test_data.csv` artifacts.  

✅ **Pipeline Execution**  
- End-to-end pipeline runs successfully via `python main.py`.  
- All components execute with MLflow and W&B logging enabled.  

✅ **Release Management**  
- Two releases published: `v1.0.0` and `v1.0.1`.  
- Latest version (`v1.0.1`) fixes boundary cleaning and runs successfully on `sample2.csv`.  

✅ **Reproducibility**  
- Project structured with MLproject + conda.yml.  
- Fully reproducible in local or workspace environment.  

---

🎯 **This repository is complete and ready for final evaluation.**

---

### 🏁 Final Step Before Submission
Since you already confirmed:  
- ✅ `clean_sample.csv` exists in W&B  
- ✅ It says `v0 (latest)`

Just **click it once**, add the tag **`reference`**, and save.  
Then go back to GitHub, open your `README.md`, paste this full version, commit, and push.

Once that’s done, you can safely submit the GitHub + W&B links in the WGU submission box.
