# README: UCR Prediction Project

## Project Overview
This project focuses on predicting Unplanned Catheter Removals (UCR) using data from the MIMIC-III database. UCR is a rare clinical event, posing significant challenges due to class imbalance and limited data availability. The project leverages advanced feature engineering, data augmentation, and tree-based machine learning models to improve prediction accuracy and provide actionable insights for healthcare applications.

---

## Workflow Overview

1. **Data Pull (Google Colab)**:
   - Use `data_pull.ipynb` to query and extract relevant data from the MIMIC-III database.
   - Save the extracted data as CSV files.
   - Move the CSV files to the local project directory for further processing.

2. **Data Cleaning and Preprocessing (Local)**:
   - Run `data_processing.ipynb` locally to clean, preprocess, and engineer features.
   - Save the cleaned dataset as Parquet files for efficient storage and loading.

3. **Model Training and Tuning (Google Colab)**:
   - Use `model_training.ipynb` to train and tune the machine learning model.
   - Load the cleaned Parquet files into the Colab environment.
   - Experiment with hyperparameters and evaluate performance using metrics like precision-recall and average precision.

---

## Key Notes
- **Google Colab**: Use for data querying (pulling) and model training. Ensure CSV outputs from Colab are moved locally for cleaning.
- **Local Environment**: Use for data cleaning and feature engineering. Save the final dataset as Parquet files for efficient handling in Colab.
- **Parquet Files**: Reduce storage space and improve read/write speed when loading data for modeling in Colab.

---

## Requirements
- **Google Colab**: For querying and model development.
- **Python 3.x**: For local preprocessing (`data_processing.ipynb`).
- **Required Libraries**:
  - pandas
  - numpy
  - scikit-learn
  - LightGBM
  - BioBERT
  - check remaining requirements on requirements.txt

---

## Instructions
1. Run `data_pull.ipynb` in Colab to extract data from MIMIC-III.
2. Save the resulting CSV files in the `data/` directory locally.
3. Execute blocks on `data_processing.ipynb` locally to preprocess data and save Parquet files in `data/`.
4. Upload the cleaned Parquet files to Colab and use `model_dev.ipynb` to train and evaluate models. 

