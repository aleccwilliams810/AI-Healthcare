# Patient Mortality Prediction Tutorial

## Project Overview
- Predicts patient mortality using MIMIC-III data.
- Uses an LGBM model for rapid and accurate mortality risk scoring.
- Focuses on leveraging basic admission, demographic, and diagnosis data for prediction.

## Purpose and Goals
- **Goal**: Build a model that uses introductory patient information to quickly predict mortality, aiding in the allocation of medical resources based on condition severity and urgency.
- **Impact**: Demonstrates the potential for a series of predictive models to be applied at various stages of a patient’s journey, becoming more accurate as more data becomes available. This initial model emphasizes making the most accurate predictions possible with minimal data.
- **Real-Time Application**: LGBM's speed allows for real-time usage, such as updating mortality risk predictions during surgery as patient vitals and other data change dynamically.

## Project Structure
- `data/`: raw and processed data files
- `notebooks/`: notebooks for data exploration and training
- `scripts/`: scripts for repeatable tasks
- `models/`: model files and reports
- `README.md`
- `requirements.txt`: dependencies

## Important Note on Data
- This repository does not include MIMIC-III data due to data use restrictions.
- Users must have approved access to MIMIC-III data through PhysioNet and load it into the project themselves.
- Follow the included code for data preprocessing and preparation after obtaining the dataset and saving it to 'Tutorial\data\raw'

## How to Run the Project
**Clone Repo**
    - git clone https://github.com/username/project-name.git
    - cd project-name/Tutorial

**Venv Setup**
   - python -m venv venv
   - .\venv\Scripts\activate 

**Install Dependencies**
   pip install -r requirements.txt

**Explore Notebooks, Scripts, and Models as Needed**

## Key Features
**LGBM Model**: Fast and robust gradient boosting model
**Feature Engineering**: Diagnosis code parsing, making the most out of minimal available data
**Evaluation**: AUC-ROC, precision, recall, etc.

## Dependencies
- numpy
- pandas
- matplotlib
- scikit-learn
- lightgbm

