# Patient Mortality Prediction Tutorial

## Project Overview
- Predicts patient mortality using MIMIC-III data.
- Uses an LGBM model for rapid and accurate mortality risk scoring.
- Focuses on leveraging basic admission, demographic, and diagnosis data for prediction.
- **Goal**: Provide quick and reliable risk assessments that assist in real-time medical decisions.

## Purpose and Goals
- **Initial Model Focus:** Accurate mortality prediction with minimal data input to aid in early intervention.
- **Potential Expansion:** The model can be part of a predictive series, improving as patient data accumulates throughout the hospital journey.
- **Real-Time Application:** LGBM's speed ensures that predictions can adapt dynamically during critical events like surgery.

## Project Structure
- `data/`: raw and processed data files - scrubbed for Git publish
- `notebooks/`: notebooks for data exploration and training
- `scripts/`: scripts for the main file
- `models/`: model files and reports - scrubbed for Git publish
- main.py
- `README.md`
- `requirements.txt`: dependencies

## Data Usage Note
- MIMIC-III data is not included due to data use agreements.
- Users need approved PhysioNet access to load and use MIMIC-III data.
**Tip: For runtime issues, use MIMIC-III demo data or sample the dataset before full processing.**

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
- csv
- google colab
- google cloud


