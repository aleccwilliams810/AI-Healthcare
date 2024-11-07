############ Run script to process/clean raw data and begin feature engineering via encoding and embedding diagnosis codes

import sys

# Add the path to the directory containing the 'Self-Tutorial' folder
sys.path.append('/content/AI-Healthcare/Self-Tutorial')

from scripts.process_patients import process_patient_data
from scripts.process_admissions import process_admissions_data
from scripts.process_diagnoses import process_diagnosis_data

def encode_data():
    patient_csv_path = 'data/raw/patients.csv'
    admissions_csv_path = 'data/raw/admissions.csv'
    diagnoses_csv_path = 'data/raw/diagnoses.csv'

    process_patient_data(patient_csv_path)
    process_admissions_data(admissions_csv_path)
    process_diagnosis_data(diagnoses_csv_path)

def main():
    encode_data()

if __name__ == "__main__":
    main()
