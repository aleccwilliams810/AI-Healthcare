############ Run script to process/clean raw data and begin feature engineering

from scripts.process_patients import process_patient_data
from scripts.process_admissions import process_admissions_data
from scripts.process_diagnoses import process_diagnosis_data
from scripts.join_postprocess import join_data, postprocess_data

def process_data():
    patient_csv_path = 'data/raw/patients.csv'
    admissions_csv_path = 'data/raw/admissions.csv'
    diagnoses_csv_path = 'data/raw/diagnoses.csv'

    process_patient_data(patient_csv_path)
    process_admissions_data(admissions_csv_path)
    process_diagnosis_data(diagnoses_csv_path)

    df = join_data()
    postprocess_data(df)

def main():
    process_data()

if __name__ == "__main__":
    main()
