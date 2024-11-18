from scripts.pat_processing import process_patient_data
from scripts.adm_processing import process_admissions_data
from scripts.diag_processing import process_diagnosis_data

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
