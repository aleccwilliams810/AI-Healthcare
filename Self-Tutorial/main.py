from scripts.pull_data import authenticate_and_initialize_client, load_data
from scripts.process_patients import process_patient_data
from scripts.process_admissions import process_admissions_data
from scripts.process_diagnoses import process_diagnosis_data
from scripts.join_postprocess import join_data, postprocess_data

def pull():
    # Authenticate and initialize the BigQuery client
    client = authenticate_and_initialize_client()
    
    # Load data from BigQuery
    patients_df, admissions_df, diagnoses_df = load_data(client)
    
    # Check if the DataFrames are loaded successfully
    if patients_df is not None and admissions_df is not None and diagnoses_df is not None:
        print("Data successfully loaded into DataFrames.")
        
    else:
        print("Failed to load one or more DataFrames.")

def process_data():
    patient_csv_path = 'Self-Tutorial/data/raw/patients.csv'
    admissions_csv_path = 'Self-Tutorial/data/raw/admissions.csv'
    diagnoses_csv_path = 'Self-Tutorial/data/raw/diagnoses.csv'

    process_patient_data(patient_csv_path)
    process_admissions_data(admissions_csv_path)
    process_diagnosis_data(diagnoses_csv_path)

    df = join_data()
    final_inputs = postprocess_data(df)

    return final_inputs
    

def main():
    pull()
    final_inputs = process_data()

if __name__ == "__main__":
    main()
