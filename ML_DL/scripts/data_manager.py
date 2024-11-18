import os
from google.colab import auth
from google.cloud import bigquery

##### Adjust the code where indicated to match your credentials

def authenticate_and_initialize_client(project_id="your_project_id"): ####### Adjust your_project_id
    """Authenticate the user and initialize the BigQuery client."""
    auth.authenticate_user()
    client = bigquery.Client(project="your_project_id") ####### Adjust your_project_id
    return client

def load_data(client):
    patients_table = """
    SELECT 
    subject_id,
    gender,
    CAST(dob AS DATE) AS dob
    FROM `physionet-data.mimiciii_clinical.patients` pat
    """

    admissions_table = """
    SELECT 
    subject_id,
    CAST(admittime AS DATE) AS admittime,
    admission_type,
    admission_location,
    language,
    religion,
    marital_status,
    ethnicity,
    diagnosis AS consult_diagnosis,
    hospital_expire_flag
    FROM `physionet-data.mimiciii_clinical.admissions` adm
    """

    diag_table = """
    SELECT 
        subject_id,
        seq_num,
        diag.icd9_code AS diag_code,
        long_title AS diag_code_desc
    FROM `physionet-data.mimiciii_clinical.diagnoses_icd` diag
    INNER JOIN `physionet-data.mimiciii_clinical.d_icd_diagnoses` diagdesc
        ON diag.icd9_code = diagdesc.icd9_code
    """

    try:
        patients_df = client.query(patients_table).result().to_dataframe()
        print("Patients data loaded successfully.")

        admissions_df = client.query(admissions_table).result().to_dataframe()
        print("Admissions data loaded successfully.")

        diagnoses_df = client.query(diag_table).result().to_dataframe()
        print("Diagnoses data loaded successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

    save_raw(patients_df, 'patients.csv')

    save_raw(admissions_df, 'admissions.csv')

    save_raw(diagnoses_df, 'diagnoses.csv')

    return patients_df, admissions_df, diagnoses_df

# Functions to ensure directories exist and save data
def save_raw(df, filename):
    raw_path = 'data/raw'
    os.makedirs(raw_path, exist_ok=True)
    df.to_csv(os.path.join(raw_path, filename), index=False)
    print(f"Saved to {os.path.join(raw_path, filename)}")

# Function to ensure directory exists and save to data/processed
def save_processed(df, filename):
    processed_path = 'data/processed'
    os.makedirs(processed_path, exist_ok=True)
    df.to_csv(os.path.join(processed_path, filename), index=False)
    print(f"Saved to {os.path.join(processed_path, filename)}")

def remove_temporary_files(*file_paths):
    for file_path in file_paths:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted temporary file: {file_path}")
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")