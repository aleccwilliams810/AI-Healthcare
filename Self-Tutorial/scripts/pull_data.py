import os
from google.colab import auth
from google.cloud import bigquery
import pandas as pd

##### Adjust the code where indicated to match your credentials

## def authenticate_and_initialize_client(project_id="your_project_id"):
##     """Authenticate the user and initialize the BigQuery client."""
##     auth.authenticate_user()
##     client = bigquery.Client(project="your_project_id")
##     return client
def authenticate_and_initialize_client(project_id="careful-broker-438616-s1"):
    """Authenticate the user and initialize the BigQuery client."""
    auth.authenticate_user()
    client = bigquery.Client(project="careful-broker-438616-s1")
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

    # Save DataFrames as CSV files at specified path
    output_dir = "Self-Tutorial/data/raw"
    patients_df.to_csv(os.path.join(output_dir, "patients.csv"), index=False)
    print("Patients data saved to Self-Tutorial/data/raw/patients.csv")

    admissions_df.to_csv(os.path.join(output_dir, "admissions.csv"), index=False)
    print("Admissions data saved to Self-Tutorial/data/raw/admissions.csv")

    diagnoses_df.to_csv(os.path.join(output_dir, "diagnoses.csv"), index=False)
    print("Diagnoses data saved to Self-Tutorial/data/raw/diagnoses.csv")

    return patients_df, admissions_df, diagnoses_df