import pandas as pd
import os
from scripts.pull_data import save_processed

def process_patient_data(csv_file):
    patients_df = pd.read_csv(csv_file)

    # Binary encoding for gender, M=1 F=0
    patients_df['gender'] = patients_df['gender'].apply(lambda x: 1 if x == 'M' else 0)

    save_processed(patients_df, 'patients_cleaned.csv')

