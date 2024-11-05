import pandas as pd
import numpy as np
from scripts.pull_data import save_processed

def process_patient_data(csv_file):
    patients_df = pd.read_csv(csv_file)

    #Convert to NaN for better compatibility with LGBM
    patients_df = patients_df.replace({None: np.nan})

    # Binary encoding for gender, M=1 F=0
    patients_df['gender'] = patients_df['gender'].apply(lambda x: 1 if x == 'M' else 0)

    save_processed(patients_df, 'patients_cleaned.csv')

