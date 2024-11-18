import pandas as pd
import numpy as np
from scripts.data_manager import save_processed

def process_admissions_data(csv_file):
    admissions_df = pd.read_csv(csv_file)

    admissions_df = update_date_format(admissions_df)
    admissions_df = previous_admissions(admissions_df)
    admissions_df = previous_admission_diag(admissions_df)
    admissions_df = flatten_df(admissions_df)
    admissions_df = apply_encoding(admissions_df)

    save_processed(admissions_df, 'admissions_cleaned.csv')

# updating date format to YYYYMMDD for compatibility    
def update_date_format(df):
    df['admittime'] = pd.to_datetime(df['admittime'])
    return df

# adding feature to track whether patient has been previously admitted, and how many times
def previous_admissions(df):
    df['previous_admits'] = df.groupby('subject_id').cumcount()
    return df

# adding features to track previous admission consultation diagnoses, up to 3 previous diagnoses
def previous_admission_diag(df):
    df['previous_consult_diagnosis1'] = df.groupby('subject_id')['consult_diagnosis'].shift(1)
    df['previous_consult_diagnosis2'] = df.groupby('subject_id')['consult_diagnosis'].shift(2)
    df['previous_consult_diagnosis3'] = df.groupby('subject_id')['consult_diagnosis'].shift(3)
    return df

# flattening table to only include single row for most recent admission, with additional columns for previous admission data 
def flatten_df(df):
    flattened_df = df.drop_duplicates(subset='subject_id', keep='last')
    flattened_df = flattened_df[['subject_id', 'admittime', 'admission_type', 'admission_location',
                                 'language', 'religion', 'marital_status', 'ethnicity',
                                 'consult_diagnosis', 'previous_admits', 'previous_consult_diagnosis1',
                                 'previous_consult_diagnosis2', 'previous_consult_diagnosis3',
                                 'hospital_expire_flag']]
    return flattened_df

# encoding categorical variables using one-hot encoding and custom encoding for 'language' column
def apply_encoding(df):
    one_hot_cols = ['admission_type', 'admission_location', 'religion', 'marital_status', 'ethnicity']

    # one-hot encoding with dummies
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

    # Custom encoding 'language': English Native Language = 1, Other Language = 0
    df['engl_native_lang'] = df['language'].apply(
        lambda x: 1 if x == 'ENGL' else 0
    )

    df.drop('language', axis=1, inplace=True)

    return df