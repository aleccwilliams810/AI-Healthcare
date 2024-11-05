import pandas as pd
import os
from scripts.pull_data import save_processed

def process_admissions_data(csv_file):
    admissions_df = pd.read_csv(csv_file)

    admissions_df = update_date_format(admissions_df)
    admissions_df = previous_admissions(admissions_df)
    admissions_df = previous_admission_diag(admissions_df)
    admissions_df = flatten_df(admissions_df)
    admissions_df = apply_encoding(admissions_df)

    save_processed(admissions_df, 'admissions_cleaned.csv')
    
def update_date_format(df):
    df['admittime'] = pd.to_datetime(df['admittime'])
    return df

def previous_admissions(df):
    df['previous_admits'] = df.groupby('subject_id').cumcount()
    return df

def previous_admission_diag(df):
    df['previous_consult_diagnosis1'] = df.groupby('subject_id')['consult_diagnosis'].shift(1)
    df['previous_consult_diagnosis2'] = df.groupby('subject_id')['consult_diagnosis'].shift(2)
    df['previous_consult_diagnosis3'] = df.groupby('subject_id')['consult_diagnosis'].shift(3)
    return df

def flatten_df(df):
    flattened_df = df.drop_duplicates(subset='subject_id', keep='last')
    flattened_df = flattened_df[['subject_id', 'admittime', 'admission_type', 'admission_location',
                                 'language', 'religion', 'marital_status', 'ethnicity',
                                 'consult_diagnosis', 'previous_admits', 'previous_consult_diagnosis1',
                                 'previous_consult_diagnosis2', 'previous_consult_diagnosis3',
                                 'hospital_expire_flag']]
    return flattened_df

def apply_encoding(df):
    one_hot_cols = ['admission_type', 'admission_location', 'religion', 'marital_status', 'ethnicity']

    # Apply one-hot encoding using get_dummies
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

    # Custom encoding for the 'language' column: English Native Language = 1, Other Language = 0, 'None' = Null
    df['engl_native_lang'] = df['language'].apply(
        lambda x: 1 if x == 'ENGL' else (None if x == 'None' else 0)
    )

    df.drop('language', axis=1, inplace=True)

    return df