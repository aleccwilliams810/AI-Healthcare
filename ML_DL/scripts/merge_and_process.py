import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm

from sklearn.cluster import KMeans
from scripts.pull_data import save_processed


###### Install scispacy model with !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz
nlp = spacy.load("en_core_sci_md")


###### Data Loading and Merging
def join_data():
    patient_path = 'data/processed/patients_cleaned.csv'
    admission_path = 'data/processed/admissions_cleaned.csv'
    diagnoses_path = 'data/processed/diagnoses_cleaned.csv'
    
    # Read CSV files and merge
    patients_df = pd.read_csv(patient_path)
    admissions_df = pd.read_csv(admission_path)
    diagnoses_df = pd.read_csv(diagnoses_path)
    
    merged_df = patients_df.merge(admissions_df, on='subject_id', how='inner')
    merged_df = merged_df.merge(diagnoses_df, on='subject_id', how='inner')
    
    return merged_df


###### Additional Feature Engineering Methods

def calc_age(df):
    #Calculates patient age in days and removes extreme outliers. Outliers must be removed before applying '.dt.days'
    #Drops unnecessary columns after processing.

    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df['admittime'] = pd.to_datetime(df['admittime'], errors='coerce')

    df['age_yrs'] = df['admittime'].dt.year - df['dob'].dt.year

    df = df[df['age_yrs'] <= 120]

    df['age_days'] = (df['admittime'] - df['dob']).dt.days 

    df.drop(columns=['dob', 'admittime', 'age_yrs'], inplace=True)

    return df





