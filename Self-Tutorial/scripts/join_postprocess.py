import pandas as pd
import os

import spacy
import numpy as np

from sklearn.cluster import KMeans

from scripts.pull_data import save_processed


patient_path = 'Self-Tutorial/data/processed/patients_cleaned.csv'
admission_path = 'Self-Tutorial/data/processed/admissions_cleaned.csv'
diagnoses_path = 'Self-Tutorial/data/processed/diagnoses_cleaned.csv'

def join_data():
    # Read the cleaned data
    patients_df = pd.read_csv(patient_path)
    admissions_df = pd.read_csv(admission_path)
    diagnoses_df = pd.read_csv(diagnoses_path)
    
    # Merge the patient and admission data
    patients_admissions_df = pd.merge(patients_df, admissions_df, on='subject_id', how='inner')
    
    # Merge the result with the diagnoses data
    joined_df = pd.merge(patients_admissions_df, diagnoses_df, on='subject_id', how='inner')
    
    return joined_df

def postprocess_data():
    df = join_data()
    df = calc_age(df)

    df = apply_embeddings_function(df)
    df = apply_kmeans(df)

    save_processed(df, 'model_inputs.csv')


def calc_age(df):
    #First calculated in years to isolate outliers to remove. (300+ yo)
    #Then calculated age in days to introduce more variance to the feature.

    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df['admittime'] = pd.to_datetime(df['admittime'], errors='coerce')

    df['age_yrs'] = df['admittime'].dt.year - df['dob'].dt.year

    df = df[df['age_yrs'] <= 120]

    df['age_days'] = (df['admittime'] - df['dob']).dt.days 

    df.drop(columns=['dob', 'admittime', 'age_yrs'], inplace=True)

    return df

# Encoding diagnosis descriptions using kmeans to group similarly vectorized descriptions together.
# Using pre-trained SciSpacy model to vectorize with Kmeans to cluster

def get_embeddings(text):
    nlp = spacy.load("en_core_sci_md")
    doc = nlp(text)

    return doc.vector

def apply_embeddings_function(df):
    # Gather all columns that contain 'consult_diag' or end with '_desc'
    string_columns = [col for col in df.columns if 'consult_diag' in col or col.endswith('_desc')]

    for col in string_columns:
        df[f'{col}_embedding'] = df[col].fillna('').apply(get_embeddings)
    
    df.drop(columns=string_columns, inplace=True)

    return df

def apply_kmeans(df):
    embedding_cols = [col for col in df.columns if '_embedding' in col]

    for col in embedding_cols:
        # Get the unique count of the original column
        unique_count = df[col.replace('_embedding', '')].nunique()

        # Determine cluster num
        n_clusters = determine_n_clusters(unique_count)
        
        # Prepare cluster mat
        embedding_matrix = np.vstack(df[col].values)
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embedding_matrix)
        
        # Add labels
        df[f'{col}_cluster'] = labels
        print(f'KMeans clustered {col} with {n_clusters} clusters.')

    return df

def determine_n_clusters(unique_count):
    # More clusters for high-variance columns
    if unique_count > 5000:
        return 150  
    elif unique_count > 2500:
        return 75
    elif unique_count > 1500:
        return 50  
    else:
        return 25