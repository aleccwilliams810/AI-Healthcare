import pandas as pd
import os

import spacy
import numpy as np

from sklearn.cluster import KMeans

from scripts.pull_data import save_processed

from tqdm import tqdm


###### Install scispacy model with !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz


patient_path = 'data/processed/patients_cleaned.csv'
admission_path = 'data/processed/admissions_cleaned.csv'
diagnoses_path = 'data/processed/diagnoses_cleaned.csv'

def join_data():

    patient_path = 'data/processed/patients_cleaned.csv'
    admission_path = 'data/processed/admissions_cleaned.csv'
    diagnoses_path = 'data/processed/diagnoses_cleaned.csv'

    # Read the cleaned data
    patients_df = pd.read_csv(patient_path)
    admissions_df = pd.read_csv(admission_path)
    diagnoses_df = pd.read_csv(diagnoses_path)
    
    # Merge the patient and admission data
    patients_admissions_df = pd.merge(patients_df, admissions_df, on='subject_id', how='inner')
    
    # Merge the result with the diagnoses data
    joined_df = pd.merge(patients_admissions_df, diagnoses_df, on='subject_id', how='inner')
    
    return joined_df

def postprocess_data(df):
    df = calc_age(df)

    df, embedding_df = apply_embeddings_function(df)
    df = apply_kmeans(df, embedding_df)

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

nlp = spacy.load("en_core_sci_md")

def get_embeddings(texts):
    #nlp.pipe enables batch processing to improve efficiency
    embeddings = []
    for doc_batch in nlp.pipe(texts, batch_size=1000, n_process=4, disable=["parser", "tagger"]):
        embeddings.append(doc_batch.vector)
    return embeddings

def apply_embeddings_function(df):
    # Gather all columns containing 'consult_diag' or ending with '_desc'
    string_columns = [col for col in df.columns if 'consult_diag' in col or col.endswith('_desc')]
    
    embedding_cols = {}

    for col in string_columns:
        print(f"Processing column: {col}")
        
        # tqdm shows progress
        embeddings = get_embeddings(tqdm(df[col].fillna('').tolist(), desc=f"Embedding {col}"))
        
        # Store embeddings with the new column name
        embedding_cols[f'{col}_embedding'] = embeddings
        print(f"Completed embeddings for column: {col}")

    # Create new DataFrame with embedding columns
    embedding_df = pd.DataFrame(embedding_cols, index=df.index)

    # Return the new  DataFrame and the original DataFrame with columns dropped
    return df.drop(columns=string_columns), embedding_df

def apply_kmeans(df, embedding_df):
    embedding_cols = [col for col in embedding_df.columns]

    cluster_labels = {}

    for col in embedding_cols:

        # Determine cluster num
        n_clusters = determine_n_clusters(embedding_df[col].nunique())
        
        # Prepare cluster matrix
        embedding_matrix = np.vstack(embedding_df[col].values)
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embedding_matrix)

        cluster_labels[f'{col}_cluster'] = labels
        
    # Add labels
    df = pd.concat([df, pd.DataFrame(cluster_labels, index=df.index)], axis=1)

    return df

def determine_n_clusters(unique_count):
    # More clusters for high-variance columns
    if unique_count > 5000:
        return 40  
    elif unique_count > 2500:
        return 30
    elif unique_count > 1500:
        return 20  
    else:
        return 10