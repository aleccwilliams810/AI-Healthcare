import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm

from sklearn.cluster import KMeans
from scripts.pull_data import save_processed, remove_temporary_files


###### Install scispacy model with !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz
nlp = spacy.load("en_core_sci_md")

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

def build_embeddings(df):
    df = calc_age(df)

    embedding_df, string_columns = apply_embeddings_function(df)

    save_processed(df, 'temp_df.csv')
    embedding_df.to_parquet('data/processed/temp_embedding_df.parquet')
    save_processed(string_columns, 'temp_string_columns.csv')

def kmeans_cluster_embeddings():
    df = pd.read_csv('data/processed/temp_df.csv')
    embedding_df = pd.read_parquet('data/processed/temp_embedding_df.parquet')
    string_columns = pd.read_csv('data/processed/temp_string_columns.csv')

    df = apply_kmeans(df, embedding_df, string_columns)

    save_processed(df, 'model_inputs.csv')

def calc_age(df):
    #First calculated in years to isolate outliers to remove. (300+ yo) 
    # Need to remove outliers first or dt.dats will not work.
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

    for col in tqdm(string_columns, desc=f"Embedding columns..."):
        print(f"\nProcessing column: {col}")
        
        # tqdm shows progress
        embeddings = get_embeddings(df[col].fillna('').tolist())
        
        # Store embeddings with the new column name
        embedding_cols[f'{col}_embedding'] = embeddings
        print(f"Completed embeddings for column: {col}")

    # Create new DataFrame with embedding columns
    embedding_df = pd.DataFrame(embedding_cols, index=df.index)

    # Return the new  DataFrame and the original DataFrame with columns dropped
    return embedding_df, df[string_columns]

def apply_kmeans(df, embedding_df, original_string_df):
    embedding_cols = embedding_df.columns

    unique_counts = {col: original_string_df[col].nunique() for col in original_string_df.columns}

    cluster_labels = {}

    for i, col in tqdm(enumerate(embedding_cols), desc=f"Clustering embeddings..."):

        # Determine cluster num using original column pre-emeddings
        original_col = original_string_df.columns[i]
        n_clusters = determine_n_clusters(unique_counts[original_col])
        
        # Prepare cluster matrix
        embedding_matrix = np.vstack(embedding_df[col].values)
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        cluster_labels[f'{col}_cluster'] = kmeans.fit_predict(embedding_matrix)
        
    # Add labels
    df = pd.concat([df, pd.DataFrame(cluster_labels, index=df.index)], axis=1)

    return df.drop(columns=original_string_df.columns)

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