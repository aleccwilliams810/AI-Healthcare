import pandas as pd
import numpy as np
from tqdm import tqdm
from scripts.data_manager import save_processed

from sklearn.cluster import KMeans


###### Clustering Embeddings
def determine_n_clusters(unique_count):
    # More clusters for high-variance columns
    if unique_count > 5000:
        return 100  
    elif unique_count > 2500:
        return 50
    elif unique_count > 1500:
        return 30
    else:
        return 20
    
def apply_kmeans(df, embedding_df, original_string_df):
    #Clusters diagnostic description embeddings using KMeans and appends cluster labels to the DataFrame.
    #Each column clustered independently of all other data

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
        kmeans = KMeans(n_clusters=n_clusters, random_state=23)

        cluster_labels[f'{col}_cluster'] = kmeans.fit_predict(embedding_matrix)
        
    # Add labels
    df = pd.concat([df, pd.DataFrame(cluster_labels, index=df.index)], axis=1)

    return df.drop(columns=original_string_df.columns)


def kmeans_cluster_embeddings():
    # For main script / testing
    # Reads intermediate processed data, applies KMeans clustering, and saves the final processed DataFrame.

    df = pd.read_csv('data/processed/temp_df.csv')
    embedding_df = pd.read_parquet('data/processed/temp_embedding_df.parquet')
    string_columns = pd.read_csv('data/processed/temp_string_columns.csv')

    df = apply_kmeans(df, embedding_df, string_columns)

    save_processed(df, 'model_inputs.csv')