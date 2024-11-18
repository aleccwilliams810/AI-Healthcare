import pandas as pd
import spacy
from tqdm import tqdm
from scripts.data_manager import save_processed


###### Install scispacy model with !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz
nlp = spacy.load("en_core_sci_md")



###### Embedding Diagnosis Descriptions
def get_embeddings(texts):
    #Generates embeddings for a list of texts using SciSpacy's en_core_sci_md NLP model.
    #Processes in batches for efficiency.

    embeddings = []
    for doc_batch in nlp.pipe(texts, batch_size=1000, n_process=4, disable=["parser", "tagger"]):
        embeddings.append(doc_batch.vector)
    return embeddings

def apply_embeddings_function(df):
    # Embeds diagnostic description columns

    # Gather all string columns
    string_columns = [col for col in df.columns if 'consult_diag' in col or col.endswith('_desc')]
    
    embedding_cols = {}

    #Process each col, tqdm will show progress bar.
    for col in tqdm(string_columns, desc=f"Embedding columns..."):
        print(f"\nProcessing column: {col}")
        
        embeddings = get_embeddings(df[col].fillna('').tolist())
        
        # Store embeddings with the new column name
        embedding_cols[f'{col}_embedding'] = embeddings
        print(f"Completed embeddings for column: {col}")

    # Create embedding DF
    embedding_df = pd.DataFrame(embedding_cols, index=df.index)

    # Return the embedding DataFrame and the original DataFrame with string columns dropped
    return embedding_df, df[string_columns]


def build_embeddings(df):
    # For main script / testing
    # Processes the DataFrame to calculate age and generate diagnostic description embeddings.
    # Saves intermediate results for clustering.

    embedding_df, string_columns = apply_embeddings_function(df)

    save_processed(df, 'temp_df.csv')
    embedding_df.to_parquet('data/processed/temp_embedding_df.parquet')
    save_processed(string_columns, 'temp_string_columns.csv')