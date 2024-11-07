############ Run script to continue processing cleaned data, embedding via vectorizing diag strings

import sys

# Add the path to the directory containing the 'Self-Tutorial' folder
sys.path.append('/content/AI-Healthcare/Self-Tutorial')

from scripts.join_postprocess import join_data, build_embeddings

def embed_data():
    df = join_data()
    build_embeddings(df)

def main():
    embed_data()

if __name__ == "__main__":
    main()
