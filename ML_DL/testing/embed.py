import sys

# Add the path to the directory
sys.path.append('/content/AI-Healthcare/ML_DL')

from scripts.merge_and_process import join_data, calc_age
from scripts.apply_embeddings import build_embeddings

def embed_data():
    df = join_data()
    df = calc_age(df)
    build_embeddings(df)

def main():
    embed_data()

if __name__ == "__main__":
    main()