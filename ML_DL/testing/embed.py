from scripts.merge_and_process import join_data
from scripts.apply_embeddings import build_embeddings

def embed_data():
    df = join_data()
    build_embeddings(df)

def main():
    embed_data()

if __name__ == "__main__":
    main()