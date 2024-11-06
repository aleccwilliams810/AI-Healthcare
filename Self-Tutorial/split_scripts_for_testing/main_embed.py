############ Run script to continue processing cleaned data, embedding via vectorizing diag strings

from scripts.join_postprocess import join_data, build_embeddings

def embed_data():
    df = join_data()
    build_embeddings(df)

def main():
    embed_data()

if __name__ == "__main__":
    main()
