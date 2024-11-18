import sys

# Add the path to the directory
sys.path.append('/content/AI-Healthcare/ML_DL')

from scripts.apply_kmeans import kmeans_cluster_embeddings

def run_kmeans():
    kmeans_cluster_embeddings()

def main():
    run_kmeans()

if __name__ == "__main__":
    main()
