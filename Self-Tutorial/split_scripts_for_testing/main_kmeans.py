############ Run script to finalize post-processing by clustering diagnosis description embeddings with kmeans

import sys

# Add the path to the directory containing the 'Self-Tutorial' folder
sys.path.append('/content/AI-Healthcare/Self-Tutorial')

from scripts.join_postprocess import kmeans_cluster_embeddings

def run_kmeans():
    kmeans_cluster_embeddings()

def main():
    run_kmeans()

if __name__ == "__main__":
    main()
