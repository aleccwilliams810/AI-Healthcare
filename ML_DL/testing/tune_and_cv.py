import sys

# Add the path to the directory
sys.path.append('/content/AI-Healthcare/ML_DL')

import pandas as pd
import pickle
import json
from models.build_nn import load_data, preprocess_data, apply_pca, random_search_ffnn

def main():
    data_path = 'data/processed/model_inputs.csv'
    target_column = 'hospital_expire_flag'

    # Load and preprocess data
    df = load_data(data_path)
    X, y = preprocess_data(df, target_column)

    # Apply PCA to halve dimensionality - too many interdependancies in engineered features
    print("Applying PCA to reduce dimensionality...")
    X_reduced, pca = apply_pca(X, n_components=X.shape[1] // 2)

    # Save the PCA-transformed dataset and PCA object
    pd.DataFrame(X_reduced).to_parquet('data/processed/X_pca.parquet')
    pd.DataFrame(y).to_parquet('data/processed/y_pca.parquet')
    with open('models/pca.pkl', 'wb') as f:
        pickle.dump(pca, f)

    print("PCA-transformed dataset and PCA object saved.")


    # Hyperparameter search space
    param_grid = {
        'hidden_layers': [[128, 64], [64, 32, 16], [256, 128]],
        'dropout_rate': [0.1, 0.2, 0.3, 0.5],
        'learning_rate': [0.001, 0.01, 0.1]
    }

    # Perform random search with sampled data and reduced dimensionality
    best_params = random_search_ffnn(X_reduced, y, param_grid, n_iter=10, sample_fraction=0.1, epochs=10, batch_size=32, initial_k_values=[10, 20, 30])

    with open('models/best_params_cv.json', 'w') as f:
        json.dump(best_params, f, indent=4)

if __name__ == "__main__":
    main()
