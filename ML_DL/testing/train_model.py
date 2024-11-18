import sys

# Add the path to the directory
sys.path.append('/content/AI-Healthcare/ML_DL')

import pandas as pd
import pickle
from models.build_nn import full_training

def main():
    # Load pre-transformed PCA dataset
    print("Loading PCA-transformed dataset for final training...")
    X_reduced = pd.read_parquet('data/processed/X_pca.parquet').values
    y = pd.read_parquet('data/processed/y_pca.parquet').values.ravel()

    # Load best parameters from cross-validation
    print("Loading best parameters from cross-validation...")
    best_params = pd.read_csv('results/best_params_cv.csv').to_dict(orient='records')[0]

    # Use the number of features (`k`) saved during CV
    num_features = best_params['num_features']
    print(f"Using {num_features} features selected during cross-validation.")

    # Train the final model
    print("Training final model with best parameters...")
    trained_model, history = full_training(
        X_reduced, y, best_params,
        n_components=num_features,  # Select top `k` features based on best_params
        epochs=30, batch_size=64
    )

    # Save the final model
    trained_model.save('models/final_model.h5')

    print("Final model saved to models/final_model.h5.")

    # Save the final model
    trained_model.save('models/final_model.h5')

    # Save training history
    with open('models/training_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    # Generate and save predictions
    print("Generating and saving predictions...")
    y_pred_proba = trained_model.predict(X_reduced).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)

    pd.DataFrame({
        'y_true': y,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }).to_csv('results/final_predictions.csv', index=False)

if __name__ == "__main__":
    main()
