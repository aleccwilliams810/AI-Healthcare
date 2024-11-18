import sys
import json
import pandas as pd
import pickle
from models.build_nn import full_training

# Add the path to the directory
sys.path.append('/content/AI-Healthcare/ML_DL')

def main():
    # Load preprocessed dataset
    print("Loading dataset for final training...")
    X = pd.read_parquet('data/processed/X_pca.parquet').values
    y = pd.read_parquet('data/processed/y_pca.parquet').values.ravel()

    # Load best parameters from cross-validation
    print("Loading best parameters from cross-validation...")
    with open('models/best_params_cv.json', 'r') as f:
        best_params = json.load(f)

    # Extract selected features directly from best_params
    selected_features = best_params['selected_features']
    print(f"Using features selected during cross-validation: {selected_features}")

    # Train the final model
    print("Training final model with best parameters...")
    trained_model, history = full_training(
        X, y, best_params,
        epochs=30, batch_size=64
    )

    # Save the final model
    trained_model.save('models/final_model.h5')
    print("Final model saved to models/final_model.h5.")

    # Save training history
    with open('models/training_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    # Generate and save predictions
    print("Generating and saving predictions...")
    X_selected = X[:, selected_features]  # Ensure predictions use selected features
    y_pred_proba = trained_model.predict(X_selected).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)

    pd.DataFrame({
        'y_true': y,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }).to_csv('models/final_predictions.csv', index=False)

if __name__ == "__main__":
    main()
