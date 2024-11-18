import sys

# Add the path to the directory
sys.path.append('/content/AI-Healthcare/ML_DL')

import pandas as pd
import pickle
from scripts.display_results import display_metrics, plot_confusion_matrix, plot_roc_curve, plot_training_history

def main():
    # Load predictions and ground truth
    print("Loading predictions and ground truth...")
    predictions = pd.read_csv('models/final_predictions.csv')
    y_true = predictions['y_true']
    y_pred = predictions['y_pred']
    y_pred_proba = predictions['y_pred_proba']

    # Display confusion matrix
    print("Displaying confusion matrix...")
    plot_confusion_matrix(y_true, y_pred)

    # Display ROC curve
    print("Displaying ROC curve...")
    plot_roc_curve(y_true, y_pred_proba)

    # Display metrics
    print("Displaying metrics...")
    display_metrics(y_true, y_pred, y_pred_proba)

    # Load and display training history
    print("Loading and displaying training history...")
    with open('models/training_history.pkl', 'rb') as f:
        history = pickle.load(f)
    plot_training_history(history)

if __name__ == "__main__":
    main()