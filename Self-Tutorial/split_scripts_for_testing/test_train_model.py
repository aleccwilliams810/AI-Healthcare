import sys

# Add the path to the directory containing the 'Self-Tutorial' folder
sys.path.append('/content/AI-Healthcare/Self-Tutorial')

import numpy as np
import pandas as pd
import lightgbm as lgb
from models.lgbm import train_model_test, display_metrics, save_results, display_best_params, display_classification_report, plot_roc_curve, plot_confusion_matrix

def main():
    # Load data 
    best_params = pd.read_csv('data/processed/best_params.csv').to_dict(orient='records')[0]

    d_train = lgb.Dataset('data/processed/d_train.bin')
    d_val = lgb.Dataset('data/processed/d_val.bin')

    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')

    best_model = train_model_test(best_params, d_train, d_val)

    y_pred_prob = best_model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(float)

    metrics_df = display_metrics(y_test, y_pred, y_pred_prob)
    save_results(y_test, y_pred, y_pred_prob, metrics_df)
    
    display_best_params(best_model)
    display_classification_report(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_prob)
    plot_confusion_matrix(y_test, y_pred_prob)

if __name__ == "__main__":
    main()
