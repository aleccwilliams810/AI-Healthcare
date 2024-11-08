import sys

# Add the path to the directory containing the 'Self-Tutorial' folder
sys.path.append('/content/AI-Healthcare/Self-Tutorial')

import pandas as pd
import lightgbm as lgb
from models.lgbm import train_model_test, display_metrics, save_results, display_classification_report, plot_roc_curve, plot_confusion_matrix

def main():
    # Load data 
    best_params = pd.read_csv('data/processed/best_params.csv').to_dict(orient='records')[0]

    d_train = lgb.Dataset('data/processed/d_train.bin')
    d_val = lgb.Dataset('data/processed/d_val.bin')

    X_test = pd.read_parquet('data/processed/X_test.parquet')
    y_test = pd.read_parquet('data/processed/y_test.parquet').squeeze()

    best_model = train_model_test(best_params, d_train, d_val)

    y_pred_prob = best_model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(float)

    print("\nBest Model Parameters and Score:")
    print(pd.DataFrame([best_params]))

    metrics_df = display_metrics(y_test, y_pred, y_pred_prob)
    save_results(y_test, y_pred, y_pred_prob, metrics_df)
    
    display_classification_report(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_prob)
    plot_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    main()
