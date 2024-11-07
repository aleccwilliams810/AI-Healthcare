import sys

# Add the path to the directory containing the 'Self-Tutorial' folder
sys.path.append('/content/AI-Healthcare/Self-Tutorial')


from models.lgbm import load_data, cross_validate
from scripts.pull_data import remove_temporary_files, save_processed
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    data_path = 'data/processed/model_inputs.csv'
    df = load_data(data_path)

    X = df.drop(columns=['hospital_expire_flag'])
    y = df['hospital_expire_flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'num_leaves': [20, 40, 100],
        'max_depth': [5, 15, 45],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'min_child_samples': [25, 50, 100],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    best_params, d_train, d_val = cross_validate(X_train, y_train, param_grid)

    best_params_df = pd.DataFrame([best_params])
    save_processed(best_params_df, 'best_params.csv')

    d_train.save_binary('data/processed/d_train.bin')
    d_val.save_binary('data/processed/d_val.bin')

    X_test.to_parquet('data/processed/X_test.parquet')
    y_test.to_parquet('data/processed/y_test.parquet')

if __name__ == '__main__':
    remove_temporary_files('data/processed/temp_df.csv', 'data/processed/temp_embedding_df.parquet', 'data/processed/temp_string_columns.csv')
    main()