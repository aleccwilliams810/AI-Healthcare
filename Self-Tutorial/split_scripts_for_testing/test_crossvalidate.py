import sys
import os

# Add the path to the directory containing the 'Self-Tutorial' folder
sys.path.append('/content/AI-Healthcare/Self-Tutorial')


from models.lgbm import load_data, cross_validate, upsample_minority
from scripts.pull_data import remove_temporary_files, save_processed
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    data_path = 'data/processed/model_inputs.csv'
    df = load_data(data_path)

    X = df.drop(columns=['hospital_expire_flag'])
    y = df['hospital_expire_flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = upsample_minority(X_train, y_train) 

    param_grid = {
        'num_leaves': [20, 40, 60, 80, 100],
        'max_depth': [5, 10, 20, 40],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 150, 200],
        'min_child_samples': [20, 50, 75, 100],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    best_params, d_train, d_val = cross_validate(X_train, y_train, param_grid)

    best_params_df = pd.DataFrame([best_params])
    save_processed(best_params_df, 'best_params.csv')

    d_train_path = 'data/processed/d_train.bin'
    d_val_path = 'data/processed/d_val.bin'

    # overwrite
    if os.path.exists(d_train_path):
        os.remove(d_train_path)
    if os.path.exists(d_val_path):
        os.remove(d_val_path)

    d_train.save_binary(d_train_path)
    d_val.save_binary(d_val_path)

    X_test.to_parquet('data/processed/X_test.parquet')

    y_test_df = pd.DataFrame(y_test)  # Convert to DataFrame to be saved as parquet and accessed in model testing
    y_test_df.to_parquet('data/processed/y_test.parquet')

if __name__ == '__main__':
    remove_temporary_files('data/processed/temp_df.csv', 'data/processed/temp_embedding_df.parquet', 'data/processed/temp_string_columns.csv')
    main()