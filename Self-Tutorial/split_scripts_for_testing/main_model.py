import sys

# Add the path to the directory containing the 'Self-Tutorial' folder
sys.path.append('/content/AI-Healthcare/Self-Tutorial')

from models.lgbm import main
from scripts.pull_data import remove_temporary_files

if __name__ == '__main__':
    remove_temporary_files('data/processed/temp_df.csv', 'data/processed/temp_embedding_df.parquet', 'data/processed/temp_string_columns.csv')
    main()