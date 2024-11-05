import pandas as pd
import numpy as np
from scripts.pull_data import save_processed

num_diagnoses = 25

def process_diagnosis_data(csv_file):
    diagnoses_df = pd.read_csv(csv_file)

    #Convert to NaN for better compatibility with LGBM
    diagnoses_df = diagnoses_df.replace({None: np.nan})

    diagnoses_df = reorder_by_diagnosis_level(diagnoses_df)
    diagnoses_df = previous_diagnoses(diagnoses_df)
    diagnoses_df = engineer_first_diag_char_encoding(diagnoses_df)
    diagnoses_df = engineer_first_3_diag_char_encoding(diagnoses_df)
    diagnoses_df = engineer_remaining_diag_char_encoding(diagnoses_df)
    
    save_processed(diagnoses_df, 'diagnoses_cleaned.csv')

def reorder_by_diagnosis_level(df):
    # Ensure seq_num sorted within each subject_id
    df.sort_values(by=['subject_id', 'seq_num'], ascending=[True, True], inplace=True)

    # Reset seq_num to avoid ties
    df['seq_num'] = df.groupby('subject_id').cumcount() + 1

    return df

def previous_diagnoses(df):
    for i in range(2, num_diagnoses + 1):
        df[f'diag{i}'] = df.groupby('subject_id')['diag_code'].shift(-(i-1))
        df[f'diag{i}_desc'] = df.groupby('subject_id')['diag_code_desc'].shift(-(i-1))

    diag_columns = [f'diag{i}' if j % 2 == 0 else f'diag{i}_desc' for i in range(2, num_diagnoses + 1) for j in range(2)]

    # Create flattened DataFrame with desired columns
    df = df[df['seq_num'] == 1][
        ['subject_id', 'seq_num', 'diag_code', 'diag_code_desc'] + diag_columns
    ].rename(columns={'diag_code': 'primary_diag', 'diag_code_desc': 'primary_diag_desc'})

    return df

def character_encoding(code_part):
    if pd.isna(code_part):
        return np.nan
    return ''.join(['10' if char == 'E' else '11' if char == 'V' else char for char in code_part])

def engineer_first_diag_char_encoding(df):
    # Apply character encoding to the first character of each diagnosis
    diag_columns = [f'diag{i}' for i in range(2, num_diagnoses + 1)]

    new_cols = {
        f'{col}_first_char_encoded': df[col].astype(str).str[0].apply(character_encoding)
        for col in diag_columns
    }

    # Concatenate the original DataFrame with new features
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    return df

def engineer_first_3_diag_char_encoding(df):
    # Apply character encoding to the first 3 characters of each diagnosis column
    diag_columns = [f'diag{i}' for i in range(2, num_diagnoses + 1)]

    # Create new columns for the first 3 characters encoding
    new_cols = {
        f'{col}_first_3_encoded': df[col].str[:3].apply(character_encoding)
        for col in diag_columns
    }

    # Concatenate the original DataFrame with new columns
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    return df

def engineer_remaining_diag_char_encoding(df):
    # Extract the remaining characters after the first 3 from each diagnosis column
    diag_columns = [f'diag{i}' for i in range(2, num_diagnoses + 1)]

    # Create new columns for remaining characters encoding
    new_cols = {
        f'{col}_remaining_chars': df[col].apply(lambda x: x[3:] if isinstance(x, str) and len(x) > 3 else (np.nan if pd.isna(x) else np.nan))
        for col in diag_columns
    }

    # Concatenate the original DataFrame with new columns
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    return df

