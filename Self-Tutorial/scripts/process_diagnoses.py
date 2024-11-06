import pandas as pd
import numpy as np
from scripts.pull_data import save_processed

num_diagnoses = 15

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
    # Sort, reset seq_num, and count diagnoses per subject
    df.sort_values(by=['subject_id', 'seq_num'], inplace=True)
    df['seq_num'] = df.groupby('subject_id').cumcount() + 1
    df['diag_count'] = df['seq_num'].groupby(df['subject_id']).transform('max')

    # Create and encode diagnosis count buckets
    df['diag_count_bucket_encoded'] = pd.cut(df['diag_count'], bins=[0, 3, 5, 7, 15, np.inf], 
                                             labels=[1, 2, 3, 4, 5], right=True).astype(int)
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
    encoded_str = ''.join([
        '10' if char == 'E' 
        else '11' if char == 'V' 
        else '' if char == 'n'
        else char for char in code_part])

    if encoded_str.isdigit():
        return int(encoded_str)
    else:
        return np.nan

def engineer_first_diag_char_encoding(df):
    # Apply character encoding to the first character of each diagnosis
    diag_columns = [f'diag{i}' for i in range(2, num_diagnoses + 1)]

    new_cols = {
        f'{col}_first_char_encoded': pd.to_numeric(
            df[col].astype(str).str[0].apply(character_encoding), errors='coerce'
        )
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
        f'{col}_first_3_encoded': pd.to_numeric(
            df[col].str[:3].apply(character_encoding), errors='coerce'
        )
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
        f'{col}_remaining_chars': pd.to_numeric(
            df[col].apply(lambda x: x[3:] if isinstance(x, str) and len(x) > 3 else (np.nan if pd.isna(x) else np.nan)), errors='coerce'
        )
        for col in diag_columns
    }

    # Concatenate the original DataFrame with new columns
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    return df

