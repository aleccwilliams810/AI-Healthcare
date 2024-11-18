import pandas as pd
import numpy as np
from scripts.data_manager import save_processed

num_diagnoses = 15

def process_diagnosis_data(csv_file):
    diagnoses_df = pd.read_csv(csv_file)

    diagnoses_df = prioritize_and_bucket_diags(diagnoses_df)
    diagnoses_df = flatten_diagnoses(diagnoses_df)
    diagnoses_df = engineer_first_diag_char_encoding(diagnoses_df)
    diagnoses_df = engineer_first_3_diag_char_encoding(diagnoses_df)
    diagnoses_df = engineer_remaining_diag_char_encoding(diagnoses_df)
    
    save_processed(diagnoses_df, 'diagnoses_cleaned.csv')

##### Setting up diagnosis data by patient, by priority, to efficiently flatten necessary data
def prioritize_and_bucket_diags(df):
    # Sort by patient and sequence, compute total diagnoses per patient, and bucket counts into encoded ranges.
    df.sort_values(by=['subject_id', 'seq_num'], inplace=True)
    df['seq_num'] = df.groupby('subject_id').cumcount() + 1
    df['diag_count'] = df['seq_num'].groupby(df['subject_id']).transform('max')

    # Create and encode diagnosis count buckets
    df['diag_count_bucket_encoded'] = pd.cut(df['diag_count'], bins=[0, 3, 5, 7, 15, np.inf], 
                                             labels=[1, 2, 3, 4, 5], right=True).astype(int)
    return df

# Using priority ranked diagnoses, keep the row containing the highest priority diag, and add remaining diagnoses columns, up to num_diagnoses
def flatten_diagnoses(df):
    for i in range(2, num_diagnoses + 1):
        df[f'diag{i}'] = df.groupby('subject_id')['diag_code'].shift(-(i-1))
        df[f'diag{i}_desc'] = df.groupby('subject_id')['diag_code_desc'].shift(-(i-1))

    diag_columns = [f'diag{i}' if j % 2 == 0 else f'diag{i}_desc' for i in range(2, num_diagnoses + 1) for j in range(2)]

    # Create flattened DataFrame with desired columns, primary diag and description - then remaining diags and descriptions
    df = df[df['seq_num'] == 1][
        ['subject_id', 'seq_num', 'diag_code', 'diag_code_desc'] + diag_columns
    ].rename(columns={'diag_code': 'primary_diag', 'diag_code_desc': 'primary_diag_desc'})

    return df

####### Feature Engineering Methods, meant to isolate significant features of diagnosis code to give model additional information about ICD-9 structure

# Encoding the ICD-9 diag code, assigning significant characters to unique values for model compatibility. (E= 10, V = 11, n= blank, extra character occassionally tied to icd-9's)
def character_encoding(code_part):
    if pd.isna(code_part):
        return np.nan
    encoded_str = ''.join([
        '10' if char == 'E' 
        else '11' if char == 'V' 
        else '' if char == 'n'
        else char for char in code_part])

    # convert to numeric data type for compatibility
    if encoded_str.isdigit():
        return float(encoded_str)
    else:
        return np.nan

# Engineered Feature: Specifically isolates first character in code
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

# Engineered Feature: Specifically isolates first 3 characters in code
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

# Engineered Feature: Specifically isolates remaining characters in code after the first 3
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

