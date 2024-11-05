import pandas as pd
import os

num_diagnoses = 25

def process_diagnosis_data(csv_file):
    diagnoses_df = pd.read_csv(csv_file)

    diagnoses_df = reorder_by_diagnosis_level(diagnoses_df)
    diagnoses_df = previous_diagnoses(diagnoses_df)
    diagnoses_df = engineer_first_diag_char_encoding(diagnoses_df)
    diagnoses_df = engineer_first_3_diag_char_encoding(diagnoses_df)
    diagnoses_df = engineer_remaining_diag_char_encoding(diagnoses_df)

    output_dir = "Self-Tutorial/data/processed"
    diagnoses_df.to_csv(os.path.join(output_dir, "diagnoses_cleaned.csv"), index=False)
    print("Diagnoses data saved to Self-Tutorial/data/processed/diagnoses_cleaned.csv")
    
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
    return ''.join(['10' if char == 'E' else '11' if char == 'V' else char for char in code_part])

def engineer_first_diag_char_encoding(df):
    # Ensure 'diag_code' is a string type
    df['diag_code'] = df['diag_code'].astype(str)

    # Apply first character encoding
    df['diag_first_char_encoded'] = df['diag_code'].str[0].apply(character_encoding)

    return df
    
def engineer_first_3_diag_char_encoding(df):
    df['diag_first_3_encoded'] = df['diag_code'].str[:3].apply(character_encoding)

    return df

def engineer_remaining_diag_char_encoding(df):
    df['diag_remaining_chars'] = df['diag_code'].apply(
        lambda x: x[3:] if len(x) > 3 else ''
    )

    return df