from scripts.data_manager import authenticate_and_initialize_client, load_data

def pull():
    # Authenticate and initialize the BigQuery client
    client = authenticate_and_initialize_client()
    
    # Load data from BigQuery
    patients_df, admissions_df, diagnoses_df = load_data(client)
    
    # Check if the DataFrames are loaded successfully
    if patients_df is not None and admissions_df is not None and diagnoses_df is not None:
        print("Data successfully loaded into DataFrames.")
        
    else:
        print("Failed to load one or more DataFrames.")

def main():
    pull()

if __name__ == "__main__":
    main()