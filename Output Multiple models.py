import pandas as pd

# Load the original submission predictions file
input_file = r"F:\Project_PRC_Eurocontrol\NEW\2710\submission_predictions.csv"
output_dir = r"F:\Project_PRC_Eurocontrol\NEW\2710"
team_id = "9062515d-e10d-4020-8962-706d92e540a0"
version_number = 27  # Starting version number

# Load the submission predictions file
submission_predictions = pd.read_csv(input_file)

# Filter columns to only include flight_id and tow predictions
tow_columns = [col for col in submission_predictions.columns if col.startswith('predicted_tow_')]

# Generate separate CSV files for each `tow` column
for tow_col in tow_columns:
    # Create a new DataFrame with only `flight_id` and the current `tow` prediction
    tow_df = submission_predictions[['flight_id', tow_col]].copy()
    tow_df.rename(columns={tow_col: 'tow'}, inplace=True)
    
    # Define the output file name based on the version number and team ID
    output_file = f"{output_dir}/team_honest_turtle_v{version_number}_{team_id}.csv"
    
    # Save to CSV
    tow_df.to_csv(output_file, index=False)
    print(f"Saved {output_file}")
    
    # Increment the version number for the next file
    version_number += 1
