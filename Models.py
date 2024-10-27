import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error
import os

# Define paths and logging
data_dir = r"F:\Project_PRC_Eurocontrol\NEW\2710"
model_dir = os.path.join(data_dir, 'Models')
output_rmse_path = os.path.join(data_dir, 'group_rmse_results.csv')
output_submission_predictions_path = os.path.join(data_dir, 'submission_predictions.csv')
os.makedirs(model_dir, exist_ok=True)

# Define paths for datasets
challenge_set_path = os.path.join(data_dir, 'Updated_challenge_set_final_final.csv')
submission_set_path = os.path.join(data_dir, 'Updated_submission_set_final_final.csv')

# Load datasets
challenge_set = pd.read_csv(challenge_set_path)
submission_set = pd.read_csv(submission_set_path)

# Helper function for preparing groups
def prepare_groups(challenge_df, submission_df, group_col, min_flights=100):
    combined_df = pd.concat([challenge_df[[group_col]], submission_df[[group_col]]], ignore_index=True)
    group_counts = combined_df[group_col].value_counts()
    valid_groups = group_counts[group_counts >= min_flights].index.tolist()
    submission_df['group_assigned'] = submission_df[group_col].apply(lambda x: x if x in valid_groups else 'Ungrouped')
    return valid_groups, submission_df

# Initialize an empty list to store RMSE results
rmse_results = []

# List of grouping columns
group_columns = [
    'wtc_Flight_distance_category',
    'wtc_day_or_night_season',
    'wtc_flight_category',
    'ADB2_RECAT-EU_Flight_distance_category',
    'ADB2_RECAT-EU_flight_category',
    'Toff_speed_binned',
    'landing_speed_binned',
]

# DataFrame to store predictions for each flight_id in submission_set
submission_predictions = pd.DataFrame({'flight_id': submission_set['flight_id']})

# Calculate RMSE for each group-specific pre-trained model and predict tow for submission set
for group_col in group_columns:
    print(f"Processing grouping column: {group_col}")
    
    # Get the list of valid groups based on flight counts
    valid_group_names, submission_set = prepare_groups(challenge_set, submission_set, group_col, min_flights=100)

    for group in valid_group_names:
        group_data = challenge_set[challenge_set[group_col] == group]
        actual_tow = group_data['tow']

        # Define the unique model path for the group
        sanitized_group = f"{group_col}_{str(group).replace('/', '_').replace(' ', '_')}"
        model_path = os.path.join(model_dir, f"model_{sanitized_group}")

        # Load the pre-trained model if it exists
        if os.path.exists(model_path):
            predictor = TabularPredictor.load(model_path)
            
            # Calculate RMSE for the challenge set
            predictions = predictor.predict(group_data.drop(columns=['flight_id', 'tow']))
            rmse_value = mean_squared_error(actual_tow, predictions, squared=False)
            print(f"RMSE for group '{group}' in column '{group_col}': {rmse_value:.4f}")
            rmse_results.append({
                'grouping_column': group_col,
                'group': group,
                'rmse': rmse_value
            })
            
            # Predict `tow` for the current group in the submission set
            group_submission_data = submission_set[submission_set[group_col] == group]
            if not group_submission_data.empty:
                group_submission_predictions = predictor.predict(group_submission_data.drop(columns=['flight_id']))
                
                # Add predictions to the submission_predictions DataFrame
                submission_predictions.loc[submission_set[group_col] == group, f'predicted_tow_{group_col}'] = group_submission_predictions.values
        else:
            print(f"Model for group '{group}' in column '{group_col}' not found. Skipping.")

# Handle any missing predictions in submission_predictions with a placeholder (e.g., NaN)
submission_predictions.fillna(np.nan, inplace=True)

# Save RMSE results to CSV
rmse_results_df = pd.DataFrame(rmse_results)
rmse_results_df.to_csv(output_rmse_path, index=False)
print(f"RMSE results saved to '{output_rmse_path}'")

# Save the final submission predictions to CSV
submission_predictions.to_csv(output_submission_predictions_path, index=False)
print(f"Predictions for submission set saved to '{output_submission_predictions_path}'")
