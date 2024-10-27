import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os
from pathlib import Path
import logging
import re
from autogluon.tabular import TabularPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths and create directories
data_dir = Path(r"F:\Project_PRC_Eurocontrol\NEW\2710")
model_dir = data_dir / 'Models'
output_rmse_path = data_dir / 'group_rmse_results.csv'
model_dir.mkdir(parents=True, exist_ok=True)

# Define paths for datasets
challenge_set_path = data_dir / 'Updated_challenge_set_final_final.csv'
submission_set_path = data_dir / 'Updated_submission_set_final_final.csv'

# Load datasets with error handling
try:
    challenge_set = pd.read_csv(challenge_set_path)
    submission_set = pd.read_csv(submission_set_path)
except FileNotFoundError as e:
    logging.error(f"File not found: {e.filename}")
    exit(1)
except pd.errors.ParserError as e:
    logging.error(f"Parsing error: {e}")
    exit(1)

# Ensure target column exists
if 'tow' not in challenge_set.columns:
    logging.error("'tow' column not found in challenge_set.")
    exit(1)

# Helper function for preparing groups
def prepare_groups(challenge_df, submission_df, group_col, min_flights=100):
    if group_col not in challenge_df.columns or group_col not in submission_df.columns:
        raise ValueError(f"Grouping column '{group_col}' not found in one of the DataFrames.")
    
    combined_df = pd.concat([challenge_df[[group_col]], submission_df[[group_col]]], ignore_index=True)
    group_counts = combined_df[group_col].value_counts()
    valid_groups = group_counts[group_counts >= min_flights].index.tolist()
    submission_df = submission_df.copy()
    submission_df['group_assigned'] = submission_df[group_col].apply(lambda x: x if x in valid_groups else 'Ungrouped')
    return valid_groups, submission_df

# Function to sanitize group names
def sanitize_group_name(group_col, group):
    sanitized = re.sub(r'[^\w\-]', '_', f"{group_col}_{str(group)}")
    return sanitized

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
    logging.info(f"Processing grouping column: {group_col}")
    
    try:
        valid_group_names, submission_set = prepare_groups(challenge_set, submission_set, group_col, min_flights=100)
    except ValueError as e:
        logging.error(e)
        continue

    for group in valid_group_names:
        group_data = challenge_set[challenge_set[group_col] == group]
        actual_tow = group_data['tow']
        
        # Define the unique model path for the group
        sanitized_group = sanitize_group_name(group_col, group)
        model_path = model_dir / f"model_{sanitized_group}"
        
        # Load the pre-trained model if it exists
        if model_path.exists():
            try:
                predictor = TabularPredictor.load(str(model_path))
            except Exception as e:
                logging.error(f"Failed to load model at '{model_path}': {e}")
                continue
            
            # Prepare features by dropping unnecessary columns
            features_to_drop = ['flight_id', 'tow']
            try:
                predictions = predictor.predict(group_data.drop(columns=features_to_drop))
            except Exception as e:
                logging.error(f"Failed to make predictions for group '{group}' in column '{group_col}': {e}")
                continue
            
            # Calculate RMSE for the group
            rmse_value = mean_squared_error(actual_tow, predictions, squared=False)
            logging.info(f"RMSE for group '{group}' in column '{group_col}': {rmse_value:.4f}")
    
            # Append RMSE result to list
            rmse_results.append({
                'grouping_column': group_col,
                'group': group,
                'rmse': rmse_value
            })
            
            # Predict `tow` for the current group in the submission set
            group_submission_data = submission_set[submission_set[group_col] == group]
            if not group_submission_data.empty:
                try:
                    group_submission_predictions = predictor.predict(group_submission_data.drop(columns=['flight_id']))
                    # Add predictions to the submission_predictions DataFrame
                    prediction_col_name = f'predicted_tow_{group_col}'
                    submission_predictions.loc[submission_set[group_col] == group, prediction_col_name] = group_submission_predictions.values
                except Exception as e:
                    logging.error(f"Failed to predict for submission data for group '{group}' in column '{group_col}': {e}")
        else:
            logging.warning(f"Model for group '{group}' in column '{group_col}' not found at '{model_path}'. Skipping.")

# Convert rmse_results list to DataFrame and save it
rmse_results_df = pd.DataFrame(rmse_results)
rmse_results_df.to_csv(output_rmse_path, index=False)
logging.info(f"RMSE results saved to '{output_rmse_path}'")

# --- Start of inverse RMSE weighting implementation ---

# Ensure all necessary columns are present in submission_predictions
predicted_tow_cols = [col for col in submission_predictions.columns if col.startswith('predicted_tow_')]

# Read the RMSE results (already loaded as rmse_results_df)
# Map RMSE values to each flight based on group assignments
for grouping_column in group_columns:
    rmse_mapping = rmse_results_df[rmse_results_df['grouping_column'] == grouping_column].set_index('group')['rmse'].to_dict()
    rmse_col_name = f'rmse_{grouping_column}'
    weight_col_name = f'weight_{grouping_column}'
    group_col_name = grouping_column
    
    # Map RMSE to submission_set
    submission_predictions[rmse_col_name] = submission_set[group_col_name].map(rmse_mapping)
    # Calculate inverse RMSE weight
    submission_predictions[weight_col_name] = 1.0 / (submission_predictions[rmse_col_name] + 1e-6)

# Calculate the weighted average of predictions
# Identify weight columns corresponding to predictions
prediction_weight_pairs = []
for grouping_column in group_columns:
    predicted_tow_col = f'predicted_tow_{grouping_column}'
    weight_col = f'weight_{grouping_column}'
    if predicted_tow_col in submission_predictions.columns and weight_col in submission_predictions.columns:
        prediction_weight_pairs.append((predicted_tow_col, weight_col))

# Initialize numerator and denominator for weighted average
numerator = pd.Series(0, index=submission_predictions.index, dtype=float)
denominator = pd.Series(0, index=submission_predictions.index, dtype=float)

for pred_col, weight_col in prediction_weight_pairs:
    # Multiply predictions by their weights
    weighted_pred = submission_predictions[pred_col] * submission_predictions[weight_col]
    # Replace NaN values with 0 in weighted predictions and weights
    weighted_pred = weighted_pred.fillna(0)
    weight = submission_predictions[weight_col].fillna(0)
    numerator += weighted_pred
    denominator += weight

# Avoid division by zero
denominator = denominator.replace(0, np.nan)

# Compute final weighted prediction
submission_predictions['final_tow_prediction'] = numerator / denominator

# Handle missing predictions (if any)
# Option 1: Fill missing predictions with the mean of available predictions
submission_predictions['final_tow_prediction'].fillna(submission_predictions[predicted_tow_cols].mean(axis=1), inplace=True)

# Prepare final submission DataFrame
final_predictions_df = submission_predictions[['flight_id', 'final_tow_prediction']].copy()
final_predictions_df.rename(columns={'final_tow_prediction': 'tow'}, inplace=True)

# Save the final predictions to CSV
output_file = data_dir / 'final_tow_predictions.csv'
final_predictions_df.to_csv(output_file, index=False)
logging.info(f"Final predictions saved to '{output_file}'")

# --- End of inverse RMSE weighting implementation ---
