import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os
from pathlib import Path
import logging
import re
from autogluon.tabular import TabularPredictor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor

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

# DataFrames to store predictions for each flight_id in challenge_set and submission_set
challenge_predictions = pd.DataFrame({'flight_id': challenge_set['flight_id']})
submission_predictions = pd.DataFrame({'flight_id': submission_set['flight_id']})

# Calculate RMSE for each group-specific pre-trained model and predict tow for both challenge and submission sets
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
                # Predict on challenge set
                challenge_preds = predictor.predict(group_data.drop(columns=features_to_drop))
                challenge_pred_col = f'predicted_tow_{group_col}'
                challenge_predictions.loc[group_data.index, challenge_pred_col] = challenge_preds.values

                # Predict on submission set
                group_submission_data = submission_set[submission_set[group_col] == group]
                if not group_submission_data.empty:
                    submission_preds = predictor.predict(group_submission_data.drop(columns=['flight_id']))
                    submission_pred_col = f'predicted_tow_{group_col}'
                    submission_predictions.loc[group_submission_data.index, submission_pred_col] = submission_preds.values

                # Calculate RMSE for the group
                rmse_value = mean_squared_error(actual_tow, challenge_preds, squared=False)
                logging.info(f"RMSE for group '{group}' in column '{group_col}': {rmse_value:.4f}")

                # Append RMSE result to list
                rmse_results.append({
                    'grouping_column': group_col,
                    'group': group,
                    'rmse': rmse_value
                })
            except Exception as e:
                logging.error(f"Failed to make predictions for group '{group}' in column '{group_col}': {e}")
                continue
        else:
            logging.warning(f"Model for group '{group}' in column '{group_col}' not found at '{model_path}'. Skipping.")

# Convert rmse_results list to DataFrame and save it
rmse_results_df = pd.DataFrame(rmse_results)
rmse_results_df.to_csv(output_rmse_path, index=False)
logging.info(f"RMSE results saved to '{output_rmse_path}'")

# --- Start of ensemble methods implementation ---

# Merge actual 'tow' values to challenge_predictions
challenge_predictions = challenge_predictions.merge(challenge_set[['flight_id', 'tow']], on='flight_id')

# Prepare data for meta-model training
# Get list of prediction columns
prediction_cols = [col for col in challenge_predictions.columns if col.startswith('predicted_tow_')]

# Remove rows with all NaN predictions
challenge_predictions.dropna(subset=prediction_cols, how='all', inplace=True)

# Fill missing prediction values with mean of available predictions for that row
challenge_predictions[prediction_cols] = challenge_predictions[prediction_cols].apply(
    lambda row: row.fillna(row.mean()), axis=1
)

# Handle any remaining NaNs by filling with overall mean tow from challenge set
overall_mean_tow = challenge_set['tow'].mean()
challenge_predictions[prediction_cols] = challenge_predictions[prediction_cols].fillna(overall_mean_tow)
submission_predictions[prediction_cols] = submission_predictions[prediction_cols].apply(
    lambda row: row.fillna(row.mean()), axis=1
)
submission_predictions[prediction_cols] = submission_predictions[prediction_cols].fillna(overall_mean_tow)

# Prepare features (predictions) and target
X_meta = challenge_predictions[prediction_cols]
y_meta = challenge_predictions['tow']
X_submission_meta = submission_predictions[prediction_cols]

# Store final predictions from different methods
final_predictions = pd.DataFrame({'flight_id': submission_predictions['flight_id']})

# 1. Inverse RMSE Weighting
# Map RMSE values to weights
rmse_results_df['inverse_rmse'] = 1 / (rmse_results_df['rmse'] + 1e-6)
rmse_results_df['weight'] = rmse_results_df['inverse_rmse']
weight_mapping = {}

for grouping_column in group_columns:
    group_rmse = rmse_results_df[rmse_results_df['grouping_column'] == grouping_column]
    for idx, row in group_rmse.iterrows():
        pred_col = f'predicted_tow_{grouping_column}'
        if pred_col in prediction_cols:
            if pred_col not in weight_mapping:
                weight_mapping[pred_col] = 0
            weight_mapping[pred_col] += row['weight']

# Normalize weights
total_weight = sum(weight_mapping.values())
for key in weight_mapping:
    weight_mapping[key] /= total_weight

# Calculate weighted average
weighted_preds = np.zeros(len(submission_predictions))
for pred_col in prediction_cols:
    weight = weight_mapping.get(pred_col, 0)
    weighted_preds += submission_predictions[pred_col] * weight

final_predictions['tow_inverse_rmse_weighting'] = weighted_preds

# 2. Stacking with Linear Regression
meta_model_lr = LinearRegression()
meta_model_lr.fit(X_meta, y_meta)
final_predictions['tow_stacking_lr'] = meta_model_lr.predict(X_submission_meta)

# 3. Stacking with Gradient Boosting Regressor
meta_model_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
meta_model_gbr.fit(X_meta, y_meta)
final_predictions['tow_stacking_gbr'] = meta_model_gbr.predict(X_submission_meta)

# 4. Weighted Average Ensemble
# Simple average of predictions
final_predictions['tow_weighted_average'] = submission_predictions[prediction_cols].mean(axis=1)

# 5. Weighted Median Ensemble
def weighted_median(values, weights):
    sorted_idx = np.argsort(values)
    values = values[sorted_idx]
    weights = weights[sorted_idx]
    cumulative_weight = np.cumsum(weights)
    cutoff = weights.sum() / 2.0
    return values[cumulative_weight >= cutoff][0]

# Compute weighted median for each row
def compute_row_weighted_median(row):
    preds = []
    weights = []
    for col in prediction_cols:
        pred = row[col]
        weight = weight_mapping.get(col, 0)
        if not pd.isna(pred):
            preds.append(pred)
            weights.append(weight)
    if preds:
        return weighted_median(np.array(preds), np.array(weights))
    else:
        return np.nan

submission_predictions['tow_weighted_median'] = submission_predictions.apply(
    compute_row_weighted_median, axis=1
)
final_predictions['tow_weighted_median'] = submission_predictions['tow_weighted_median']

# 6. Neural Network Meta-Model
meta_model_nn = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', random_state=42, max_iter=500)
meta_model_nn.fit(X_meta, y_meta)
final_predictions['tow_stacking_nn'] = meta_model_nn.predict(X_submission_meta)

# 7. Voting Regressor Ensemble
estimators = []
for idx, col in enumerate(prediction_cols):
    # Use individual columns as predictors
    reg = LinearRegression()
    reg.fit(challenge_predictions[[col]], y_meta)
    estimators.append((f'model_{idx}', reg))

voting_regressor = VotingRegressor(estimators=estimators)
voting_regressor.fit(challenge_predictions[prediction_cols], y_meta)
final_predictions['tow_voting_regressor'] = voting_regressor.predict(X_submission_meta)

# Save final predictions to CSV
output_file = data_dir / 'final_tow_predictions_all_methods.csv'
final_predictions.to_csv(output_file, index=False)
logging.info(f"Final predictions saved to '{output_file}'")

# --- End of ensemble methods implementation ---
