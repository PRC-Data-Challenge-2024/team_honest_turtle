import pandas as pd
import numpy as np
import warnings
import os
import logging
from autogluon.tabular import TabularPredictor, FeatureMetadata
import torch
import chardet
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Suppress warnings
warnings.filterwarnings('ignore')

# Import GPU-accelerated libraries
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU is available. Using RAPIDS cuDF and cuML for GPU-based processing.")
    logging.info("GPU is available. Using RAPIDS cuDF and cuML for GPU-based processing.")
except ImportError:
    print("RAPIDS cuDF and cuML not installed. Proceeding with CPU-based processing.")
    logging.warning("RAPIDS cuDF and cuML not installed. Proceeding with CPU-based processing.")
    GPU_AVAILABLE = False

# ================================
# ====== Configure Logging =======
# ================================
# Base directories
data_dir = r"F:\Project_PRC_Eurocontrol\NEW\Input_data\BASE_DATA\7"
model_dir = os.path.join(data_dir, 'Models')
os.makedirs(model_dir, exist_ok=True)
os.chdir(data_dir)

# Set up logging
log_file_path = os.path.join(model_dir, 'process_log23.10.24_AG_2152.txt')
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Starting the data processing and model training script without collinearity-based feature removal.")
print("Starting the data processing and model training script without collinearity-based feature removal.")

# ==========================================
# ====== Define Helper Functions ===========
# ==========================================

def load_csv_with_encoding(filepath):
    """
    Load a CSV file with automatic encoding detection.
    """
    try:
        with open(filepath, 'rb') as f:
            rawdata = f.read(10000000)  # Read first 10MB
            result = chardet.detect(rawdata)
            encoding = result['encoding']
            confidence = result['confidence']
            logging.info(f"Detected encoding for '{filepath}': {encoding} with confidence {confidence}")
            print(f"Detected encoding for '{filepath}': {encoding} with confidence {confidence}")
        
        parse_dates = ['actual_offblock_time', 'arrival_time'] if 'challenge_set.csv' in filepath or 'submission_set.csv' in filepath else None
        df = pd.read_csv(filepath, encoding=encoding, parse_dates=parse_dates)
        logging.info(f"Successfully loaded '{filepath}'. Shape: {df.shape}")
        print(f"Successfully loaded '{filepath}'. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load '{filepath}': {e}")
        print(f"Failed to load '{filepath}': {e}")
        raise

def create_Airport_pair(df, df_name):
    """
    Create a unique Airport_pair identifier by concatenating 'adep' and 'ades'.
    """
    if 'adep' in df.columns and 'ades' in df.columns:
        if 'country_code_adep' in df.columns and 'country_code_ades' in df.columns:
            df['Airport_pair'] = df['country_code_adep'] + '_' + df['adep'] + '_' + df['country_code_ades'] + '_' + df['ades']
        else:
            df['Airport_pair'] = df['adep'] + '_' + df['ades']
    else:
        print(f"Warning: The required columns 'adep' and 'ades' are missing in the dataset {df_name}. Skipping 'Airport_pair' creation.")
        df['Airport_pair'] = None
    return df

def extract_year_month(df, df_name):
    """
    Extract Year and Month from 'actual_offblock_time' datetime column.
    """
    if 'actual_offblock_time' in df.columns:
        df['Year_departure'] = df['actual_offblock_time'].dt.year
        df['Month_departure'] = df['actual_offblock_time'].dt.month
    else:
        logging.warning(f"'actual_offblock_time' is missing in {df_name}. Unable to extract year and month.")
        print(f"Warning: 'actual_offblock_time' is missing in {df_name}. Unable to extract year and month.")
    return df

def resolve_duplicate_columns(df, df_name):
    """
    Remove duplicate columns that resulted from merges.
    """
    df = df.loc[:, ~df.columns.duplicated()]
    logging.info(f"Removed duplicate columns in {df_name}.")
    print(f"Removed duplicate columns in {df_name}.")
    return df

def identify_datetime_columns(df):
    """
    Identify columns that are datetime or can be converted to datetime.
    """
    datetime_cols = []
    for col in df.columns:
        if df[col].dtype.kind in ['M', 'O']:  # 'M' is for datetime64, 'O' is for object
            try:
                pd.to_datetime(df[col], errors='raise')
                datetime_cols.append(col)
            except (ValueError, TypeError):
                continue
    return datetime_cols

def convert_datetime_columns(df_train, df_submission, datetime_cols):
    """
    Convert datetime columns to numerical features.
    """
    for col in datetime_cols:
        # Convert to UNIX timestamp
        df_train[col + '_timestamp'] = pd.to_datetime(df_train[col], errors='coerce').astype(np.int64) // 10**9
        df_submission[col + '_timestamp'] = pd.to_datetime(df_submission[col], errors='coerce').astype(np.int64) // 10**9
        
        # Optionally extract additional features
        df_train[col + '_year'] = pd.to_datetime(df_train[col], errors='coerce').dt.year
        df_train[col + '_month'] = pd.to_datetime(df_train[col], errors='coerce').dt.month
        df_train[col + '_day'] = pd.to_datetime(df_train[col], errors='coerce').dt.day
        df_train[col + '_hour'] = pd.to_datetime(df_train[col], errors='coerce').dt.hour
        df_train[col + '_minute'] = pd.to_datetime(df_train[col], errors='coerce').dt.minute
        
        df_submission[col + '_year'] = pd.to_datetime(df_submission[col], errors='coerce').dt.year
        df_submission[col + '_month'] = pd.to_datetime(df_submission[col], errors='coerce').dt.month
        df_submission[col + '_day'] = pd.to_datetime(df_submission[col], errors='coerce').dt.day
        df_submission[col + '_hour'] = pd.to_datetime(df_submission[col], errors='coerce').dt.hour
        df_submission[col + '_minute'] = pd.to_datetime(df_submission[col], errors='coerce').dt.minute
        
        # Drop the original datetime column
        df_train.drop(columns=[col], inplace=True)
        df_submission.drop(columns=[col], inplace=True)
    
    return df_train, df_submission

def impute_missing_values(train_df, submission_df, target_columns, predictor_save_path, time_limit=100):
    """
    Impute missing values in the specified columns using AutoGluon with GPU support.
    
    Parameters:
    - train_df: pandas DataFrame for training
    - submission_df: pandas DataFrame for submission
    - target_columns: list of column names to impute
    - predictor_save_path: directory to save imputation models
    - time_limit: time limit per imputation model in seconds (default: 100 seconds)
    
    Returns:
    - train_df: DataFrame with imputed values
    - submission_df: DataFrame with imputed values
    - imputation_metrics: DataFrame containing RMSE and MAE for each imputed column
    """
    os.makedirs(predictor_save_path, exist_ok=True)
    metrics = []
    
    for col in target_columns:
        print(f"\nImputing missing values for column: {col}")
        logging.info(f"Imputing missing values for column: {col}")
        
        # Define features (exclude the target column and any identifier columns)
        features = [f for f in train_df.columns if f != col and f != 'flight_id']
        
        # Split data into known and unknown
        known_df = train_df[train_df[col].notna()]
        unknown_df = train_df[train_df[col].isna()]
        
        if unknown_df.empty:
            print(f"No missing values found for column: {col}. Skipping imputation.")
            logging.info(f"No missing values found for column: {col}. Skipping imputation.")
            continue
        
        # Further split known_df into training and validation for evaluation
        train_known, val_known = train_test_split(known_df, test_size=0.2, random_state=42)
        
        # Initialize AutoGluon predictor for imputation using regression
        imputer = TabularPredictor(
            label=col,
            path=os.path.join(predictor_save_path, f'imputer_{col}'),
            problem_type='regression',
            eval_metric='mae',
            verbosity=4  # Set to 3 for detailed logs
        )
        
        # Fit the imputer with AutoGluon using GPU if available
        imputer.fit(
            train_data=train_known,
            tuning_data=val_known,
            hyperparameters={
                'GBM': {},
                'XGB': {}
            },
            feature_metadata=FeatureMetadata.from_df(train_known[features]),
            ag_args_fit={
                'num_gpus': 1 if GPU_AVAILABLE else 0,
                'memory_limit': 16000,
            },
            time_limit=time_limit,  # Adjusted time limit
        )
        
        # Predict missing values
        preds = imputer.predict(unknown_df[features])
        
        # Fill in the missing values
        train_df.loc[train_df[col].isna(), col] = preds
        
        print(f"Imputed {unknown_df.shape[0]} missing values for column: {col}")
        logging.info(f"Imputed {unknown_df.shape[0]} missing values for column: {col}")
        
        # Evaluate imputation on validation set
        val_preds = imputer.predict(val_known[features])
        rmse = np.sqrt(mean_squared_error(val_known[col], val_preds))
        mae = mean_absolute_error(val_known[col], val_preds)
        
        print(f"Imputation Performance for {col} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        logging.info(f"Imputation Performance for {col} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        metrics.append({
            'column': col,
            'RMSE': rmse,
            'MAE': mae
        })
        
        # Optionally, save the imputation model
        imputer.save(os.path.join(predictor_save_path, f'imputer_{col}'))
        
    imputation_metrics = pd.DataFrame(metrics)
    return train_df, submission_df, imputation_metrics

# ==========================================
# ====== Load All Datasets =================
# ==========================================

# Define input file paths
input_files = {
    'ADB2': os.path.join(data_dir, 'ADB2.csv'),
    'Airport_Dimension': os.path.join(data_dir, 'Airport_Dimension.csv'),
    'Airport_Movement_2022': os.path.join(data_dir, 'Airport_Movement_2022.csv'),
    'Airport_weather': os.path.join(data_dir, 'Airport_weather.csv'),
    'challenge_set': os.path.join(data_dir, 'challenge_set.csv'),
    'submission_set': os.path.join(data_dir, 'final_submission_set.csv'),
    'Flight_Characteristics': os.path.join(data_dir, 'Flight_Characteristics_191024.csv'),
    'Flight_blocks_processed': os.path.join(data_dir, 'flight_blocks_processed.csv'),
    'Euro_passenger': os.path.join(data_dir, 'Euro_passenger_2010_2024.csv'),
    'combined_takeoff_speeds': os.path.join(data_dir, 'combined_takeoff_speeds.csv'),
    'all_landing_speeds': os.path.join(data_dir, 'all_landing_speeds_2022.csv')  # New Landing Speeds Dataset
}

# Load all datasets
datasets = {}
for key, path in input_files.items():
    datasets[key] = load_csv_with_encoding(path)
    gc.collect()  # Free up memory

# Extract individual DataFrames
flight_characteristics = datasets['Flight_Characteristics']
challenge_set = datasets['challenge_set']
submission_set = datasets['submission_set']
climate_data = datasets['Airport_weather']
adb_data = datasets['ADB2']
flight_blocks_processed = datasets['Flight_blocks_processed']
euro_passenger = datasets['Euro_passenger']
airport_dimension = datasets['Airport_Dimension']
airport_movement = datasets['Airport_Movement_2022']
combined_takeoff_speeds = datasets['combined_takeoff_speeds']
all_landing_speeds = datasets['all_landing_speeds']  # New Landing Speeds DataFrame

logging.info("All datasets loaded successfully.")
print("All datasets loaded successfully.")

# ==========================================
# ====== Data Cleaning and Preparation =====
# ==========================================

# Replace inf values with NaN in flight_characteristics
flight_characteristics.replace([np.inf, -np.inf], np.nan, inplace=True)
logging.info("Replaced inf values with NaN in flight_characteristics.")
print("Replaced inf values with NaN in flight_characteristics.")

# Drop 'tow' from flight_characteristics to prevent duplication
if 'tow' in flight_characteristics.columns:
    flight_characteristics = flight_characteristics.drop(columns=['tow'])
    logging.info("Dropped 'tow' from flight_characteristics to prevent duplication.")
    print("Dropped 'tow' from flight_characteristics to prevent duplication.")

# **Ensure 'flight_id' is present in all relevant datasets before merging**
required_datasets = ['challenge_set', 'submission_set', 'Flight_Characteristics', 'flight_blocks_processed', 'Airport_weather']
for ds_name in ['challenge_set', 'submission_set']:
    if 'flight_id' not in datasets[ds_name].columns:
        logging.error(f"'flight_id' column is missing in {ds_name}.")
        print(f"Error: 'flight_id' column is missing in {ds_name}.")
        raise KeyError(f"'flight_id' column is missing in {ds_name}.")

# Merge flight_characteristics with challenge_set and submission_set via 'flight_id'
challenge_set = pd.merge(challenge_set, flight_characteristics, on='flight_id', how='left')
submission_set = pd.merge(submission_set, flight_characteristics, on='flight_id', how='left')
logging.info("Merged flight_characteristics with challenge_set and submission_set.")
print("Merged flight_characteristics with challenge_set and submission_set.")

# Free up memory
del flight_characteristics
gc.collect()

# Verify if 'tow' exists in challenge_set
if 'tow' in challenge_set.columns:
    # Check for missing values
    missing_tow = challenge_set['tow'].isna().sum()
    print(f"\nNumber of missing 'tow' in challenge_set: {missing_tow}")
    
    if missing_tow > 0:
        print(f"Filling {missing_tow} missing 'tow' values with median.")
        median_tow = challenge_set['tow'].median()
        challenge_set['tow'].fillna(median_tow, inplace=True)
        logging.info(f"Filled {missing_tow} missing 'tow' entries with median value: {median_tow}.")
else:
    logging.error("'tow' column is missing in challenge_set after merging flight_characteristics.")
    print("'tow' column is missing in challenge_set after merging flight_characteristics.")
    raise KeyError("'tow' column is missing in challenge_set after merging flight_characteristics.")

# Merge flight_blocks_processed data with challenge_set and submission_set via 'flight_id'
if 'flight_id' in flight_blocks_processed.columns:
    challenge_set = pd.merge(challenge_set, flight_blocks_processed, on='flight_id', how='left')
    submission_set = pd.merge(submission_set, flight_blocks_processed, on='flight_id', how='left')
    logging.info("Merged flight_blocks_processed with challenge_set and submission_set.")
    print("Merged flight_blocks_processed with challenge_set and submission_set.")
else:
    logging.error("'flight_id' column is missing in flight_blocks_processed.")
    print("Error: 'flight_id' column is missing in flight_blocks_processed.")
    raise KeyError("'flight_id' column is missing in flight_blocks_processed.")

# Free up memory
del flight_blocks_processed
gc.collect()

# Remove duplicate columns
challenge_set = resolve_duplicate_columns(challenge_set, 'challenge_set')
submission_set = resolve_duplicate_columns(submission_set, 'submission_set')

# Merge climate_data with challenge_set and submission_set via 'flight_id'
if 'flight_id' in climate_data.columns:
    challenge_set = pd.merge(challenge_set, climate_data, on='flight_id', how='left')
    submission_set = pd.merge(submission_set, climate_data, on='flight_id', how='left')
    logging.info("Merged climate_data with challenge_set and submission_set.")
    print("Merged climate_data with challenge_set and submission_set.")
else:
    logging.error("'flight_id' column is missing in climate_data.")
    print("Error: 'flight_id' column is missing in climate_data.")
    raise KeyError("'flight_id' column is missing in climate_data.")

# Free up memory
del climate_data
gc.collect()

#===========================================
# ====== Airport Data ======================
#----------------------------------------

# Merge airport_dimension on 'adep'
if 'adep' in challenge_set.columns and 'Airport_CODE' in airport_dimension.columns:
    challenge_set = pd.merge(challenge_set, airport_dimension, left_on='adep', right_on='Airport_CODE', how='left', suffixes=('', '_adep'))
    submission_set = pd.merge(submission_set, airport_dimension, left_on='adep', right_on='Airport_CODE', how='left', suffixes=('', '_adep'))
    logging.info("Merged airport_dimension with challenge_set and submission_set on 'adep'.")
    print("Merged airport_dimension with challenge_set and submission_set on 'adep'.")
else:
    logging.warning("'adep' or 'Airport_CODE' column not found in challenge_set or airport_dimension for merging.")
    print("Warning: 'adep' or 'Airport_CODE' column not found in challenge_set or airport_dimension for merging.")

# Merge airport_dimension on 'ades'
if 'ades' in challenge_set.columns and 'Airport_CODE' in airport_dimension.columns:
    challenge_set = pd.merge(challenge_set, airport_dimension, left_on='ades', right_on='Airport_CODE', how='left', suffixes=('', '_ades'))
    submission_set = pd.merge(submission_set, airport_dimension, left_on='ades', right_on='Airport_CODE', how='left', suffixes=('', '_ades'))
    logging.info("Merged airport_dimension with challenge_set and submission_set on 'ades'.")
    print("Merged airport_dimension with challenge_set and submission_set on 'ades'.")
else:
    logging.warning("'ades' or 'Airport_CODE' column not found in challenge_set or airport_dimension for merging.")
    print("Warning: 'ades' or 'Airport_CODE' column not found in challenge_set or airport_dimension for merging.")

# Free up memory
del airport_dimension
gc.collect()

# ==========================================
# ====== Merging Takeoff and Landing Speeds ==
# ==========================================

# Add a 'source' column to each dataset for later separation
challenge_set['source'] = 'challenge'
submission_set['source'] = 'submission'

# Concatenate challenge_set and submission_set into a combined dataset
combined_set = pd.concat([challenge_set, submission_set], ignore_index=True)
logging.info("Combined challenge_set and submission_set into a single dataset for imputation.")
print("Combined challenge_set and submission_set into a single dataset for imputation.")

# Free up memory
del challenge_set, submission_set
gc.collect()

# ==========================================
# ====== Merging Takeoff and Landing Speeds with Suffixes ===
# ==========================================

# Rename columns to add 'Toff_' suffix except 'flight_id' and 'adep'
combined_takeoff_speeds_renamed = combined_takeoff_speeds.rename(
    columns=lambda x: f"Toff_{x}" if x != 'flight_id' else x
)

# Merge combined_takeoff_speeds with combined_set via 'flight_id'
combined_set = pd.merge(combined_set, combined_takeoff_speeds_renamed, on='flight_id', how='left')
logging.info("Merged combined_takeoff_speeds with combined_set with 'Toff_' suffix.")
print("Merged combined_takeoff_speeds with combined_set with 'Toff_' suffix.")

# Rename columns to add 'landing_' suffix except 'flight_id' and 'ades'
all_landing_speeds_renamed = all_landing_speeds.rename(
    columns=lambda x: f"landing_{x}" if x != 'flight_id' else x
)

# Merge all_landing_speeds with combined_set via 'flight_id'
combined_set = pd.merge(combined_set, all_landing_speeds_renamed, on='flight_id', how='left')
logging.info("Merged all_landing_speeds with combined_set with 'landing_' suffix.")
print("Merged all_landing_speeds with combined_set with 'landing_' suffix.")

# Merge adb_data as needed (assuming 'aircraft_type' is the key)
if 'aircraft_type' in combined_set.columns and 'aircraft_type' in adb_data.columns:
    # Rename adb_data columns with 'ADB2_' prefix except 'aircraft_type'
    adb_data_renamed = adb_data.rename(
        columns=lambda x: f"ADB2_{x}" if x != 'aircraft_type' else x
    )
    
    combined_set = pd.merge(combined_set, adb_data_renamed, on='aircraft_type', how='left')
    logging.info("Merged adb_data with combined_set with 'ADB2_' prefix.")
    print("Merged adb_data with combined_set with 'ADB2_' prefix.")
else:
    logging.warning("'aircraft_type' column not found in combined_set or adb_data for merging.")
    print("Warning: 'aircraft_type' column not found in combined_set or adb_data for merging.")

# Free up memory
del adb_data, combined_takeoff_speeds, all_landing_speeds
gc.collect()

# Remove duplicate columns after all merges
combined_set = resolve_duplicate_columns(combined_set, 'combined_set')

# Create 'Airport_pair' in combined_set
combined_set = create_Airport_pair(combined_set, 'combined_set')

# Extract Year and Month from 'actual_offblock_time'
combined_set = extract_year_month(combined_set, 'combined_set')

# Verify that 'Month_departure' has been created
if 'Month_departure' not in combined_set.columns:
    logging.error("'Month_departure' column is missing in combined_set after extracting year and month.")
    print("Error: 'Month_departure' column is missing in combined_set after extracting year and month.")
    raise KeyError("'Month_departure' column is missing in combined_set after extracting year and month.")
else:
    print("\nSample 'Month_departure' in combined_set:")
    print(combined_set['Month_departure'].head())

# ==========================================
# ====== Feature Engineering Starts Here ===
# ==========================================

# Define target variable
target_variable = 'tow'

# ==========================================
# ====== Prepare Data for Modeling =========
# ==========================================

# Define features to use (excluding the target variable, 'flight_id', and 'source')
features_to_keep = [col for col in combined_set.columns if col not in [target_variable, 'flight_id', 'source']]

# **Remove the specified columns from the features_to_keep**
columns_to_remove = [
    'HC Number Eng', 'HC Dp/Foo Avg (g/kN)', 'HC Dp/Foo Sigma (g/kN)', 
    'HC Dp/Foo Min (g/kN)', 'HC Dp/Foo Max (g/kN)', 'HC Dp/Foo Characteristic (g/kN)', 
    'HC Dp/Foo Characteristic (% of Reg limit)', 'HC LTO Total mass (g)', 'CO EI T/O (g/kg)', 
    'CO EI C/O (g/kg)', 'CO EI App (g/kg)', 'CO EI Idle (g/kg)', 'CO Number Test', 
    'CO Number Eng', 'CO Dp/Foo Avg (g/kN)', 'CO Dp/Foo Sigma (g/kN)', 'CO Dp/Foo Min (g/kN)', 
    'CO Dp/Foo Max (g/KN)', 'CO Dp/Foo Characteristic (g/kN)', 
    'CO Dp/Foo Characteristic (% of Reg limit)', 'CO LTO Total Mass (g)', 'NOx EI T/O (g/kg)', 
    'NOx EI C/O (g/kg)', 'NOx EI App (g/kg)', 'NOx EI Idle (g/kg)', 'NOx Number Test', 
    'NOx Number Eng', 'NOx Dp/Foo Avg (g/kN)', 'NOx Dp/Foo Sigma (g/kN)', 
    'NOx Dp/Foo Min (g/kN)', 'NOx Dp/Foo Max (g/kN)', 'NOx Dp/Foo Characteristic (g/kN)', 
    'NOx Dp/Foo Characteristic (% of original standard)', 'NOx Dp/Foo Characteristic (% of CAEP/2 standard)', 
    'NOx Dp/Foo Characteristic (% of CAEP/4 standard)', 'NOx Dp/Foo Characteristic (% of CAEP/6 standard)', 
    'NOx Dp/Foo Characteristic (% of CAEP/8 standard)', 'NOx LTO Total mass (g)', 'SN T/O', 
    'SN C/O', 'SN App', 'SN Idle', 'SN Number Test', 'SN Number Eng', 'SN Max', 'SN Sigma', 
    'SN Range Min', 'SN Range Max', 'SN Characteristic', 'SN Characteristic (% of Reg limit)', 
    'Fuel H/C Ratio Min', 'Fuel H/C Ratio Max', 'Fuel Arom Min (%)', 'Fuel Arom Max (%)', 
    'Fuel Flow T/O (kg/sec)', 'Fuel Flow C/O (kg/sec)', 'Fuel Flow App (kg/sec)', 
    'Fuel Flow Idle (kg/sec)', 'Fuel LTO Cycle (kg)', 'Loads Power Extraction (kW)', 
    'Combustor Description', 'Compliance with fuel venting requirements', 'Current Engine Status', 
    'Current Engine Status Date', 'Data Status', 'Data Superseded', 'Data corr as Annex 16', 
    'Eng Type', 'Engine_Name', 'Final Test Date', 'Fuel Spec', 'Initial Test Date', 
    'Loads Power Extraction @Power', 'Loads Stage Bleed @Power', 'Loads Stage Bleed CF (%)', 
    'Manufacturer_x', 'Manufacturer_y', 'RECAT-EU', 'Remark 1', 'Remark 2', 
    'Superseded by UID No', 'Test Engine Status', 'Test Location', 'Test Organisation', 
    'Type_of_Aircraft', 'UID No','fuel_lto_cycle_kg_adsb',

]

# Remove specified columns from features_to_keep
features_to_keep = [col for col in features_to_keep if col not in columns_to_remove]

# Ensure all features are present in combined_set
missing_features = set(features_to_keep) - set(combined_set.columns)
if missing_features:
    logging.error(f"Missing features in combined_set: {missing_features}")
    print(f"Error: Missing features in combined_set: {missing_features}")
    # Handle missing features by adding them with default values
    for feature in missing_features:
        combined_set[feature] = -999  # Using the same fill value
        logging.warning(f"Added missing feature '{feature}' to combined_set with default value -999.")
        print(f"Added missing feature '{feature}' to combined_set with default value -999.")

# ==========================================
# ====== Identify Columns with Missing Values
# ==========================================

# **Only impute specified columns**
columns_to_impute = [
    'total_flight_duration_sec_adsb',
    'total_distance_flown_nm_adsb_adsb',
    'number_of_position_reports_adsb',
    'average_groundspeed_knots_adsb',
    'max_groundspeed_knots_adsb',
    'average_vertical_rate_ft_per_min_adsb',
    'max_vertical_rate_ft_per_min_adsb',
    'num_climb_descent_cycles_adsb',
    'average_track_change_deg_adsb',
    'average_temperature_k_adsb',
    'average_wind_speed_m_per_s_adsb',
    'average_specific_humidity_adsb',
    'average_altitude_change_rate_ft_per_sec_adsb',
    'great_circle_distance_nm_adsb',
    'route_efficiency_adsb',
    'climb_average_speed_knots_adsb',
    'climb_average_vertical_rate_ft_per_min_adsb',
    'climb_acceleration_knots_per_sec_adsb',
    'climb_distance_nm_adsb',
    'climb_max_speed_knots_adsb',
    'climb_max_altitude_ft_adsb',
    'cruise_average_speed_knots_adsb',
    'cruise_average_vertical_rate_ft_per_min_adsb',
    'cruise_acceleration_knots_per_sec_adsb',
    'cruise_time_sec_adsb',
    'cruise_distance_nm_adsb',
    'cruise_average_altitude_ft_adsb',
    'cruise_max_speed_knots_adsb',
    'cruise_max_altitude_ft_adsb',
    'descent_average_speed_knots_adsb',
    'descent_average_vertical_rate_ft_per_min_adsb',
    'descent_acceleration_knots_per_sec_adsb',
    'descent_time_sec_adsb',
    'descent_distance_nm_adsb',
    'descent_max_speed_knots_adsb',
    'descent_max_altitude_ft_adsb',
    'landing_average_speed_knots_adsb',
    'landing_average_vertical_rate_ft_per_min_adsb',
    'landing_acceleration_knots_per_sec_adsb',
    'landing_time_sec_adsb',
    'landing_distance_nm_adsb',
    'landing_max_speed_knots_adsb',
    'landing_max_altitude_ft_adsb',
    'v2_speed_knots_adsb',
    'landing_speed_knots_adsb',
    'total_fuel_consumption_kg_adsbtakeoff_avg_direction_avg',
    'takeoff_avg_temperature_avg',
    'takeoff_avg_vertical_rate_avg',
    'takeoff_avg_altitude_avg',
    'takeoff_avg_speed_avg',
    'landing_avg_direction_avg',
    'landing_avg_temperature_avg',
    'landing_avg_vertical_rate_avg',
    'landing_avg_altitude_avg',
    'landing_avg_speed_avg',
    'adep_temperature_kelvin',
    'adep_dew_point_kelvin',
    'adep_relative_humidity_percentToff_groundspeed',
    'Toff_direction',
    'Toff_wind_speed',
    'Toff_wind_direction',
    'Toff_speed',
    'landing_groundspeed',
    'landing_direction',
    'landing_wind_speed',
    'landing_wind_direction',
    'landing_speed'
]

# Verify that these columns exist in the DataFrame
existing_columns_to_impute = [col for col in columns_to_impute if col in combined_set.columns]
missing_columns = set(columns_to_impute) - set(existing_columns_to_impute)
if missing_columns:
    print(f"Warning: The following columns to impute are missing in combined_set: {missing_columns}")
    logging.warning(f"The following columns to impute are missing in combined_set: {missing_columns}")
columns_to_impute = existing_columns_to_impute  # Update the list to include only existing columns

# **Filter out columns without missing values**
columns_to_impute = [col for col in columns_to_impute if combined_set[col].isna().any()]

print(f"\nColumns to impute using ML-based models: {columns_to_impute}")
logging.info(f"Columns to impute using ML-based models: {columns_to_impute}")

# ==========================================
# ====== ML-Based Imputation ============
# ==========================================

# Define imputation model save path
imputation_model_path = os.path.join(model_dir, 'imputation_models')
os.makedirs(imputation_model_path, exist_ok=True)

# Perform ML-Based Imputation using AutoGluon with GPU
combined_set, _, imputation_metrics = impute_missing_values(
    train_df=combined_set,
    submission_df=None,  # No separate submission set during combined imputation
    target_columns=columns_to_impute,
    predictor_save_path=imputation_model_path,
    time_limit=60  # Adjust as needed
)

# ==========================================
# ====== Save Imputation Metrics ===========
# ==========================================

# Save imputation metrics to CSV
imputation_metrics_path = os.path.join(model_dir, 'imputation_metrics_AG_1503.csv')
imputation_metrics.to_csv(imputation_metrics_path, index=False)
print(f"\nImputation metrics saved to '{imputation_metrics_path}'")
logging.info(f"Imputation metrics saved to '{imputation_metrics_path}'")

# ==========================================
# ====== Verify Imputation ==================
# ==========================================

print("\nVerifying imputation results:")
logging.info("Verifying imputation results.")

for index, row in imputation_metrics.iterrows():
    col = row['column']
    rmse = row['RMSE']
    mae = row['MAE']
    print(f"Imputation for '{col}': RMSE = {rmse:.4f}, MAE = {mae:.4f}")
    logging.info(f"Imputation for '{col}': RMSE = {rmse:.4f}, MAE = {mae:.4f}")

# ==========================================
# ====== Split Combined Set Back ===========
# ==========================================

# Separate the combined_set back into challenge_set and submission_set based on 'source'
challenge_set_filtered = combined_set[combined_set['source'] == 'challenge'].drop(columns=['source'])
submission_set_filtered = combined_set[combined_set['source'] == 'submission'].drop(columns=['source'])

logging.info("Separated the combined dataset back into challenge_set and submission_set after imputation.")
print("Separated the combined dataset back into challenge_set and submission_set after imputation.")
# ==========================================
# ====== Remove Duplicate flight_id and Row-based Duplicates =========
# ==========================================

def remove_duplicates(df, dataset_name):
    """
    Remove duplicate rows based on 'flight_id' and entire row duplicates.
    
    Parameters:
    - df: pandas DataFrame
    - dataset_name: string indicating the dataset name for logging
    """
    initial_count = df.shape[0]
    
    # Remove duplicates based on 'flight_id'
    df.drop_duplicates(subset=['flight_id'], inplace=True)
    after_flight_id_drop = df.shape[0]
    duplicates_removed_flight_id = initial_count - after_flight_id_drop
    logging.info(f"Removed {duplicates_removed_flight_id} duplicate rows based on 'flight_id' from {dataset_name}.")
    print(f"Removed {duplicates_removed_flight_id} duplicate rows based on 'flight_id' from {dataset_name}.")
    
    # Remove entirely duplicate rows
    df.drop_duplicates(inplace=True)
    final_count = df.shape[0]
    duplicates_removed_row = after_flight_id_drop - final_count
    if duplicates_removed_row > 0:
        logging.info(f"Removed {duplicates_removed_row} entirely duplicate rows from {dataset_name}.")
        print(f"Removed {duplicates_removed_row} entirely duplicate rows from {dataset_name}.")
    else:
        logging.info(f"No entirely duplicate rows found in {dataset_name}.")
        print(f"No entirely duplicate rows found in {dataset_name}.")

# Remove duplicates from challenge_set_filtered
remove_duplicates(challenge_set_filtered, 'challenge_set_filtered')

# Remove duplicates from submission_set_filtered
remove_duplicates(submission_set_filtered, 'submission_set_filtered')
# ==========================================
# ====== Save Imputed and Cleaned Datasets ==
# ==========================================

imputed_dataset_path = os.path.join(model_dir, 'imputed_challenge_set.csv')
challenge_set_filtered.to_csv(imputed_dataset_path, index=False)
print(f"\nImputed and cleaned training dataset saved to '{imputed_dataset_path}'")
logging.info(f"Imputed and cleaned training dataset saved to '{imputed_dataset_path}'")

# Similarly, save the imputed and cleaned submission dataset
imputed_submission_path = os.path.join(model_dir, 'imputed_submission_set.csv')
submission_set_filtered.to_csv(imputed_submission_path, index=False)
print(f"Imputed and cleaned submission dataset saved to '{imputed_submission_path}'")
logging.info(f"Imputed and cleaned submission dataset saved to '{imputed_submission_path}'")



# ==========================================
# ====== Save Final Imputed Datasets =========
# ==========================================

# Note: You have already saved 'imputed_challenge_set.csv' and 'imputed_submission_set.csv' above.
# If you need to perform any additional saving or formatting, you can add it here.

# ==========================================
# ====== Process Completed ==================
# ==========================================

print("Data processing and imputation completed. Imputed datasets are saved.")
logging.info("Data processing and imputation completed.")
