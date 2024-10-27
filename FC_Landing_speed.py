import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import gc  # Garbage collector
import traceback
from tqdm import tqdm

# Define the function to get the data path based on the date
def get_data_path(date):
    """Returns the data path based on the given date."""
    return r'F:\Par_Files'

# Define the overall date range
start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 12, 31)

# Generate a list of dates
date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

# Load aircraft data from challenge and submission sets
def load_aircraft_type_map():
    challenge_file = r"F:\Project_PRC_Eurocontrol\NEW\Input_data\BASE_DATA\challenge_set.csv"
    submission_file = r"F:\Project_PRC_Eurocontrol\NEW\Input_data\BASE_DATA\final_submission_set.csv"
    try:
        challenge_df = pd.read_csv(challenge_file, usecols=['flight_id', 'aircraft_type'])
        submission_df = pd.read_csv(submission_file, usecols=['flight_id', 'aircraft_type'])
        return pd.concat([challenge_df, submission_df]).set_index('flight_id')['aircraft_type'].to_dict()
    except Exception as e:
        #print(f"Error loading challenge or submission files: {e}")
        return {}

# Define the function to calculate wind-corrected Indicated Airspeed (IAS)
def calculate_wind_corrected_ias(groundspeed, windspeed, wind_direction, track):
    wind_direction_rad = np.radians(wind_direction)
    track_rad = np.radians(track)
    headwind_component = windspeed * np.cos(wind_direction_rad - track_rad)
    wind_corrected_ias = groundspeed - headwind_component
    return wind_corrected_ias

# Define the landing VAT IAS lookup table
landing_vat_lookup = {
    'A20N': 135, 'A21N': 140, 'A310': 139, 'A319': 130, 'A320': 137,
    'A321': 141, 'A332': 140, 'A333': 140, 'A343': 150, 'A359': 140,
    'AT76': 120, 'B38M': 145, 'B39M': 150, 'B737': 137, 'B738': 147,
    'B739': 150, 'B752': 130, 'B763': 140, 'B772': 140, 'B773': 149,
    'B77W': 149, 'B788': 140, 'B789': 140, 'BCS1': 140, 'BCS3': 135,
    'C56X': 117, 'CRJ9': 135, 'E190': 131, 'E195': 135, 'E290': 135
}

# Process landing speed based on VAT IAS criteria
def process_landing_speed(flight_df, aircraft_type):
    vat_lookup = landing_vat_lookup
    vat_speed_ias = vat_lookup.get(aircraft_type, None)
    if vat_speed_ias is None:
    
        return None

    # Filter for landing conditions: 0 < altitude <= 200 and vertical_rate < 0
    landing_condition = (flight_df['altitude'] > 0) & (flight_df['altitude'] <= 200) & (flight_df['vertical_rate'] < 0)
    landing_df = flight_df[landing_condition].copy()

    if landing_df.empty:
        return pd.DataFrame(columns=['flight_id', 'groundspeed', 'direction', 'wind_speed', 'wind_direction', 'speed'])

    # Sort by timestamp to identify the moment just before touchdown
    landing_df = landing_df.sort_values('timestamp')

    # Resample to 1-second intervals
    landing_df = landing_df.set_index('timestamp').resample('1S').mean().dropna().reset_index()

    # Calculate wind-corrected IAS
    landing_df['wind_corrected_ias'] = landing_df.apply(
        lambda row: calculate_wind_corrected_ias(
            row['groundspeed'], row['wind_speed'], row['wind_direction'], row['track']
        ), axis=1
    )

    # Apply VAT speed bounds (0.5 to 1.3 times VAT IAS)
    lower_bound = 0.5 * vat_speed_ias
    upper_bound = 1.3 * vat_speed_ias
    within_vat_bounds = (landing_df['wind_corrected_ias'] >= lower_bound) & (landing_df['wind_corrected_ias'] <= upper_bound)

    valid_landing_df = landing_df[within_vat_bounds].copy()

    # If no valid results, return empty DataFrame
    if valid_landing_df.empty:
        return pd.DataFrame(columns=['flight_id', 'groundspeed', 'direction', 'wind_speed', 'wind_direction', 'speed'])

    # Select the last valid entry per flight (just before touchdown)
    result_df = valid_landing_df.iloc[[-1]].copy()

    # Select required columns and rename
    result_df = result_df[['flight_id', 'groundspeed', 'track', 'wind_speed', 'wind_direction', 'wind_corrected_ias']]
    result_df.rename(columns={'track': 'direction', 'wind_corrected_ias': 'speed'}, inplace=True)

    return result_df

# Main function to process data, clean, calculate, and save the results
def main():
    # Load the aircraft type mapping
    aircraft_type_map = load_aircraft_type_map()

    # Loop through all days in the date range
    for D1 in tqdm(date_list, desc="Processing Dates"):
        #print(f"\nProcessing files for {D1.strftime('%Y-%m-%d')}")

        # Get the data path for the specific date
        data_path = get_data_path(D1)
        if data_path is None:
            #print(f"No data path for date {D1.strftime('%Y-%m-%d')}. Skipping.")
            continue

        # Construct the file path
        file = os.path.join(data_path, f"{D1.strftime('%Y-%m-%d')}.parquet")
        if not os.path.exists(file):
            #print(f"File for {D1.strftime('%Y-%m-%d')} is missing. Skipping.")
            continue

        # Read the Parquet file
        columns_to_read = [
            'flight_id', 'icao24', 'longitude', 'latitude', 'altitude', 'timestamp',
            'groundspeed', 'track', 'vertical_rate', 'u_component_of_wind', 'v_component_of_wind'
        ]

        try:
            df = pd.read_parquet(file, columns=columns_to_read)
        except Exception as e:
            #print(f"Error reading file: {e}")
            continue

        # Ensure timestamp is datetime and timezone-aware (UTC)
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)

        # Remove rows with invalid timestamps
        df = df.dropna(subset=['timestamp'])

        # Convert wind components to wind speed and direction
        df['wind_speed'] = np.sqrt(df['u_component_of_wind']**2 + df['v_component_of_wind']**2)
        df['wind_direction'] = np.degrees(np.arctan2(df['v_component_of_wind'], df['u_component_of_wind']))
        df['wind_direction'] = (df['wind_direction'] + 360) % 360  # Normalize to [0, 360)

        # Initialize a list to store final results
        final_results = []

        # Process each flight
        for flight_id, flight_data in df.groupby('flight_id'):
            # Check if flight_id is in the imported aircraft type map
            if flight_id not in aircraft_type_map:
                #print(f"Flight ID {flight_id} not found in aircraft type map. Skipping flight.")
                continue

            aircraft_type = aircraft_type_map.get(flight_id, None)

            # Process the flight data and get landing speed
            landing_speed_df = process_landing_speed(flight_data, aircraft_type)
            if landing_speed_df is not None and not landing_speed_df.empty:
                landing_speed_df['flight_id'] = flight_id  # Ensure flight_id is in the results
                final_results.append(landing_speed_df)

        # After processing all flights, save the result for the day
        if final_results:
            final_result_df = pd.concat(final_results, ignore_index=True)
            output_file = os.path.join(r"F:\Project_PRC_Eurocontrol\NEW\Landing_Speeds", f"landing_speeds_{D1.strftime('%Y-%m-%d')}.csv")
            try:
                final_result_df.to_csv(output_file, index=False)
                print(f"Landing speeds saved to {output_file}")
            except Exception as e:
                print(f"Error saving landing speeds to CSV: {e}")
        else:
            print("No valid landing speeds; skipping CSV save.")

        # Clean up memory
        gc.collect()

    print("\nProcessing complete.")

# Run the main function
if __name__ == "__main__":
    main()
