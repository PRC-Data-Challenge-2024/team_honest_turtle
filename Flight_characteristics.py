import numpy as np
import pandas as pd

import os
from datetime import datetime, timedelta
import gc  # Garbage collector
import traceback
from tqdm import tqdm
from traffic.core import Flight, Traffic  # Import Flight and Traffic classes

# Define the function to get the data path based on the date
def get_data_path(date):
    """Returns the data path based on the given date.
    All files are located in 'F:\\Par_Files' as per specification.
    """
    return r'F:\Par_Files'

# Define the overall date range
start_date = datetime(2022, 1, 1)  # Adjusted to start earlier for meaningful processing
end_date = datetime(2022, 12, 31)  # Include 12/31/2022

# Generate a list of dates
date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

# Load Arrival Times and Departure Times from Submission and Challenge Sets
submission_file = r"F:\Project_PRC_Eurocontrol\NEW\Input_data\BASE_DATA\final_submission_set.csv"
challenge_file = r"F:\Project_PRC_Eurocontrol\NEW\Input_data\BASE_DATA\challenge_set.csv"

try:
    # Load required columns including 'actual_offblock_time' and 'taxiout_time'
    submission_df = pd.read_csv(submission_file, usecols=['flight_id', 'arrival_time', 'actual_offblock_time', 'taxiout_time'])
    challenge_df = pd.read_csv(challenge_file, usecols=['flight_id', 'arrival_time', 'actual_offblock_time', 'taxiout_time'])
    combined_df = pd.concat([submission_df, challenge_df], ignore_index=True)

    # Ensure all times are in datetime format and timezone-aware (UTC)
    combined_df['arrival_time'] = pd.to_datetime(combined_df['arrival_time'], errors='coerce', utc=True)
    combined_df['actual_offblock_time'] = pd.to_datetime(combined_df['actual_offblock_time'], errors='coerce', utc=True)

    # Calculate the departure_time_clip as actual_offblock_time + taxiout_time (in minutes)
    combined_df['taxiout_time'] = pd.to_timedelta(combined_df['taxiout_time'], unit='m')
    combined_df['departure_time_clip'] = combined_df['actual_offblock_time'] + combined_df['taxiout_time']

    # Create mappings for flight_id to departure_time_clip and arrival_time
    flight_departure_map = combined_df.set_index('flight_id')['departure_time_clip'].to_dict()
    flight_arrival_map = combined_df.set_index('flight_id')['arrival_time'].to_dict()

    print(f"Loaded departure and arrival times for {len(flight_departure_map)} flights.")
except Exception as e:
    print(f"Error reading time files: {e}")
    traceback.print_exc()
    flight_departure_map = {}
    flight_arrival_map = {}

# Define the Haversine formula to calculate distance between two lat/lon points
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Vectorized calculation of the great-circle distance between two points on the Earth's surface.
    Returns distance in nautical miles.
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    earth_radius_nm = 3440.065  # Earth's radius in nautical miles
    return earth_radius_nm * c

# Define data processing functions

def identify_phases(flight_df):
    """
    Identifies flight phases based on altitude and vertical rate.
    Phases:
    - Climb: <= 5,000 ft with positive vertical rate
    - Cruise: > 10,000 ft
    - Descent: <= 5,000 ft with negative vertical rate
    """
    # Initialize phase column with 'UNKNOWN'
    flight_df['phase'] = 'UNKNOWN'

    # Vectorized conditions
    climb_condition = (flight_df['altitude'] <= 5000) & (flight_df['vertical_rate'] > 0)
    cruise_condition = (flight_df['altitude'] > 10000)
    descent_condition = (flight_df['altitude'] <= 5000) & (flight_df['vertical_rate'] < 0)

    # Assign phases
    flight_df.loc[climb_condition, 'phase'] = 'Climb'
    flight_df.loc[cruise_condition, 'phase'] = 'Cruise'
    flight_df.loc[descent_condition, 'phase'] = 'Descent'

    return flight_df

def clean_data(flight_df):
    """
    Cleans the flight data by removing impossible changes and interpolating missing values.
    Additional Cleaning:
    1. Remove data points with altitude <0 or speed <0.
    2. Remove data points with altitude >55,000 ft.
    3. Remove data points where speed >300 knots and altitude <200 ft.
    4. Remove data points with invalid latitude or longitude.
    5. Remove data points with groundspeed >700 knots.
    6. Remove duplicate timestamps within the same flight.
    """
    # Ensure the data is sorted by timestamp
    flight_df = flight_df.sort_values('timestamp')

    # Apply vectorized conditions for data cleaning
    condition = (
        (flight_df['altitude'] >= 0) &
        (flight_df['groundspeed'] >= 0) &
        (flight_df['altitude'] <= 55000) &
        ~((flight_df['groundspeed'] > 300) & (flight_df['altitude'] < 200)) &
        (flight_df['latitude'].between(-90, 90)) &
        (flight_df['longitude'].between(-180, 180)) &
        (flight_df['groundspeed'] <= 700)
    )
    flight_df = flight_df[condition]

    # Remove duplicate timestamps within the same flight
    flight_df = flight_df.drop_duplicates(subset=['timestamp'])

    # Rule 2: Remove outliers using median filtering
    window_size = 25
    half_window = window_size // 2

    altitudes = flight_df['altitude'].values
    filtered_altitudes = altitudes.copy()

    # Use a vectorized approach with rolling median
    alt_series = pd.Series(altitudes)
    rolling_median = alt_series.rolling(window=window_size, center=True).median()
    differences = np.abs(altitudes - rolling_median)
    filtered_altitudes[differences > 150] = np.nan
    flight_df['altitude'] = filtered_altitudes

    # Interpolate missing values
    flight_df['altitude'] = flight_df['altitude'].interpolate()
    flight_df['groundspeed'] = flight_df['groundspeed'].interpolate()

    return flight_df

def compute_track_unwrapped(flight_df):
    """
    Computes the unwrapped track angle from the track column.
    """
    # Handle NaN values in 'track'
    flight_df['track'] = flight_df['track'].fillna(0)

    track_rad = np.radians(flight_df['track'].values)
    unwrapped_rad = np.unwrap(track_rad)
    flight_df['track_unwrapped'] = np.degrees(unwrapped_rad)

    return flight_df

def calculate_acceleration(flight_df):
    """
    Calculates acceleration based on change in speed and time.
    Returns the average acceleration.
    """
    # Ensure data is sorted
    flight_df = flight_df.sort_values('timestamp').reset_index(drop=True)

    speed_diff = flight_df['groundspeed'].diff().values
    time_diff_sec = flight_df['timestamp'].diff().dt.total_seconds().values
    acceleration = np.divide(speed_diff, time_diff_sec, out=np.zeros_like(speed_diff), where=time_diff_sec != 0)

    # Handle infinite and NaN values
    acceleration = np.nan_to_num(acceleration, nan=0.0, posinf=0.0, neginf=0.0)
    average_acceleration = np.mean(acceleration)

    return average_acceleration

def calculate_v2_speed(flight_df, departure_time_clip):
    """
    Calculates V2 speed in knots.
    V2 Speed: Average speed within 50 seconds after departure_time_clip with positive speed and positive vertical rate.
    """
    v2_window_sec = 50
    if pd.isna(departure_time_clip):
        return np.nan

    v2_end_time = departure_time_clip + timedelta(seconds=v2_window_sec)
    v2_phase = flight_df[
        (flight_df['timestamp'] >= departure_time_clip) &
        (flight_df['timestamp'] <= v2_end_time) &
        (flight_df['groundspeed'] > 0) &
        (flight_df['vertical_rate'] > 0)
    ]

    if not v2_phase.empty:
        v2_speed = v2_phase['groundspeed'].mean()
    else:
        v2_speed = np.nan

    return v2_speed

def calculate_landing_speed(flight_df, arrival_time):
    """
    Calculates Landing speed in knots.
    Landing Speed: Average speed in the last 50 seconds before arrival with positive speed and negative vertical rate.
    """
    landing_window_sec = 50
    if pd.isna(arrival_time):
        return np.nan

    landing_start_time = arrival_time - timedelta(seconds=landing_window_sec)
    landing_phase = flight_df[
        (flight_df['timestamp'] >= landing_start_time) &
        (flight_df['timestamp'] <= arrival_time) &
        (flight_df['groundspeed'] > 0) &
        (flight_df['vertical_rate'] < 0)
    ]

    if not landing_phase.empty:
        landing_speed = landing_phase['groundspeed'].mean()
    else:
        landing_speed = np.nan

    return landing_speed

def clip_speed_values(flight_df):
    """
    Clips Indicated Airspeed (IAS) values between 50 knots and 300 knots.
    """
    flight_df['IAS'] = flight_df['groundspeed']  # Assuming IAS = groundspeed if no wind data
    # Clip IAS between 50 and 300 knots
    flight_df['IAS_clipped'] = flight_df['IAS'].clip(lower=50, upper=300)
    return flight_df

def compute_rotation_degrees(flight_df):
    """
    Computes the total rotation degrees based on changes in track_unwrapped.
    """
    track_changes = np.abs(np.diff(flight_df['track_unwrapped'].values))
    track_changes = np.where(track_changes > 180, 360 - track_changes, track_changes)
    total_rotation = np.nansum(track_changes)
    return total_rotation

def calculate_track_changes(flight_df):
    """
    Calculates average track change in degrees.
    """
    track_changes = np.abs(np.diff(flight_df['track_unwrapped'].values))
    track_changes = np.where(track_changes > 180, 360 - track_changes, track_changes)
    average_track_change = np.nanmean(track_changes)
    return average_track_change

def calculate_num_climb_descent_cycles(flight_df):
    """
    Calculates the number of climb/descent cycles based on vertical rate sign changes.
    """
    vertical_rates = flight_df['vertical_rate'].values
    signs = np.sign(vertical_rates)
    sign_changes = np.sum(signs[1:] != signs[:-1])
    return sign_changes

def calculate_average_altitude_change_rate(flight_df):
    """
    Calculates the average altitude change rate in ft/sec.
    """
    altitude_diff = np.abs(np.diff(flight_df['altitude'].values))
    time_diff_sec = np.diff(flight_df['timestamp'].astype('int64') // 1e9)
    altitude_change_rate = np.divide(altitude_diff, time_diff_sec, out=np.zeros_like(altitude_diff), where=time_diff_sec != 0)
    average_altitude_change_rate = np.nanmean(altitude_change_rate)
    return average_altitude_change_rate

def calculate_specific_metrics(flight_df):
    """
    Calculates specific metrics like average temperature, wind speed, and specific humidity.
    """
    average_temperature_K = flight_df['temperature'].mean() if 'temperature' in flight_df.columns else np.nan

    if 'u_component_of_wind' in flight_df.columns and 'v_component_of_wind' in flight_df.columns:
        wind_speed = np.sqrt(flight_df['u_component_of_wind']**2 + flight_df['v_component_of_wind']**2)
        average_wind_speed_m_per_s = wind_speed.mean()
    else:
        average_wind_speed_m_per_s = np.nan

    average_specific_humidity = flight_df['specific_humidity'].mean() if 'specific_humidity' in flight_df.columns else np.nan

    return average_temperature_K, average_wind_speed_m_per_s, average_specific_humidity

def compute_route_efficiency(flight_df):
    """
    Computes route efficiency as the ratio of great-circle distance to total distance flown.
    """
    start_lat = flight_df['latitude'].iloc[0]
    start_lon = flight_df['longitude'].iloc[0]
    end_lat = flight_df['latitude'].iloc[-1]
    end_lon = flight_df['longitude'].iloc[-1]
    great_circle_distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    total_distance_flown_nm_adsb = flight_df['distance_nm'].sum()
    if total_distance_flown_nm_adsb > 0:
        route_efficiency = great_circle_distance / total_distance_flown_nm_adsb
    else:
        route_efficiency = np.nan
    return great_circle_distance, route_efficiency

def process_flight_data(flight_id, flight_df):
    """
    Processes individual flight data: cleaning, phase identification, metric computation,
    and returns a dict with all phase and overall metrics.
    """
    try:
        # Clean data with additional clipping rules
        flight_df = clean_data(flight_df)

        # Identify flight phases
        flight_df = identify_phases(flight_df)

        # Sort data by timestamp
        flight_df = flight_df.sort_values('timestamp').reset_index(drop=True)

        # Compute track_unwrapped if not present
        if 'track_unwrapped' not in flight_df.columns or flight_df['track_unwrapped'].isna().all():
            flight_df = compute_track_unwrapped(flight_df)

        # Calculate distances between consecutive points
        flight_df['prev_latitude'] = flight_df['latitude'].shift()
        flight_df['prev_longitude'] = flight_df['longitude'].shift()
        flight_df['distance_nm'] = haversine_distance(
            flight_df['prev_latitude'],
            flight_df['prev_longitude'],
            flight_df['latitude'],
            flight_df['longitude']
        )
        flight_df['distance_nm'] = flight_df['distance_nm'].fillna(0)

        # Clip IAS values between 50 and 300 knots
        flight_df = clip_speed_values(flight_df)

        # Initialize a dictionary to store all metrics for this flight
        flight_metrics = {'flight_id': flight_id}

        # Overall Flight Metrics
        flight_duration = (flight_df['timestamp'].iloc[-1] - flight_df['timestamp'].iloc[0]).total_seconds()
        flight_metrics['total_flight_duration_sec'] = flight_duration
        flight_metrics['total_distance_flown_nm_adsb'] = flight_df['distance_nm'].sum()
        flight_metrics['number_of_position_reports'] = len(flight_df)
        flight_metrics['average_groundspeed_knots'] = flight_df['groundspeed'].mean()
        flight_metrics['max_groundspeed_knots'] = flight_df['groundspeed'].max()
        flight_metrics['average_vertical_rate_ft_per_min'] = flight_df['vertical_rate'].mean()
        flight_metrics['max_vertical_rate_ft_per_min'] = flight_df['vertical_rate'].max()

        # Number of Climb/Descent Cycles
        vertical_sign_changes = calculate_num_climb_descent_cycles(flight_df)
        flight_metrics['num_climb_descent_cycles'] = vertical_sign_changes

        # Average Track Change
        average_track_change = calculate_track_changes(flight_df)
        flight_metrics['average_track_change_deg'] = average_track_change

        # Average Temperature, Wind Speed, Specific Humidity
        average_temperature_K, average_wind_speed_m_per_s, average_specific_humidity = calculate_specific_metrics(flight_df)
        flight_metrics['average_temperature_K'] = average_temperature_K
        flight_metrics['average_wind_speed_m_per_s'] = average_wind_speed_m_per_s
        flight_metrics['average_specific_humidity'] = average_specific_humidity

        # Average Altitude Change Rate
        average_altitude_change_rate = calculate_average_altitude_change_rate(flight_df)
        flight_metrics['average_altitude_change_rate_ft_per_sec'] = average_altitude_change_rate

        # Route Efficiency
        great_circle_distance_nm, route_efficiency = compute_route_efficiency(flight_df)
        flight_metrics['great_circle_distance_nm'] = great_circle_distance_nm
        flight_metrics['route_efficiency'] = route_efficiency

        # Phase-wise Metrics
        phase_groups = flight_df.groupby('phase')

        for phase, group in phase_groups:
            if phase not in ['Climb', 'Cruise', 'Descent']:
                continue  # Skip unknown or other phases

            # Compute metrics for each phase
            flight_metrics[f"{phase}_average_speed_knots"] = group['groundspeed'].mean()
            flight_metrics[f"{phase}_average_vertical_change_rate_ft_per_min"] = group['vertical_rate'].mean()
            flight_metrics[f"{phase}_acceleration_knots_per_sec"] = calculate_acceleration(group)
            phase_duration = (group['timestamp'].iloc[-1] - group['timestamp'].iloc[0]).total_seconds()
            flight_metrics[f"{phase}_time_sec"] = phase_duration
            flight_metrics[f"{phase}_distance_nm"] = group['distance_nm'].sum()

            if phase == 'Cruise':
                flight_metrics[f"{phase}_average_altitude_ft"] = group['altitude'].mean()

            # Additional Metrics: Max Speed and Max Altitude
            flight_metrics[f"{phase}_max_speed_knots"] = group['groundspeed'].max()
            flight_metrics[f"{phase}_max_altitude_ft"] = group['altitude'].max()

        # V2 Speed
        departure_time_clip = flight_departure_map.get(flight_id, None)
        flight_metrics['V2_speed_knots'] = calculate_v2_speed(flight_df, departure_time_clip)

        # Landing Speed
        arrival_time = flight_arrival_map.get(flight_id, None)
        flight_metrics['Landing_speed_knots'] = calculate_landing_speed(flight_df, arrival_time)

        return flight_metrics
    except Exception as e:
        print(f"Error processing flight {flight_id}: {e}")
        traceback.print_exc()
        return None

def main():
    # Initialize a set to track processed flight_ids to avoid duplication
    processed_flight_ids = set()

    # Start of the main processing loop with progress bar
    for i in tqdm(range(len(date_list)), desc="Processing Dates"):
        D1 = date_list[i]
        D2 = date_list[i + 1] if i + 1 < len(date_list) else D1  # Process D1 and D2 as consecutive days

        print(f"\nProcessing files for {D1.strftime('%Y-%m-%d')} and {D2.strftime('%Y-%m-%d')}")

        # Get the data paths for D1 and D2
        data_path1 = get_data_path(D1)
        data_path2 = get_data_path(D2)

        # Check if data paths are valid
        if data_path1 is None or data_path2 is None:
            print(f"No data path for dates {D1.strftime('%Y-%m-%d')} or {D2.strftime('%Y-%m-%d')}. Skipping.")
            continue

        # Construct file paths
        file1 = os.path.join(data_path1, f"{D1.strftime('%Y-%m-%d')}.parquet")
        file2 = os.path.join(data_path2, f"{D2.strftime('%Y-%m-%d')}.parquet")

        # Check if files exist
        if not os.path.exists(file1) or not os.path.exists(file2):
            print(f"One or both files for {D1.strftime('%Y-%m-%d')} and {D2.strftime('%Y-%m-%d')} are missing. Skipping.")
            continue

        # Read the Parquet files
        columns_to_read = [
            'flight_id', 'icao24', 'longitude', 'latitude', 'altitude', 'timestamp',
            'groundspeed', 'track', 'vertical_rate',
            'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity'
        ]

        try:
            df1 = pd.read_parquet(file1, columns=columns_to_read)
            df2 = pd.read_parquet(file2, columns=columns_to_read)
        except Exception as e:
            print(f"Error reading files: {e}")
            traceback.print_exc()
            continue

        # Combine data from the two days
        df = pd.concat([df1, df2], ignore_index=True)
        del df1, df2  # Free up memory
        gc.collect()

        # Ensure timestamp is datetime and timezone-aware (UTC)
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)

        # Remove rows with invalid timestamps
        df = df.dropna(subset=['timestamp'])

        # Reset index to ensure 'timestamp' is a column
        df.reset_index(inplace=True, drop=True)

        # Create a Traffic object from the DataFrame and apply direct operations
        try:
            t = Traffic(df)

            # Apply filtering to smooth vertical glitches
            t_filtered = t.filter()

            # Resample at 1-second intervals
            t_resampled = t_filtered.resample('1s')

            # Execute all operations
            t_evaluated = t_resampled.eval()
        except Exception as e:
            print(f"Error processing Traffic object for dates {D1.strftime('%Y-%m-%d')} and {D2.strftime('%Y-%m-%d')}: {e}")
            traceback.print_exc()
            continue

        # Prepare a list to collect feature data
        feature_list = []

        # Process each flight
        for flight in tqdm(t_evaluated, desc=f"Computing Metrics for {D1.strftime('%Y-%m-%d')}"):
            flight_id = flight.flight_id

            # Skip already processed flight_ids to prevent duplication
            if flight_id in processed_flight_ids:
                continue

            # Convert flight data to DataFrame
            flight_df = flight.data.reset_index()
            flight_df['flight_id'] = flight_id  # Ensure 'flight_id' is in the DataFrame

            # Process the flight data and get metrics
            flight_metrics = process_flight_data(flight_id, flight_df)
            if flight_metrics:
                # Append the metrics to the feature list
                feature_list.append(flight_metrics)
                # Add the flight_id to the set of processed flight_ids
                processed_flight_ids.add(flight_id)

        # Create a DataFrame from the features and save to a CSV file after processing all flights
        if feature_list:
            features_df = pd.DataFrame(feature_list)
            output_file = os.path.join(r"D:\Project_PRC_Eurocontrol\PAR_Files\N7_151024", f"flight_features_{D1.strftime('%Y-%m-%d')}.csv")
            try:
                # If the file exists, append without header; else, create with header
                if os.path.exists(output_file):
                    features_df.to_csv(output_file, mode='a', index=False, header=False)
                else:
                    features_df.to_csv(output_file, index=False)
                print(f"Features saved to {output_file}")
            except Exception as e:
                print(f"Error saving features to CSV: {e}")
                traceback.print_exc()
        else:
            print("No features generated; skipping CSV save.")

        # Clean up to free memory
        del df, t, t_filtered, t_resampled, t_evaluated, feature_list
        gc.collect()

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
