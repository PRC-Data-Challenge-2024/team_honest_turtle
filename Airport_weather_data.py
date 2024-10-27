import pandas as pd
from datetime import datetime
from meteostat import Stations, Hourly
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    filename='weather_data_extraction.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Function to get weather data using Meteostat
def get_weather_data(lat, lon, time):
    """
    Fetch weather data using Meteostat for a specific latitude, longitude, and time.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        time (datetime): Time for which the data is to be extracted.

    Returns:
        dict: Weather data including temperature, wind components, pressure, etc.
    """
    # Remove timezone information to make it offset-naive
    time_naive = time.replace(tzinfo=None)

    # Create start and end time for the given date (full day)
    start = time_naive.replace(hour=0, minute=0, second=0)
    end = time_naive.replace(hour=23, minute=59, second=59)

    # Find weather stations near the given coordinates
    stations = Stations()
    stations = stations.nearby(lat, lon)
    stations_df = stations.fetch()

    # Log the number of stations found
    logging.info(f"Found {len(stations_df)} stations near coordinates ({lat}, {lon}).")

    if not stations_df.empty:
        # The 'distance' column is already provided in meters
        stations_df['distance_km'] = stations_df['distance'] / 1000  # Convert meters to kilometers

        # Filter stations within the specified radius
        radius_km = 500
        stations_within_radius = stations_df[stations_df['distance_km'] <= radius_km]

        logging.info(f"{len(stations_within_radius)} stations within {radius_km} km radius.")

        if not stations_within_radius.empty:
            # Iterate over stations within radius
            for station_id in stations_within_radius.index:
                # Fetch hourly data for the station
                data = Hourly(station_id, start, end)
                data = data.fetch()

                if not data.empty:
                    logging.info(f"Data found for station {station_id} at distance {stations_within_radius.loc[station_id]['distance_km']:.2f} km.")
                    temperature = data['temp'].values[0] + 273.15 if 'temp' in data.columns else None  # Convert to Kelvin
                    dew_point = data['dwpt'].values[0] + 273.15 if 'dwpt' in data.columns else None  # Convert to Kelvin
                    relative_humidity = data['rhum'].values[0] if 'rhum' in data.columns else None
                    precipitation = data['prcp'].values[0] if 'prcp' in data.columns else None
                    snow_depth = data['snow'].values[0] if 'snow' in data.columns else None
                    wind_direction = data['wdir'].values[0] if 'wdir' in data.columns else None
                    wind_speed = data['wspd'].values[0] * 0.539957 if 'wspd' in data.columns else None  # Convert to knots
                    wind_gust = data['wpgt'].values[0] * 0.539957 if 'wpgt' in data.columns else None  # Convert to knots
                    pressure = data['pres'].values[0] if 'pres' in data.columns else None
                    sunshine = data['tsun'].values[0] if 'tsun' in data.columns else None
                    weather_code = data['coco'].values[0] if 'coco' in data.columns else None

                    return {
                        'temperature_kelvin': temperature,
                        'dew_point_kelvin': dew_point,
                        'relative_humidity_percent': relative_humidity,
                        'precipitation_mm': precipitation,
                        'snow_depth_mm': snow_depth,
                        'wind_direction_degrees': wind_direction,
                        'wind_speed_knots': wind_speed,
                        'wind_gust_knots': wind_gust,
                        'pressure_hPa': pressure,
                        'sunshine_minutes': sunshine,
                        'weather_condition_code': weather_code,
                        'station_id': station_id,
                        'station_distance_km': stations_within_radius.loc[station_id]['distance_km']
                    }
                else:
                    logging.info(f"No data for station {station_id} on date {start.date()}")
            else:
                logging.warning(f"No weather data available from stations within {radius_km} km radius.")
                return None
        else:
            logging.warning(f"No weather stations found within {radius_km} km radius for coordinates ({lat}, {lon})")
            return None
    else:
        logging.warning(f"No weather stations found near coordinates ({lat}, {lon})")
        return None

# Main function to orchestrate the data extraction process
def main():
    # Load input CSV files
    challenge_set_path = r"F:\Project_PRC_Eurocontrol\NEW\Input_data\BASE_DATA\challenge_set.csv"
    submission_set_path = r"F:\Project_PRC_Eurocontrol\NEW\Input_data\BASE_DATA\final_submission_set.csv"
    airports_useful_path = r"F:\Project_PRC_Eurocontrol\NEW\Input_data\BASE_DATA\Airports_Useful.csv"
    existing_weather_data_path = r"F:\Project_PRC_Eurocontrol\NEW\Input_data\BASE_DATA\Airport_weather_data2.csv"

    # Read datasets
    challenge_set_df = pd.read_csv(challenge_set_path)
    submission_set_df = pd.read_csv(submission_set_path)
    airports_useful_df = pd.read_csv(airports_useful_path)
    existing_weather_df = pd.read_csv(existing_weather_data_path)

    logging.info("All datasets loaded successfully.")

    # Use the correct column names for merging
    if 'Airport_CODE' in airports_useful_df.columns:
        adep_merge_key = 'Airport_CODE'
    else:
        raise KeyError("No suitable column found in airports_useful_df for merging with adep and ades.")

    # Combine challenge and submission sets
    combined_df = pd.concat([challenge_set_df, submission_set_df], ignore_index=True)
    logging.info("Challenge and submission sets combined successfully.")

    # Merge combined dataset with airports data to get latitude and longitude
    combined_df = combined_df.merge(airports_useful_df, left_on='adep', right_on=adep_merge_key, how='left', suffixes=('', '_adep'))
    combined_df = combined_df.merge(airports_useful_df, left_on='ades', right_on=adep_merge_key, how='left', suffixes=('', '_ades'))

    logging.info("Flight details merged with airport latitude and longitude.")

    # Filter out flights with missing weather data
    missing_weather_df = combined_df[~combined_df['flight_id'].isin(existing_weather_df['flight_id'])]
    logging.info(f"{len(missing_weather_df)} flights with missing weather data identified.")

    # Prepare a list of results
    results = []

    # Use ThreadPoolExecutor to speed up data fetching
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_flight = {
            executor.submit(process_flight, row): row['flight_id'] for index, row in missing_weather_df.iterrows()
        }

        for future in tqdm(as_completed(future_to_flight), total=len(future_to_flight), desc="Processing flights"):
            flight_id = future_to_flight[future]
            try:
                flight_data = future.result()
                if flight_data:
                    results.append(flight_data)
            except Exception as e:
                logging.error(f"Error processing flight {flight_id}: {e}")

    # Convert results to a DataFrame
    new_results_df = pd.DataFrame(results)

    # Concatenate new results with existing weather data
    final_results_df = pd.concat([existing_weather_df, new_results_df], ignore_index=True)

    # Save the updated results to a CSV file
    final_results_df.to_csv('Airport_weather_data_updated.csv', index=False)
    logging.info("Data extraction complete! Results saved to Airport_weather_data_updated.csv")

# Process flight function remains unchanged
def process_flight(row):
    flight_id = row['flight_id']

    # Initialize a dictionary to store flight data
    flight_data = row.to_dict()

    # Extract departure (adep) data
    adep_lat = row['Airport_latitude']
    adep_lon = row['Airport_longitude']
    actual_offblock_time = row['actual_offblock_time']  # In ISO 8601 format with 'Z'

    # Parse the offblock time as a datetime object
    try:
        offblock_time_dt = datetime.fromisoformat(actual_offblock_time.replace("Z", "+00:00"))
        logging.info(f"Parsed offblock time for flight {flight_id}: {offblock_time_dt}")
    except ValueError as e:
        logging.error(f"Error parsing offblock time for flight {flight_id}: {e}")
        return None

    logging.info(f"Fetching departure weather data for flight ID: {flight_id}")
    departure_weather = get_weather_data(adep_lat, adep_lon, offblock_time_dt)

    if departure_weather:
        # Store departure weather data in flight_data dictionary
        flight_data.update({
            'adep_temperature_kelvin': departure_weather['temperature_kelvin'],
            'adep_dew_point_kelvin': departure_weather['dew_point_kelvin'],
            'adep_relative_humidity_percent': departure_weather['relative_humidity_percent'],
            'adep_precipitation_mm': departure_weather['precipitation_mm'],
            'adep_snow_depth_mm': departure_weather['snow_depth_mm'],
            'adep_wind_direction_degrees': departure_weather['wind_direction_degrees'],
            'adep_wind_speed_knots': departure_weather['wind_speed_knots'],
            'adep_wind_gust_knots': departure_weather['wind_gust_knots'],
            'adep_pressure_hPa': departure_weather['pressure_hPa'],
            'adep_sunshine_minutes': departure_weather['sunshine_minutes'],
            'adep_weather_condition_code': departure_weather['weather_condition_code'],
        })
        logging.info(f"Departure weather data extracted for flight ID: {flight_id}")
    else:
        logging.warning(f"No departure weather data found for flight ID: {flight_id}")

    # Extract arrival (ades) data
    ades_lat = row['Airport_latitude_ades']
    ades_lon = row['Airport_longitude_ades']
    arrival_time = row['arrival_time']  # In ISO 8601 format with 'Z'

    # Parse the arrival time as a datetime object
    try:
        arrival_time_dt = datetime.fromisoformat(arrival_time.replace("Z", "+00:00"))
        logging.info(f"Parsed arrival time for flight {flight_id}: {arrival_time_dt}")
    except ValueError as e:
        logging.error(f"Error parsing arrival time for flight {flight_id}: {e}")
        return None

    logging.info(f"Fetching arrival weather data for flight ID: {flight_id}")
    arrival_weather = get_weather_data(ades_lat, ades_lon, arrival_time_dt)

    if arrival_weather:
        # Store arrival weather data in flight_data dictionary
        flight_data.update({
            'ades_temperature_kelvin': arrival_weather['temperature_kelvin'],
            'ades_dew_point_kelvin': arrival_weather['dew_point_kelvin'],
            'ades_relative_humidity_percent': arrival_weather['relative_humidity_percent'],
            'ades_precipitation_mm': arrival_weather['precipitation_mm'],
            'ades_snow_depth_mm': arrival_weather['snow_depth_mm'],
            'ades_wind_direction_degrees': arrival_weather['wind_direction_degrees'],
            'ades_wind_speed_knots': arrival_weather['wind_speed_knots'],
            'ades_wind_gust_knots': arrival_weather['wind_gust_knots'],
            'ades_pressure_hPa': arrival_weather['pressure_hPa'],
            'ades_sunshine_minutes': arrival_weather['sunshine_minutes'],
            'ades_weather_condition_code': arrival_weather['weather_condition_code'],
            'ades_station_id': arrival_weather['station_id'],
            'ades_station_distance_km': arrival_weather['station_distance_km']
        })
        logging.info(f"Arrival weather data extracted for flight ID: {flight_id}")
    else:
        logging.warning(f"No arrival weather data found for flight ID: {flight_id}")

    return flight_data

if __name__ == "__main__":
    main()
