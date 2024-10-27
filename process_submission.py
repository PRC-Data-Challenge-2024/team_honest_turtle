# process_submission.py

import pandas as pd
import math
import numpy as np
import os

# Define the file paths
submission_file = r"F:\Project_PRC_Eurocontrol\NEW\Input_data\BASE_DATA\7\Updated_submission_set_final.csv"

# Define the list of columns to delete
columns_to_delete = [
    'v2_speed_knots_adsb',
    'fuel_consumption_takeoff_kg_adsb',
    'fuel_consumption_climb_kg_adsb',
    'fuel_consumption_cruise_kg_adsb',
    'fuel_consumption_descent_kg_adsb',
    'fuel_consumption_landing_kg_adsb',
    'ADB2_Unnamed: 0',
    'ADB2_GSDB No',
    'ADB2_B/P Ratio',
    'ADB2_Pressure Ratio',
    'ADB2_Rated Thrust (kN)',
    'ADB2_HC EI T/O (g/kg)',
    'ADB2_HC EI C/O (g/kg)',
    'ADB2_HC EI App (g/kg)',
    'ADB2_HC EI Idle (g/kg)',
    'ADB2_HC Number Test',
    'ADB2_HC Number Eng',
    'ADB2_HC Dp/Foo Avg (g/kN)',
    'ADB2_HC Dp/Foo Sigma (g/kN)',
    'ADB2_HC Dp/Foo Min (g/kN)',
    'ADB2_HC Dp/Foo Max (g/kN)',
    'ADB2_HC Dp/Foo Characteristic (g/kN)',
    'ADB2_HC Dp/Foo Characteristic (% of Reg limit)',
    'ADB2_HC LTO Total mass (g)',
    'ADB2_CO EI T/O (g/kg)',
    'ADB2_CO EI C/O (g/kg)',
    'ADB2_CO EI App (g/kg)',
    'ADB2_CO EI Idle (g/kg)',
    'ADB2_CO Number Test',
    'ADB2_CO Number Eng',
    'ADB2_CO Dp/Foo Avg (g/kN)',
    'ADB2_CO Dp/Foo Sigma (g/kN)',
    'ADB2_CO Dp/Foo Min (g/kN)',
    'ADB2_CO Dp/Foo Max (g/kN)',  # Corrected from 'g/KN' to 'g/kN'
    'ADB2_CO Dp/Foo Characteristic (g/kN)',
    'ADB2_CO Dp/Foo Characteristic (% of Reg limit)',
    'ADB2_CO LTO Total Mass (g)',
    'ADB2_NOx EI T/O (g/kg)',
    'ADB2_NOx EI C/O (g/kg)',
    'ADB2_NOx EI App (g/kg)',
    'ADB2_NOx EI Idle (g/kg)',
    'ADB2_NOx Number Test',
    'ADB2_NOx Number Eng',
    'ADB2_NOx Dp/Foo Avg (g/kN)',
    'ADB2_NOx Dp/Foo Sigma (g/kN)',
    'ADB2_NOx Dp/Foo Min (g/kN)',
    'ADB2_NOx Dp/Foo Max (g/kN)',
    'ADB2_NOx Dp/Foo Characteristic (g/kN)',
    'ADB2_NOx Dp/Foo Characteristic (% of original standard)',
    'ADB2_NOx Dp/Foo Characteristic (% of CAEP/2 standard)',
    'ADB2_NOx Dp/Foo Characteristic (% of CAEP/4 standard)',
    'ADB2_NOx Dp/Foo Characteristic (% of CAEP/6 standard)',
    'ADB2_NOx Dp/Foo Characteristic (% of CAEP/8 standard)',
    'ADB2_NOx LTO Total mass (g)',
    'ADB2_SN T/O',
    'ADB2_SN C/O',
    'ADB2_SN App',
    'ADB2_SN Idle',
    'ADB2_SN Number Test',
    'ADB2_SN Number Eng',
    'ADB2_SN Max',
    'ADB2_SN Sigma',
    'ADB2_SN Range Min',
    'ADB2_SN Range Max',
    'ADB2_SN Characteristic',
    'ADB2_SN Characteristic (% of Reg limit)',
    'ADB2_Fuel H/C Ratio Min',
    'ADB2_Fuel H/C Ratio Max',
    'ADB2_Fuel Arom Min (%)',
    'ADB2_Fuel Arom Max (%)',
    'ADB2_Fuel Flow T/O (kg/sec)',
    'ADB2_Fuel Flow C/O (kg/sec)',
    'ADB2_Fuel Flow App (kg/sec)',
    'ADB2_Fuel Flow Idle (kg/sec)',
    'ADB2_Fuel LTO Cycle (kg)'
]

def bin_ratio_column(df, column_name, new_column_name, bin_width=0.05):
    """
    Bins a ratio column into specified width percentages.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # Convert to float instead of int to handle decimal bin_width
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce').fillna(0).astype(float)
    max_ratio = df[column_name].max()
    upper_limit = math.ceil(max_ratio / bin_width) * bin_width
    bins = np.arange(0, upper_limit + bin_width, bin_width)
    # Convert bin edges to percentages for labels
    labels = [f"{int(i * 100)}%" for i in bins[:-1]]
    df[new_column_name] = pd.cut(df[column_name], bins=bins, labels=labels, right=False, include_lowest=True)
    df[new_column_name] = df[new_column_name].cat.add_categories(f"{int(upper_limit * 100)}%+")
    df.loc[df[column_name] >= upper_limit, new_column_name] = f"{int(upper_limit * 100)}%+"
    return df

def create_combined_column(df, columns, new_column_name, delimiter='_'):
    """
    Combines multiple columns into a single column separated by a delimiter.
    """
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in DataFrame.")

    if new_column_name in df.columns:
        df.drop(columns=[new_column_name], inplace=True)

    df[columns] = df[columns].astype(str)
    df[columns] = df[columns].replace('nan', 'Unknown')
    df[new_column_name] = df[columns].agg(delimiter.join, axis=1)
    df[new_column_name] = df[new_column_name].astype(str)
    return df

def process_submission_file(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data from: {file_path}")

        # Step 1: Create Combined Grouping Columns
        combined_columns = [
            (['wtc', 'Flight_distance_category'], 'wtc_Flight_distance_category'),
            (['wtc', 'day_or_night', 'season'], 'wtc_day_or_night_season'),
            (['wtc', 'flight_category'], 'wtc_flight_category'),
            (['wtc', 'airline'], 'wtc_airline'),
            (['airline', 'Flight_distance_category'], 'airline_Flight_distance_category'),
            (['airline', 'flight_category'], 'airline_flight_category'),
            (['wtc', 'Airport_pair'], 'wtc_Airport_pair'),
            (['ADB2_RECAT-EU', 'Flight_distance_category'], 'ADB2_RECAT-EU_Flight_distance_category'),
            (['ADB2_RECAT-EU', 'flight_category'], 'ADB2_RECAT-EU_flight_category')
        ]

        for cols, new_col in combined_columns:
            df = create_combined_column(df, cols, new_col)
            print(f"Created combined column: {new_col}")

        # Step 2: Binning and Combining Ratio Columns
        ratio_columns = [
            ('Toff_speed_to_ADB2_V2_(IAS)_Knots_ratio', 'Toff_speed_binned', 'Toff_speed_binned_aircraft_type'),
            ('landing_speed_to_ADB2_Vat_(IAS)_Knots_ratio', 'landing_speed_binned', 'landing_speed_binned_aircraft_type')
        ]

        for original, binned, combined in ratio_columns:
            df = bin_ratio_column(df, original, binned, bin_width=0.05)
            df = create_combined_column(df, [binned, 'aircraft_type'], combined)
            print(f"Binned {original} into {binned} and created combined column: {combined}")

        # Step 3: Delete Specified Columns
        df.drop(columns=columns_to_delete, inplace=True, errors='ignore')
        print(f"Deleted specified columns from: {file_path}")

        # Step 4: Create Additional Ratio Features
        # 4.1 Pressure Ratios
        pressure_columns_present = ['ADB2_Ambient Baro Min (kPa)', 'ADB2_Ambient Baro Max (kPa)', 'adep_pressure_hPa']
        if all(col in df.columns for col in pressure_columns_present):
            df['adep_pressure_kPa'] = df['adep_pressure_hPa'] * 0.1
            df['Baro_Min_to_Adep_Pressure'] = df['ADB2_Ambient Baro Min (kPa)'] / df['adep_pressure_kPa'].replace(0, np.nan)
            df['Baro_Max_to_Adep_Pressure'] = df['ADB2_Ambient Baro Max (kPa)'] / df['adep_pressure_kPa'].replace(0, np.nan)
            df.drop(columns=['adep_pressure_kPa'], inplace=True)
            print("Created pressure ratio features.")
        else:
            missing = [col for col in pressure_columns_present if col not in df.columns]
            print(f"Missing pressure columns {missing}. Skipping pressure ratio features.")

        # 4.2 Temperature Ratios
        temperature_columns_present = ['ADB2_Ambient Temp Min (K)', 'ADB2_Ambient Temp Max (K)', 'adep_temperature_kelvin']
        if all(col in df.columns for col in temperature_columns_present):
            df['Temp_Min_to_Adep_Temp'] = df['ADB2_Ambient Temp Min (K)'] / df['adep_temperature_kelvin'].replace(0, np.nan)
            df['Temp_Max_to_Adep_Temp'] = df['ADB2_Ambient Temp Max (K)'] / df['adep_temperature_kelvin'].replace(0, np.nan)
            print("Created temperature ratio features.")
        else:
            missing = [col for col in temperature_columns_present if col not in df.columns]
            print(f"Missing temperature columns {missing}. Skipping temperature ratio features.")

        # Step 5: Save the Updated DataFrame
        new_file_path = file_path.replace('.csv', '_final.csv')
        df.to_csv(new_file_path, index=False)
        print(f"Processed and saved: {new_file_path}\n")

    except Exception as e:
        print(f"Error processing {file_path}: {e}\n")

def main():
    process_submission_file(submission_file)

if __name__ == "__main__":
    main()
