import os
import pandas as pd

# Directory containing the .tsv.gz files
data_dir = r"F:\Project_PRC_Eurocontrol\NEW\Input_data\Passenger_route"

# Define the list of country codes based on the pattern you provided
country_codes = [
    'be', 'bg', 'cz', 'dk', 'de', 'ee', 'ie', 'el', 'es', 'fr', 'hr', 'it', 'cy', 'lv', 'lt', 
    'lu', 'hu', 'mt', 'nl', 'at', 'pl', 'pt', 'ro', 'si', 'sk', 'fi', 'se', 'is', 'no', 'ch', 
    'uk', 'ba', 'me', 'mk', 'tr'
]

# Initialize an empty list to store dataframes
dfs = []

# Loop through each country code to load the respective .tsv.gz file
for code in country_codes:
    # Create the full path for each country file
    file_path = os.path.join(data_dir, f'estat_avia_par_{code}.tsv.gz')
    
    # Check if the file exists
    if os.path.exists(file_path):
        print(f"Processing {file_path}")
        
        # Read the TSV file with compression
        df = pd.read_csv(file_path, delimiter='\t', compression='gzip', low_memory=False)
        
        # Extract the last 15 characters from the 'freq,unit,tra_meas,airp_pr\\TIME_PERIOD' column as Airport_pair
        df['Airport_pair'] = df['freq,unit,tra_meas,airp_pr\\TIME_PERIOD'].str[-15:]
        
        # Drop the 'freq,unit,tra_meas,airp_pr\\TIME_PERIOD' column
        df = df.drop(columns=['freq,unit,tra_meas,airp_pr\\TIME_PERIOD'])
        
        # Reshape the data from wide to long format using melt
        df_melted = df.melt(id_vars=['Airport_pair'], var_name='Year_Month', value_name='Passengers')
        
        # Filter out rows with missing passenger values (i.e., ':')
        df_melted = df_melted[df_melted['Passengers'] != ':']
        
        # Convert passengers to numeric (after filtering)
        df_melted['Passengers'] = pd.to_numeric(df_melted['Passengers'], errors='coerce')
        
        # Split 'Year_Month' into 'Year' and 'Month' (Month can be monthly or quarterly)
        df_melted[['Year', 'Month']] = df_melted['Year_Month'].str.extract(r'(\d{4})(?:-(\d{2}|Q\d))?')
        
        # Replace missing months with "Annual" to handle yearly data
        df_melted['Month'] = df_melted['Month'].fillna('Annual')
        
        # Filter out non-numeric months like 'Annual' and keep only valid numeric months
        df_melted = df_melted[df_melted['Month'].str.isnumeric()]
        
        # Filter to keep only data from 2010 to 2024
        df_filtered = df_melted[(df_melted['Year'].astype(int) >= 2010) & (df_melted['Year'].astype(int) <= 2024)]
        
        # Remove rows where Passengers are equal to 0
        df_filtered = df_filtered[df_filtered['Passengers'] > 0]
        
        # Append the filtered dataframe to the list
        dfs.append(df_filtered)
    else:
        print(f"File for country {code} not found.")

# Concatenate all dataframes into a single dataframe
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Reorder the columns to have Airport_pair, Year, Month, Passengers
    df_final = combined_df[['Airport_pair', 'Year', 'Month', 'Passengers']]
    
    # Save the final result to a CSV file
    output_file = r"F:\Project_PRC_Eurocontrol\NEW\Input_data\Euro_passenger_2010_2024.csv"
    df_final.to_csv(output_file, index=False)
    
    print(f"Processed data saved to {output_file}")
else:
    print("No dataframes to concatenate.")
