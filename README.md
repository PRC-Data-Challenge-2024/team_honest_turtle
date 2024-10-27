# PRC Data Challenge 2024 - team_honest_turtle

## Introduction

Hello! I’m Somnath Panigrahi, currently pursuing my Master’s degree in Air Transport and Logistics at TU Dresden. In the PRC Data Challenge 2024, I’m competing as part of **team_honest_turtle**. This challenge offers a unique opportunity to apply data science techniques to the aviation sector by accurately estimating the **Actual TakeOff Weight (ATOW)** of flights based on flight and trajectory data.

The PRC Data Challenge aims to develop an open Machine Learning (ML) model capable of estimating ATOW, a critical metric used to calculate fuel consumption and emissions. By moving beyond generic assumptions of ATOW (often set as a fixed percentage of Maximum TakeOff Weight or MTOW), this challenge seeks to create a transparent, reproducible model that supports environmental impact assessments and fuel efficiency improvements.

### Datasets Overview

Two main datasets have been provided for this challenge:

1. **Data for Modeling**: This dataset contains flight details for **369,013 flights** in Europe in 2022, along with high-resolution trajectory data for a majority of these flights. The data has been organized as follows:

   - **Flight List** (`challenge_set.csv`):
     - **Flight Identification**: `flight_id` (unique identifier), `callsign` (obfuscated)
     - **Origin/Destination**: `adep` (departure airport ICAO code), `ades` (destination airport ICAO code), `name_adep`, `name_ades`, `country_code_adep`, `country_code_ades`
     - **Timing**: `date`, `actual_offblock_time`, `arrival_time`
     - **Aircraft Information**: `aircraft_type`, `wtc` (Wake Turbulence Category)
     - **Airline**: `airline` (obfuscated code)
     - **Operational Metrics**: `flight_duration`, `taxiout_time`, `flown_distance`, `tow` (TakeOff Weight in kg)

   - **Trajectory Data**:
     - Each file contains trajectory data at **1-second intervals** in `.parquet` format, resulting in approximately **158 GB** total.
     - **Fields**:
       - `flight_id`, `icao24`: Identifiers for each flight
       - **4D Position**: `longitude`, `latitude`, `altitude`, `timestamp`
       - **Speed and Track**: `groundspeed`, `track`, `track_unwrapped`, `vertical_rate`
       - **Meteorological Data** (where available): `u_component_of_wind`, `v_component_of_wind`, `temperature`

2. **Dataset for Submission**:
   - **Submission Set** (`submission_set.csv`): Contains **105,959 flights** that require ATOW estimates. The structure matches the Flight List, but with an empty `tow` column. For the final ranking, an additional **52,190 flights** will be included in `final_submission_set.csv`.

### Data Access

The data is accessible through two S3 buckets:

- `competition-data/`: Contains data for modeling, including the flight list (`challenge_set.csv`), daily trajectory files, and submission sets.
- `submissions/`: Stores the teams’ submissions for ranking, though contents are not listed publicly per account.

The `competition-data/` bucket is organized as follows:


### Data Types and Importance

Key data types include:

- **Categorical Variables**: `adep`, `ades`, `aircraft_type`, `airline`, `country_code_adep`, `country_code_ades`
- **Numerical Variables**: `flight_duration`, `taxiout_time`, `flown_distance`, `altitude`, `groundspeed`, `vertical_rate`
- **Temporal Variables**: `date`, `actual_offblock_time`, `arrival_time`, `timestamp`

### Real-World Applications

Estimating ATOW accurately has important real-world applications. Precise ATOW estimates allow for:

- **Fuel Efficiency**: By refining fuel load requirements, airlines can reduce fuel waste and minimize costs.
- **Environmental Impact Reduction**: Accurate ATOW is a vital input for emission models that estimate gaseous emissions based on actual fuel burned, supporting aviation’s commitment to environmental sustainability.
- **Operational Planning**: Airlines can optimize routes, plan for better weight distribution, and improve flight safety by understanding the precise takeoff weight requirements.

This challenge underscores the potential for data science to address pressing aviation issues, with an open approach that promotes transparency, reproducibility, and collaboration for a better understanding of aviation's environmental impact.

# Methodology

## Additional Datasets and Data Enrichment

To enhance the predictive accuracy of our model, we integrated several external datasets to provide additional features relevant to aircraft performance, weather, and airport data. Here is an overview of the additional data sources used:

1. **Aircraft Database**:
   - We enriched our flight data with detailed aircraft information from authoritative sources:
     - **EUROCONTROL Aircraft Database**: Provides comprehensive aircraft performance parameters that are critical for assessing fuel burn rates and load capacities.
       - [EUROCONTROL Aircraft Database](https://contentzone.eurocontrol.int/aircraftperformance/default.aspx)
     - **FAA Aircraft Characteristics Database**: Supplies specifications on various aircraft types and their performance profiles.
       - [FAA Aircraft Characteristics Database](https://www.faa.gov/airports/engineering/aircraft_char_database)

2. **Engine Parameters**:
   - By linking aircraft data with **engine performance specifications**, we derived insights into thrust, fuel consumption rates, and altitude-specific performance, which are instrumental in estimating the Actual TakeOff Weight (ATOW).

3. **Weather Data**:
   - Weather data significantly influences fuel burn and load planning. For real-time and historical weather data, we utilized:
     - **Meteostat API**: Offers open-source meteorological data, including wind speed, temperature, and precipitation at various altitudes.
       - [Meteostat API](https://github.com/meteostat)

4. **Airport Data**:
   - **OurAirports Database**: Provided detailed information on airport characteristics, including location, runway dimensions, and operational constraints, which can affect departure and landing efficiency.
     - [OurAirports Database](https://ourairports.com/data/)

5. **Airport Movement Data**:
   - **Eurostat Airport Movement Data (2010-2024)**: Provides historical and current data on airport movements, including aircraft types, flight frequencies, and peak operation periods, which aid in understanding the operational context and potential delays.

6. **ADS-B Data Extraction**:
   - **ADS-B Dataset (285GB)**: To capture high-resolution flight trajectories, we extracted data from a 285GB ADS-B dataset, utilizing specialized libraries:
     - **Traffic**: For handling, filtering, and visualizing ADS-B trajectories.
     - **OpenAP**: Used to analyze flight paths and compute derived features such as rate of climb/descent and in-flight adjustments.

## Model Training and Techniques

For modeling, we experimented with various machine learning approaches, integrating AutoML methods for parameter tuning and leveraging advanced machine learning algorithms to optimize prediction accuracy. Here are the core methods and packages used:

1. **Automated Machine Learning (AutoML)**:
   - We used AutoML to automate hyperparameter tuning and model selection. This allowed for an efficient workflow that explored various modeling techniques without extensive manual configuration.

2. **Machine Learning Algorithms**:
   - **LightGBM**: A gradient boosting framework optimized for speed and performance on large datasets.
   - **CatBoost**: Particularly effective on categorical features, CatBoost helped handle the large volume of categorical data related to airports, aircraft types, and operational patterns.
   - **Neural Networks**:
     - **NNTorch**: A neural network library built on PyTorch, used to design a deep learning model that captures complex non-linear interactions between variables.
     - **Standard Neural Networks**: We also tested custom neural network architectures to explore intricate relationships within the data.

Each of these algorithms was fine-tuned based on feature importance, cross-validation, and hold-out test sets to ensure generalizability of the model.

## Summary of Methodology

By integrating external datasets and employing a robust selection of machine learning techniques, this methodology is designed to produce a reliable and accurate model for estimating ATOW. The combination of high-resolution ADS-B data, enriched aircraft and weather information, and state-of-the-art modeling approaches positions our model to offer substantial real-world applications in fuel efficiency, operational optimization, and environmental impact assessment.

# Feature Description and Sources

## Basic Flight Information

1. **flight_id**: Unique identifier for each flight. *Source: Generated by the data provider.*
2. **date**: Date of the flight. *Source: Provided in the flight list.*
3. **callsign**: Obfuscated callsign for privacy purposes. *Source: Provided in the flight list.*
4. **adep**: Aerodrome of Departure (ICAO code). *Source: Provided in the flight list.*
5. **name_adep**: Name of the departure airport. *Source: OurAirports Database.*
6. **country_code_adep**: Country code for the departure airport. *Source: OurAirports Database.*
7. **ades**: Aerodrome of Destination (ICAO code). *Source: Provided in the flight list.*
8. **name_ades**: Name of the destination airport. *Source: OurAirports Database.*
9. **country_code_ades**: Country code for the destination airport. *Source: OurAirports Database.*
10. **actual_offblock_time**: Actual time the aircraft left the gate. *Source: Provided in the flight list.*
11. **arrival_time**: Arrival time at the destination gate. *Source: Provided in the flight list.*

## Aircraft and Airline Information

12. **aircraft_type**: ICAO aircraft type designator. *Source: EUROCONTROL and FAA Aircraft Database.*
13. **wtc**: Wake Turbulence Category of the aircraft. *Source: EUROCONTROL and FAA Aircraft Database.*
14. **airline**: Obfuscated code for the airline operator. *Source: Provided in the flight list.*

## Flight Timing and Operational Metrics

15. **flight_duration**: Total duration of the flight in minutes. *Source: Calculated from off-block and arrival times.*
16. **taxiout_time**: Duration in minutes from pushback to takeoff. *Source: Calculated from ADS-B data.*
17. **flown_distance**: Great-circle distance (in nautical miles) between departure and arrival airports. *Source: Calculated from geographic coordinates.*
18. **tow**: Estimated TakeOff Weight (TOW) of the flight. *Source: Provided in the flight list.*

## ADS-B Derived Metrics

19. **total_flight_duration_sec_adsb**: Total flight duration from ADS-B data (in seconds). *Source: ADS-B data extraction.*
20. **total_distance_flown_nm_adsb**: Total distance flown based on ADS-B data (in nautical miles). *Source: ADS-B data extraction.*
21. **number_of_position_reports_adsb**: Total number of position reports recorded by ADS-B. *Source: ADS-B data extraction.*

## In-Flight Performance Metrics

22. **average_groundspeed_knots_adsb**: Average groundspeed of the aircraft in knots. *Source: Derived from ADS-B data.*
23. **max_groundspeed_knots_adsb**: Maximum groundspeed recorded during the flight. *Source: Derived from ADS-B data.*
24. **average_vertical_rate_ft_per_min_adsb**: Average rate of climb/descent in feet per minute. *Source: ADS-B data extraction.*
25. **max_vertical_rate_ft_per_min_adsb**: Maximum rate of climb/descent in feet per minute. *Source: ADS-B data extraction.*
26. **num_climb_descent_cycles_adsb**: Number of climb and descent cycles during the flight. *Source: Derived from ADS-B data.*
27. **average_track_change_deg_adsb**: Average change in track direction during the flight (in degrees). *Source: ADS-B data extraction.*

## Meteorological Information (ADS-B Path)

28. **average_temperature_k_adsb**: Average temperature along the flight path (Kelvin). *Source: Meteostat API.*
29. **average_wind_speed_m_per_s_adsb**: Average wind speed along the flight path (meters per second). *Source: Meteostat API.*
30. **average_specific_humidity_adsb**: Average specific humidity along the flight path. *Source: Meteostat API.*
31. **average_altitude_change_rate_ft_per_sec_adsb**: Average rate of altitude change during the flight (feet per second). *Source: ADS-B data extraction.*

## Route Efficiency Metrics

32. **great_circle_distance_nm_adsb**: Great-circle distance calculated from ADS-B trajectory data (nautical miles). *Source: Calculated from ADS-B data.*
33. **route_efficiency_adsb**: Efficiency of the flown route compared to the great-circle distance. *Source: Calculated from ADS-B data.*

## Climb, Cruise, Descent, and Landing Parameters

34-50. **Climb Parameters**: Metrics such as **average speed**, **vertical rate**, **time**, **distance** in the climb phase. *Source: ADS-B and EUROCONTROL/FAA Aircraft Database.*
51-67. **Cruise Parameters**: Includes **cruise speed**, **altitude**, and **duration** in the cruise phase. *Source: ADS-B and EUROCONTROL/FAA Aircraft Database.*
68-84. **Descent Parameters**: Metrics such as **average speed**, **vertical rate**, and **time** during descent. *Source: ADS-B and EUROCONTROL/FAA Aircraft Database.*
85-101. **Landing Parameters**: Includes metrics like **landing speed** and **descent rate** on final approach. *Source: Derived from ADS-B data.*

## Total Fuel Consumption

102. **total_fuel_consumption_kg_adsb**: Total fuel consumed during the flight in kilograms. *Source: Estimated based on aircraft type and flight phase using ADS-B data.*

## Takeoff and Landing Environment Metrics

103-112. **Takeoff Environment**: Average conditions such as **direction**, **temperature**, **vertical rate**, and **altitude** at takeoff. *Source: Meteostat API and ADS-B data.*
113-122. **Landing Environment**: Similar metrics at the landing phase, including **average temperature** and **speed**. *Source: Meteostat API and ADS-B data.*

## Weather Conditions at ADEP and ADES

123-134. **Weather Data for ADEP and ADES**: Includes **temperature**, **dew point**, **relative humidity**, **precipitation**, and **wind speed** at departure and arrival airports. *Source: Meteostat API.*

## Airport Characteristics

135-144. **Airport Runway Details**: Includes **runway length**, **width**, **elevation**, and **latitude/longitude** for both departure and destination airports. *Source: OurAirports Database.*

## Aircraft Performance Specifications (ADB2)

145-174. **Aircraft-Specific Parameters (ADB2)**: Performance specifications such as **MTOW (Maximum TakeOff Weight)**, **initial climb rate**, **cruise speed**, **descent rate**, **approach speed**, **fuel efficiency** metrics, and noise levels. *Source: EUROCONTROL and FAA Aircraft Database.*

## Environmental and Operational Ratios

175-184. **Performance Ratios**: Ratios comparing ADS-B metrics (like speed and climb rate) to ADB2 baseline metrics, assessing performance against standard aircraft specifications. *Source: Derived from ADS-B and ADB2 data.*

## Temperature and Humidity Differences (ADEP - ADES)

185-190. **Environmental Differences**: Differences in **temperature**, **humidity**, and **altitude** between departure and destination. *Source: Meteostat API.*

## Flight Efficiency and Runway Ratios

191-200. **Operational Efficiency Metrics**: Calculated metrics like **flight duration efficiency** and **altitude change rate efficiency**.
201-210. **Runway Ratios**: Ratios between runway lengths and aircraft characteristics (e.g., wing span) at departure and arrival airports. *Source: OurAirports Database and EUROCONTROL/FAA Aircraft Database.*

## Categorical and Derived Features

211-220. **Flight and Route Categories**: Derived categories such as **flight type**, **day or night**, **season**, and **distance category**.
221-230. **Binned Speed Groups**: Includes binned speed groups for takeoff and landing speeds, representing aircraft performance groups.

## Environmental Ratios

231-236. **Barometric and Temperature Ratios**: Comparisons of barometric pressure and temperature at ADEP to aircraft environmental limits. *Source: Derived from Meteostat API and ADB2 data.*

