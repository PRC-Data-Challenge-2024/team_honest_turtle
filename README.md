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
