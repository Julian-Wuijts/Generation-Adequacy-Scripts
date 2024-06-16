"""
File: MCS_state_sampling_derated.py
Author: Julian Wuijts
Date: 05-07-2024
Description: A Python script to create a generation profile for a non-sequential MCS with the option for wind generation and derated states.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from COPT_wind import CreatePowerProfile

def object_to_numpy_array(object):
    processed_list = []
    for value in object:
        try:
            processed_list.append(float(value))
        except (ValueError, TypeError):
            processed_list.append(np.nan)
    return np.array(processed_list)

# Precompute random values
def generate_random_values(total_hours):
    U = np.random.rand(total_hours)
    return U

# Generate wind speed list using precomputed random values
def generate_wind_speed_list(total_hours, alpha, beta):
    U = generate_random_values(total_hours)
    wind_speed_list = alpha * ((-np.log(U))**(1/beta))
    return wind_speed_list

# Generate generator profile state using precomputed random values
def generate_generator_profile_state(total_hours, state_probability,generator_capacity):
    U = generate_random_values(total_hours)

    state_probability_cleaned = state_probability[~np.isnan(state_probability)]
    state_probability_boundaries = np.cumsum(state_probability_cleaned)
    
    state = np.searchsorted(state_probability_boundaries, U, side='right')
    state_outage = generator_capacity[state]
    state_outage = np.array(state_outage, dtype=float)
    generator_profile_state = 1-(state_outage/max(generator_capacity))

    return generator_profile_state

# Optimize generation_profile function
def generation_profile(generator_data, alpha, beta, turbine_specs):
    
    generator_data = generator_data.to_numpy()
    
    state_probability = generator_data[3:, 2::2]
    generator_capacity_list = generator_data[3:, 1::2]
    generator_type_list = generator_data[0, 1::2]
    number_of_generators = generator_data[1, 1::2].astype(int)
    number_of_states = generator_data[3,0]
    generator_number = sum(number_of_generators)

    repeated_generator_type_list = np.repeat(generator_type_list, number_of_generators)
    repeated_generator_capacity_list = np.repeat(generator_capacity_list, number_of_generators, axis=1)
    repeated_state_probability = np.repeat(state_probability, number_of_generators, axis=1)

    # Concatenate the repeated arrays into the final structure
    generator_data_filtered = np.vstack([repeated_generator_type_list, repeated_generator_capacity_list, repeated_state_probability])

    # Splitting the concatenated data into individual components
    generator_type_list = generator_data_filtered[0]
    generator_capacity_list = generator_data_filtered[1:number_of_states+1]
    state_probability_list = generator_data_filtered[number_of_states+1:]
    state_probability_list = np.array([object_to_numpy_array(row) for row in state_probability_list])

    total_hours = 8736
    wind_speed_list = generate_wind_speed_list(total_hours, alpha, beta)
    P_output_list, _, _ = CreatePowerProfile(wind_speed_list, turbine_specs)
    wind_power_profile = P_output_list[:total_hours]

    summed_generator_profile = np.zeros(total_hours)

    for i in range(generator_number):
        generator_profile_state = generate_generator_profile_state(total_hours, state_probability_list[:,i],generator_capacity_list[:,i])
        

        if generator_type_list[i] == 0:
            generator_profile_capacity = max(generator_capacity_list[:,i]) * generator_profile_state
        elif generator_type_list[i] == 1:
            generator_profile_capacity = wind_power_profile * generator_profile_state
        else:
            print('Invalid generator type')

        summed_generator_profile += generator_profile_capacity

    return summed_generator_profile

# Use the optimized generation_profile function
if __name__ == "__main__":
    beta = 1.8459062218183633
    alpha = 4.703721846769401

    ## CHANGE THE DATAPATH HERE ##
    generator_data = pd.read_excel("/Users/jswui/Desktop/RBTS_MCS_wind_state_sampling_derated.xlsx", sheet_name="Generators") #Import generator data from Excel
    turbine_specs = pd.read_excel("/Users/jswui/Desktop/RBTS_MCS_wind_state_sampling_derated.xlsx", sheet_name="Turbines")
    
    summed_generator_profile = generation_profile(generator_data, alpha, beta, turbine_specs)
    time = np.arange(1, 8737)

    plt.plot(time, summed_generator_profile)
    plt.xlabel('Time [hours]')
    plt.ylabel('Output [MW]')
    plt.title('Generator Availability')
    plt.grid(True)
    plt.show()
