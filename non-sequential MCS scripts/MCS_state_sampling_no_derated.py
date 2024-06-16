"""
File: MCS_state_sampling_derated.py
Author: Julian Wuijts
Date: 05-07-2024
Description: A Python script to create a generation profile for a non-sequential MCS with the option for wind generation,
but without derated states to speed up the computational time.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from COPT_wind import CreatePowerProfile

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
def generate_generator_profile_state(total_hours, FOR):
    U = generate_random_values(total_hours)
    generator_profile_state = (U >= FOR).astype(int)
    return generator_profile_state

# Optimize generation_profile function
def generation_profile(generator_data, alpha, beta, turbine_specs):
    generator_data = generator_data.to_numpy()
    
    FOR = generator_data[2, 2::2]
    generator_capacity_list = generator_data[2, 1::2]
    generator_type_list = generator_data[3, 1::2]
    number_of_generators = generator_data[4, 1::2].astype(int)
    generator_number = sum(number_of_generators)

    total_hours = 8736
    wind_speed_list = generate_wind_speed_list(total_hours, alpha, beta)
    P_output_list, _, _ = CreatePowerProfile(wind_speed_list, turbine_specs)
    wind_power_profile = P_output_list[:total_hours]

    generator_data_filtered = np.repeat([generator_type_list, generator_capacity_list, FOR], number_of_generators, axis=1)
    generator_type_list = generator_data_filtered[0]
    generator_capacity_list = generator_data_filtered[1]
    FOR = generator_data_filtered[2]

    summed_generator_profile = np.zeros(total_hours)

    for i in range(generator_number):
        generator_profile_state = generate_generator_profile_state(total_hours, FOR[i])

        if generator_type_list[i] == 0:
            generator_profile_capacity = generator_capacity_list[i] * generator_profile_state
        elif generator_type_list[i] == 1:
            generator_profile_capacity = wind_power_profile * generator_profile_state
        else:
            raise ValueError('Invalid generator type')

        summed_generator_profile += generator_profile_capacity

    return summed_generator_profile

# Use the optimized generation_profile function
if __name__ == "__main__":
    beta = 1.8459062218183633
    alpha = 4.703721846769401

    generator_data = pd.read_excel("/Users/jswui/Desktop/RBTS_MCS_wind_state_sampling.xlsx", sheet_name="Generators") #Import generator data from Excel
    turbine_specs = pd.read_excel("/Users/jswui/Desktop/RBTS_MCS_wind_state_sampling.xlsx", sheet_name="Turbines")
    
    summed_generator_profile = generation_profile(generator_data, alpha, beta, turbine_specs)
    time = np.arange(1, 8737)

    plt.plot(time, summed_generator_profile)
    plt.xlabel('Time [hours]')
    plt.ylabel('Output [MW]')
    plt.title('Generator Availability')
    plt.grid(True)
    plt.show()
