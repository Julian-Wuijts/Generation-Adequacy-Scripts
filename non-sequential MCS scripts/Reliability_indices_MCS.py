"""
File: Reliability_indices_MCS.py
Author: Julian Wuijts
Date: 05-07-2024
Description: A Python script to obtain the reliability indices LOLE and EENS using a non-sequential MCS method.
"""

import pandas as pd 
import numpy as np
import time

from Load_curves import CreateLoadCurve
from MCS_state_sampling_no_derated import generation_profile
#from MCS_state_sampling_derated import generation_profile
from matplotlib import pyplot as plt

def calc_LOLE_MCS(load,stopping_criteria):
    print('LOLE MCS calculation \n')
    num_of_iterations = 0
    LOLE_list = []
    LOLE_average_value_list = []
    LOLE_average_value = 0
    
    while num_of_iterations < stopping_criteria: 
        time_unit = 0
        
        summed_generator_profile = generation_profile(generator_data,alpha,beta,turbine_specs)
    
        net_energy = summed_generator_profile-load
        time_unit = np.sum(net_energy < 0)
        LOLE_list.append(time_unit)

        num_of_iterations +=1
        LOLE_average_value = np.mean(LOLE_list)
        LOLE_average_value_list.append(LOLE_average_value)

    print('number of iterations', num_of_iterations)
    print('LOLE value',LOLE_average_value)
    num_of_iterations_list = np.arange(1, num_of_iterations + 1)

    std_dev_temp_list = (LOLE_list - LOLE_average_value) ** 2
    std_dev_temp = np.sum(std_dev_temp_list)

    std_dev = (1/(num_of_iterations*(num_of_iterations-1)))*std_dev_temp
    CoV = (std_dev/np.sqrt(num_of_iterations))/LOLE_average_value
    print('CoV LOLE',CoV)

    end_time = time.time()
    execution_time = end_time - start_time
    print('execution time: ',execution_time,' seconds')

    plt.figure(1)
    plt.plot(num_of_iterations_list, LOLE_average_value_list)
    plt.axhline(y=LOLE_average_value, color='r', linestyle='--', label='LOLE Average')
    plt.xlabel('Number of iterations')
    plt.ylabel('LOLE average value [hours/year]')
    plt.title('MCS LOLE calculation')
    plt.grid(True)
    plt.show()        
    
    return


def calc_EENS_MCS(load,stopping_criteria):
    print('EENS MCS calculation \n')
    num_of_iterations = 0
    EENS_list = []
    EENS_average_value_list = []
    EENS_average_value = 0
    
    while num_of_iterations < stopping_criteria: 
        energy_not_served = 0
        summed_generator_profile = generation_profile(generator_data,alpha,beta,turbine_specs)

        net_energy = summed_generator_profile-load
        net_energy_negative = np.minimum(net_energy, 0)
        energy_not_served = np.sum(-net_energy_negative)
        EENS_list.append(energy_not_served)

        num_of_iterations +=1
        EENS_average_value = np.mean(EENS_list)
        EENS_average_value_list.append(EENS_average_value)

    print('number of iterations', num_of_iterations)
    print('EENS value',EENS_average_value)
    num_of_iterations_list = list(range(1,num_of_iterations+1))

    std_dev_temp_list = []

    std_dev_temp_list = (EENS_list - EENS_average_value) ** 2
    std_dev_temp = np.sum(std_dev_temp_list)
    std_dev = (1/(num_of_iterations*(num_of_iterations-1)))*std_dev_temp
    CoV = (std_dev/np.sqrt(num_of_iterations))/EENS_average_value
    print('CoV EENS',CoV)

    end_time = time.time()
    execution_time = end_time - start_time
    print('execution time: ',execution_time,' seconds')

    plt.figure(2)
    plt.plot(num_of_iterations_list, EENS_average_value_list)
    plt.axhline(y=EENS_average_value, color='r', linestyle='--', label='EENS Average')
    plt.xlabel('Number of iterations')
    plt.ylabel('EENS average value [MWh/year]')
    plt.title('MCS EENS calculation')
    plt.grid(True)
    plt.show()    

    return 


start_time = time.time()

# Specify the alpha and beta parameters based on the wind characteristics #
alpha = 4.703721846769401
beta  = 1.8459062218183633

## CHANGE THE PATH HERE ##
generator_data = pd.read_excel("/Users/jswui/Desktop/RBTS_MCS_wind_state_sampling.xlsx", sheet_name="Generators") #Import generator data from Excel
turbine_specs = pd.read_excel("/Users/jswui/Desktop/RBTS_MCS_wind_state_sampling.xlsx", sheet_name="Turbines")
    
YPL_RBTS = 185
YPL_RTS = 2850

YPL = YPL_RBTS

_ , YPL_plot, WPL_plot, DPL_plot, HPL_plot, WPL_plot_sorted, DPL_plot_sorted, HPL_plot_sorted = CreateLoadCurve(YPL)


calc_LOLE_MCS(HPL_plot,10000)
#calc_EENS_MCS(HPL_plot,10000)