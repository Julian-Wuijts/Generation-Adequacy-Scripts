"""
File: COPT_wind.py
Author: Julian Wuijts
Date: 05-07-2024
Description: A Python script to create a COPT for wind turbines.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from tabulate import tabulate

def remove_duplicates(dupl):         #Function for removing duplicate values in the capacity outage array
    return list(dict.fromkeys(dupl))

def CreatePowerProfile(wind_speed,Turbine_power_profile_excel):
    
    #Extract the turbine power profile data from the excel file
    Turbine_power_profile = Turbine_power_profile_excel.to_numpy()
    Turbine_power_profile = Turbine_power_profile[:,1:]
    P_rated = Turbine_power_profile[0,1]
    V_cut_in = float(Turbine_power_profile[1,1])
    V_rated = float(Turbine_power_profile[2,1])
    V_cut_out = float(Turbine_power_profile[3,1])

    #Calculate the constants for creating the power profile
    A = 1/(V_cut_in-V_rated)**2 * (V_cut_in*(V_cut_in+V_rated)-4*V_cut_in*V_rated*((V_cut_in+V_rated)/(2*V_rated))**3)
    B = 1/(V_cut_in-V_rated)**2 * (4*(V_cut_in+V_rated)*((V_cut_in+V_rated)/(2*V_rated))**3-(3*V_cut_in+V_rated))
    C = 1/(V_cut_in-V_rated)**2 * (2-4*((V_cut_in+V_rated)/(2*V_rated))**3)

    P_output_list = []

    for i in wind_speed:
        if i < V_cut_in:
            P_output = 0
        elif V_cut_in <= i < V_rated:
            P_output = (A+B*i+C*(i)**2)*P_rated
        elif V_rated <= i < V_cut_out:
            P_output = P_rated
        else:
            P_output = 0
        P_output_list.append(P_output)

    return P_output_list, P_rated, wind_speed

def combine_wind_mechanical_outage (all_states, number_of_turbines, output_states, turbine_power_profile_probability_test):
    #Enter the power outage and mechanical outage from 0 MW / turbines to the maximum MW / number of turbines
    multi_state_COPT =[]
    for k, value1 in enumerate(all_states): 
        probability_all_turbines =[]
        for i in range(number_of_turbines+1):
            result_sum = 0
            probability_one_turbine_list = []
            for j in range(number_of_wind_output_states):
                result = 0
                result_temp =0
                if i == 0:
                    result = turbine_power_profile_probability_test[j] 
                else:
                    valid_state_check = (all_states[k]/i - output_states[j])
                    if k == 0:
                        if j == 0:
                            result = turbine_power_profile_probability_test[j]
                    else:
                        if valid_state_check >= 0:
                            result_temp = turbine_power_profile_probability_test[j]
                result_sum+=result_temp
                probability_one_turbine = result*individual_prob[i]
                probability_one_turbine_list.append(probability_one_turbine)

            probability_one_turbine_sum = result_sum*individual_prob[i]
            probability_one_turbine_list_sum = sum(probability_one_turbine_list)
            probabilities_summed = probability_one_turbine_sum+probability_one_turbine_list_sum
            probability_all_turbines.append(probabilities_summed)
                
        multi_state_COPT.append(sum(probability_all_turbines))

    multi_state_COPT_individual_temp = [0] + multi_state_COPT
    multi_state_COPT_individual = [multi_state_COPT_individual_temp[i+1] - multi_state_COPT_individual_temp[i] for i in range(len(multi_state_COPT_individual_temp)-1)]
    return multi_state_COPT_individual

def Apportioning (Apportioned_states, all_states_outage, Outage_multi_state_COPT_individual):

    #Initialising the apportioning variables
    Apportioned_probability_lower = 0 
    Apportioned_probability_upper = 0
    Apportioned_list_lower_outage = []
    Apportioned_list_upper_outage = []
    Apportioned_list = [0] *len(Apportioned_states)


    #print('COPT individual probabilities', Outage_multi_state_COPT_individual)

    for index_i, i in enumerate(Apportioned_states):
        count = 0
        for index_j, j in enumerate(all_states_outage):
            if index_i < len(Apportioned_states)-1:
                if i == j and count == 0:
                    Apportioned_probability_lower_temp = Outage_multi_state_COPT_individual[index_j]
                    Apportioned_probability_upper_temp = 0
                    count = 1
                elif Apportioned_states[index_i] < j < Apportioned_states[index_i+1]:
                    # Using the apportioning formula
                    Apportioned_probability_lower_temp = Outage_multi_state_COPT_individual[index_j] * ((Apportioned_states[index_i+1]-j) / (Apportioned_states[index_i+1]-Apportioned_states[index_i]))
                    Apportioned_probability_upper_temp = Outage_multi_state_COPT_individual[index_j] * ((j-Apportioned_states[index_i]) / (Apportioned_states[index_i+1]-Apportioned_states[index_i]))
                else:
                    Apportioned_probability_lower_temp = 0
                    Apportioned_probability_upper_temp = 0
                Apportioned_probability_lower+=Apportioned_probability_lower_temp
                Apportioned_probability_upper+=Apportioned_probability_upper_temp
            else:
                if i == j:
                    Apportioned_probability_lower_temp = Outage_multi_state_COPT_individual[index_j]
                    Apportioned_probability_upper_temp = 0
                Apportioned_probability_lower+=Apportioned_probability_lower_temp
                Apportioned_probability_upper+=Apportioned_probability_upper_temp
        Apportioned_list[index_i]+=Apportioned_probability_lower
        if index_i < len(Apportioned_states)-1:
            Apportioned_list[index_i+1]+=Apportioned_probability_upper 

    Apportioned_list_cumulative_outage = Apportioned_list
    Apportioned_list_individual_temp_outage = [0] + Apportioned_list_cumulative_outage
    Apportioned_list_individual_outage = [Apportioned_list_individual_temp_outage[i+1] - Apportioned_list_individual_temp_outage[i] for i in range(len(Apportioned_list_individual_temp_outage)-1)]

    return Apportioned_list_individual_outage, Apportioned_list_cumulative_outage

def Windplots(time, wind_speed, P_output_list, P_output_test):
    # Plot the wind speed distribution
    plt.figure(1)
    plt.hist(wind_speed, bins=25, edgecolor='black',density=True)
    plt.title('Wind speed probability distribution for winds of the coast of Trondheim in 2019')
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Probability')

    # Plot the wind speed chronologically
    plt.figure(2)
    plt.plot(time, wind_speed, label='Wind speeds')
    plt.xlabel('Time [hours]')
    plt.ylabel('Wind Speed [m/s]')
    plt.title('Wind speeds of the coast of Trondheim in 2019')

    # Plot the power output probability distribution individually or cumulatively
    plt.figure(3)
    plt.hist(P_output_list, bins=11, edgecolor='black',density=True) # Plot the individual probabilites
    #plt.hist(P_output_list, bins=20, density=True, cumulative=True) # Plot the cumulative probabilities
    plt.xlabel('Output Power [MW]')
    plt.ylabel('Probability')
    plt.title('Power output probability distribution for a turbine of the coast of Trondheim in 2019')

    # Plot the power output of a turbine chronologically
    plt.figure(4)
    plt.plot(time, P_output_list, label='Power output')
    plt.xlabel('Time [hours]')
    plt.ylabel('Power [MW]')
    plt.title('Power output for the DTU 10 MW reference turbine of the coast of Trondheim in 2019')

    # Plot the power curve for a wind turbine
    plt.figure(5)
    plt.plot(wind_speed_test, P_output_test, label='Power output')
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Power Output [MW]')
    plt.title('Power output profile for the DTU 10 MW reference wind turbine')

    plt.show()
    return

time = list(range(1,8761))
############### Specify wind turbine generator data ###############

number_of_wind_output_states = 5
FOR = 0.04
number_of_turbines = 10
P_rated = 2 # in MW
turbine_power_profile_probability_test = [0.50897,0.24450,0.11688,0.05944,0.07021] #5

####### Construct the mechanical outage probability table ########
if __name__ == "__main__":
    availability = 1 - FOR
    x = range(number_of_turbines+1)
    cumulative_prob = binom.cdf(x, number_of_turbines,availability)
    individual_prob = binom.pmf(x, number_of_turbines,availability)

    individual_prob_outage = individual_prob[::-1]

    Installed_capacity = P_rated*number_of_turbines
    Capacity_outage = np.linspace(0,Installed_capacity,number_of_turbines+1)

    Mechanical_outage_headers = ['Capacity Outage [MW]','Individual probability']
    individual_prob_outage_rounded = ["{:.5f}".format(x) for x in individual_prob_outage]
    Mechanical_outage_table = list(zip(Capacity_outage,individual_prob_outage_rounded))
    Mechanical_outage_table_format = tabulate(Mechanical_outage_table, headers=Mechanical_outage_headers, tablefmt="grid", numalign="center",stralign="center")

    print('Mechanical outage probability table \n')
    print(Mechanical_outage_table_format)

    print('\n mechanical availability',individual_prob*100,'\n')


output_states = np.linspace(0,P_rated,number_of_wind_output_states)

all_states = []
for i in range(1,number_of_turbines+1):
    for j in range(len(output_states)):
        possible_state=i*output_states[j]
        all_states.append(possible_state)

all_states = remove_duplicates(all_states)


all_states.sort()
all_states_outage = all_states
all_states_outage = [number_of_turbines*P_rated - x for x in all_states_outage]
all_states_outage.sort()

Wind_speed_excel = pd.read_excel("/Users/jswui/Desktop/Thesis codes/Analytical/COPT_wind.xlsx", sheet_name="Wind_speed")
Wind_speed_test = pd.read_excel("/Users/jswui/Desktop/Thesis codes/Analytical/COPT_wind.xlsx", sheet_name="Wind_speed_power_curve")
Turbine_power_profile_excel = pd.read_excel("/Users/jswui/Desktop/Thesis codes/Analytical/COPT_wind.xlsx", sheet_name="Turbine_power_profile")

#Extract the wind speeds from the excel file
wind_speed = Wind_speed_excel.to_numpy()
wind_speed = wind_speed[np.ix_(np.arange(2, wind_speed.shape[0]),[1])]
wind_speed = wind_speed.flatten()

wind_speed_test = Wind_speed_test.to_numpy()
wind_speed_test = wind_speed_test[2:,1]
wind_speed_test = wind_speed_test.flatten()



if __name__ == "__main__":

    P_output_list, P_rated, wind_speed = CreatePowerProfile(wind_speed,Turbine_power_profile_excel)
    P_output_test, P_rated, wind_speed_test = CreatePowerProfile(wind_speed_test,Turbine_power_profile_excel)

    # Go from a list of power outputs to a state probability table

    State_probability_list = []

    for j, value in enumerate(output_states):
        counter = 0
        for k in P_output_list:
            if output_states[j] < output_states[-1]:
                if output_states[j] <= k < output_states[j+1]: 
                    counter+=1
            elif output_states[j] == output_states[-1] and k == output_states[-1]:
                    counter+=1
        State_probability = counter/len(P_output_list)  
        State_probability_list.append(State_probability)


multi_state_COPT_individual = combine_wind_mechanical_outage(all_states, number_of_turbines, output_states, turbine_power_profile_probability_test)

Output_table_headers = ['Capacity \n Output \n[MW]',' Individual\n probability']
multi_state_COPT_individual_rounded = ["{:.5f}".format(x) for x in multi_state_COPT_individual]
Output_table = list(zip(all_states,multi_state_COPT_individual_rounded))
Output_table_format = tabulate(Output_table, headers=Output_table_headers, tablefmt="grid", numalign="center",stralign="center")

print('\n Capacity Output Probability Table \n')
print(Output_table_format)

Outage_table_headers = ['Capacity Outage [MW]','Individual probability']
Outage_multi_state_COPT_individual = multi_state_COPT_individual[::-1]
Outage_multi_state_COPT_individual_rounded = ["{:.5f}".format(x) for x in Outage_multi_state_COPT_individual]
Outage_table = list(zip(all_states_outage,Outage_multi_state_COPT_individual_rounded))
Outage_table_format = tabulate(Outage_table, headers=Outage_table_headers, tablefmt="grid", numalign="center",stralign="center")

print('\n Capacity Outage Probability Table\n')
print(Outage_table_format)

# Specify the states to reduce the COPT to 
Apportioned_states = [0,5,10,15,20]

Apportioned_list_individual_outage, Apportioned_list_cumulative_outage = Apportioning(Apportioned_states, all_states_outage, Outage_multi_state_COPT_individual)

# Create the apportioned COPT
Outage_table_reduced_headers = ['Capacity Outage [MW]','Individual probability']
Apportioned_list_individual_outage_rounded = ["{:.5f}".format(x) for x in Apportioned_list_individual_outage]
Outage_table_reduced = list(zip(Apportioned_states,Apportioned_list_individual_outage_rounded))
Outage_table_reduced_format = tabulate(Outage_table_reduced, headers=Outage_table_reduced_headers, tablefmt="grid", numalign="center",stralign="center")

print('\n Apportioned COPT\n')
print(Outage_table_reduced_format)

Windplots(time, wind_speed, P_output_list, P_output_test)
