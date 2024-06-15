"""
File: COPT_derated_states.py
Author: Julian Wuijts
Date: 05-07-2024
Description: A Python script to create a COPT from a list of generators.
"""

#Importing packages
import pandas as pd 
import numpy as np 
from tabulate import tabulate

 
def remove_duplicates(dupl):         #Function for removing duplicate values in the capacity outage array
    return list(dict.fromkeys(dupl))

def calc_COPT(generator_data):

    number_of_generators = int(sum(pd.to_numeric(generator_data.iloc[1], errors='coerce').notna())/2) # Count the number of capacity outages and probabilities that are defined in a row, and divide by 2 to obtain the number of generators

    # Define the indices for the capacity and probability in the generator data variable
    select_capacity = [(2 * n)-1 for n in range(1,number_of_generators+1)] 
    select_probability = [(2 * n) for n in range(1,number_of_generators+1)]

    # Create a matrix with the possbile capacity states the generators. There is one generator per row and one generator capacity state per column
    Gen_capacity_matrix = generator_data.iloc[1:,select_capacity].T
    Gen_capacity_matrix = Gen_capacity_matrix.reset_index(drop=True)
    Gen_capacity_matrix.columns = range(Gen_capacity_matrix.shape[1]) # Reset the indices of both the columns and rows to start from zero

    # Create a matrix with the probabilities of being in a certain capacity state. The probabilities for one generator are given in a row. 
    Gen_unavailability_matrix = generator_data.iloc[1:,select_probability].T
    Gen_unavailability_matrix = Gen_unavailability_matrix.reset_index(drop=True)
    Gen_unavailability_matrix.columns = range(Gen_unavailability_matrix.shape[1]) # Reset the indices of both the columns and rows to start from zero

    #Initialise lists for the COPT recursive algorithm
    Gen_outage_capacity = []            
    Gen_outage_capacity_update =[]

    Old_gen_outage_capacity = [0]
    Gen_outage_capacity = [0]
    P_old_list_prev_gen = []

    for row in Gen_capacity_matrix.index:          # Go through all generators one by one
        for column in Gen_capacity_matrix.columns: # Go through all capacity states of one generator
            Gen_outage_capacity_update_temp = [x+Gen_capacity_matrix.loc[row,column] for x in Gen_outage_capacity] # Add the capacity of the current generator outage state to all previous total capacity outages
            Gen_outage_capacity_update = Gen_outage_capacity_update + Gen_outage_capacity_update_temp              # Append the new total capacity outages for the current generator outage state to the end of a list with the previous total capacity outages
        Gen_outage_capacity = Gen_outage_capacity + Gen_outage_capacity_update                                     # Append all the new total capacity outages for one generator to a list with the previous total capacity outages
        Gen_outage_capacity = remove_duplicates(Gen_outage_capacity)                                               # Remove duplicate capacity outage values from the list
        Gen_outage_capacity.sort()                                                                                 # Sort the capacity outage values in ascending order

        P_new_list = [] # Initalise the list with the cumulative probabilities for the COPT

        for i in range(len(Gen_outage_capacity)): # Go through all outage states for the generator that is added to the COPT

            # Initialise the lists for temporarily storing probability values
            P_new = []
            P_old_list = []
            P_new_sum = 0

            for j in range(len(Gen_unavailability_matrix.columns)): # Go through all capacity states of one generator    
                Capacity_outage_minus_new_gen = [Gen_outage_capacity[i] - x for x in Gen_capacity_matrix.values[row]] # Create a list with the generation surplus / deficit

                if Capacity_outage_minus_new_gen[j] <= 0 : # If x_j - g_i is smaller than 0, the cumulative probability is one (improve this explanation)
                    P_old = 1
                elif Capacity_outage_minus_new_gen[j] in Old_gen_outage_capacity:                 # If x_j - g_i matches a capacity outage level from the previous iteration,  
                    P_old_index = Old_gen_outage_capacity.index(Capacity_outage_minus_new_gen[j]) # assign the probability from the previous iteration to the current
                    P_old = P_old_list_prev_gen[P_old_index]
                elif Capacity_outage_minus_new_gen[j] > max(Old_gen_outage_capacity): # If x_j - g_i is larger than the largest capacity outage from the previous iteration,
                    P_old = 0                                                         # the cumulative probability is zero
                else: # If x_j - g_i is in between outage capacity values, it gets the probability from the upper outage capacity value
                    for k in range(len(Old_gen_outage_capacity)-1):
                        if Old_gen_outage_capacity[k] <= Capacity_outage_minus_new_gen[j] < Old_gen_outage_capacity[k+1]:
                            P_old = P_old_list_prev_gen[k+1]
                            break
                        else:
                            P_old = 0  

                P_old_list.append(P_old)  # Add the probability for the current capacity outage state to a list
                P_new_temp = Gen_unavailability_matrix.values[row,j] * P_old_list[j] # Mulitply the probability of the capacity outage state with the (un)availability of the current generator capacity state
                P_new.append(P_new_temp)  # Add the new probability to a list

            P_new_sum = np.sum(P_new)     # Sum all probabilities for one outage state
            P_new_list.append(P_new_sum)  # Add the probability for one outage state to a list with all outage states

        #Update the outage capacity and probability lists by converting the new list to the old list    
        Old_gen_outage_capacity = Gen_outage_capacity 
        P_old_list_prev_gen = P_new_list
    return Gen_outage_capacity, P_new_list

if __name__ == "__main__": # Only print these graphs if the COPT_derated_states script is ran


    ## UPDATE THE PATH BELOW TO WHERE YOUR FILE IS LOCATED. USE / INSTEAD OF THE \ GENERATED BY WINDOWS ##
    generator_data = pd.read_excel("/Users/jswui/Desktop/COPT_example.xlsx", sheet_name="Generators") #Import generator data from Excel


    Gen_outage_capacity, P_new_list = calc_COPT(generator_data)

    COPT_table_headers = ['Generator Outage Capacity [MW]','Cumulative probability']
    P_new_list_rounded = ["{:.5f}".format(x) for x in P_new_list]
    COPT = list(zip(Gen_outage_capacity,P_new_list_rounded))
    COPT_format = tabulate(COPT, headers=COPT_table_headers, tablefmt="grid", numalign="center",stralign="center")

    print('\n\n')
    print(COPT_format)

    # This command saves the COPT as a .txt file to the current folder. It can be uncommented if the table is too large to print in the terminal

    # with open("COPT_table.txt", "w") as f: 
    #     print(COPT_format, file=f)