"""
File: Reliability_indices.py
Author: Julian Wuijts
Date: 05-07-2024
Description: A Python script to obtain the reliability indices LOLP, LOLE and EENS.
"""

import pandas as pd 

from Load_curves import CreateLoadCurve
from COPT_derated_states import calc_COPT


def calc_LOLP(Gen_outage_capacity,P_outage,load):
    Installed_generation = Gen_outage_capacity[-1]
    Max_outage = Installed_generation - load
    j = 0
    if Max_outage <= 0 :
        P= 1
    elif Max_outage in Gen_outage_capacity:
        P_index = Gen_outage_capacity.index(Max_outage)
        P = P_outage[P_index]
    elif Max_outage > max(Gen_outage_capacity):
        P = 0
    else:
        for j in range(len(Gen_outage_capacity)-1):
            if Gen_outage_capacity[j] <= Max_outage < Gen_outage_capacity[j+1]:
                P = P_outage[j+1]
                break
            else:
                print('Something went wrong!')
    return P

def calc_LOLE(Gen_outage_capacity,P_outage, load):
    i = 0
    j = 0
    P_list = []
    Installed_generation = Gen_outage_capacity[-1]
    for i in range(len(time)):
        Max_outage = Installed_generation - (load[i])
        if Max_outage <= 0 :
            P= 1
        elif Max_outage in Gen_outage_capacity:
            P_index = Gen_outage_capacity.index(Max_outage)
            P = P_outage[P_index+1]
        elif Max_outage > max(Gen_outage_capacity):
            P = 0
        else:
            for j in range(len(Gen_outage_capacity)-1):
                if Gen_outage_capacity[j] <= Max_outage < Gen_outage_capacity[j+1]:
                    P = P_outage[j+1]
                    break
                else:
                    pass
        P_list.append(P)
    LOLE = sum(P_list)
    return LOLE

def calc_EENS(Gen_outage_capacity,P_individual, load):
    i = 0
    j = 0
    EENS = 0
    #EENS_list = []
    Installed_generation = Gen_outage_capacity[-1]
    for i in range(len(time)):
        E_sum = 0
        for j in range(len(Gen_outage_capacity)):
            Capacity_outage = (Gen_outage_capacity[j] -(Installed_generation - load[i]))
            if Capacity_outage > 0:
                E_individual = Capacity_outage*P_individual[j]
                E_sum += E_individual
            else:
                pass
        EENS += E_sum
    return EENS

generator_data = pd.read_excel("/Users/jswui/Desktop/COPT_RBTS.xlsx", sheet_name="Generators") #Import generator data from Excel
YPL_RBTS = 185
YPL_RTS = 2850

YPL = YPL_RBTS

Gen_outage_capacity, P_outage = calc_COPT(generator_data)
time, YPL_plot, WPL_plot, DPL_plot, HPL_plot, WPL_plot_sorted, DPL_plot_sorted, HPL_plot_sorted = CreateLoadCurve(YPL)

P_outage2 = P_outage
P_outage2.append(0)
P_individual = [P_outage2[i - 1] - P_outage2[i] for i in range(1, len(P_outage2))]


# Calculate LOLP
# P = calc_LOLP(Gen_outage_capacity,P_outage,YPL)
# print('\nLOLP', P,'\n')

# Calculate LOLE
LOLE_HPL = calc_LOLE(Gen_outage_capacity,P_outage,HPL_plot)
print('\nLOLE HPL',LOLE_HPL,'hours/year \n')
print('\nLOLP dash HPL',LOLE_HPL/8736,'\n')

# LOLE_DPL = calc_LOLE(Gen_outage_capacity,P_outage,DPL_plot)/24
# print('\nLOLE DPL',LOLE_DPL,'days/year \n')

LOLE_YPL = calc_LOLE(Gen_outage_capacity,P_outage,YPL_plot)
print('\nLOLE YPL',LOLE_YPL,'hours/year \n')
print('\nLOLE YPL',LOLE_YPL/24,'days/year \n')
print('\nLOLP dash YPL',LOLE_YPL/8736,'\n\n')

# Calculate EENS
EENS_HPL = calc_EENS(Gen_outage_capacity,P_individual, HPL_plot)
print('\nEENS HPL', EENS_HPL, 'MWh / year\n')
print('\nENDS HPL', EENS_HPL/8736,'\n')

EENS_YPL = calc_EENS(Gen_outage_capacity,P_individual, YPL_plot)
print('\nEENS YPL', EENS_YPL, 'MWh / year\n')
print('\nENDS YPL', EENS_YPL/8736,'\n')
