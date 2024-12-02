"""
File: Load_curves.py
Author: Julian Wuijts
Date: 05-07-2024
Description: A Python script to create the IEEE load curves.
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def CreateLoadCurve(YPL):
    #Import the load data from excel
    WPL_excel = pd.read_excel("/Users/jswui/Desktop/Load_curves.xlsx", sheet_name="WPL")
    DPL_excel = pd.read_excel("/Users/jswui/Desktop/Load_curves.xlsx", sheet_name="DPL")
    HPL_excel = pd.read_excel("/Users/jswui/Desktop/Load_curves.xlsx", sheet_name="HPL")

    time = list(range(1,8737)) # Create a list with hourly increments for the duration of a year (52 weeks * 7 days * 24 hours = 8736 hours)

    ############### YPL ###############

    YPL_plot = YPL * np.ones(8736) # Create hourly increments for the YPL model


    ############### WPL ###############

    #Adapt the WPL data from excel and store them in a list
    WPL = WPL_excel.to_numpy()
    WPL = WPL[np.ix_([1],np.arange(1, WPL.shape[1]))]
    WPL = WPL.flatten()
    WPL_list = []

    for i in range(len(WPL)):                    # Add hourly increments to each WPL value (7 days * 24 hours = 168 hours)
        WPL_list.extend(WPL[i]*np.ones(168))      
        
    WPL_plot = [x * YPL / 100 for x in WPL_list] # Divide WPL_list by 100 to get a percentage and multiply it with the YPL to get the absolute value
    WPL_plot_sorted = np.sort(WPL_plot)[::-1]    # Order the WPL_plot list in descending order for the Load Duration Curve


    ############### DPL ###############

    # Adapt the DPL data from excel and store them in a list
    DPL = DPL_excel.to_numpy()
    DPL = DPL[np.ix_([1],np.arange(1, DPL.shape[1]))]
    DPL = DPL.flatten()
    DPL_list_temp= []
    DPL_list = []

    for j in range(len(DPL)): # Duplicate DPL 24 times for each day to obtain a week of load data with hourly increments
        DPL_list_temp.extend(list(DPL[j]*np.ones(24)))

    for k in range(52):       # Duplicate DPL_list_temp 52 times to go from a week to a year
        DPL_list.extend(DPL_list_temp)

    DPL_plot = np.multiply(WPL_plot, DPL_list) / 100 # Divide DPL_list by 100 to get a percentage and multiply it with the WPL_plot values to get the absolute value
    DPL_plot_sorted = np.sort(DPL_plot)[::-1]        # Order the DPL_plot list in descending order for the Load Duration Curve


    ############### HPL ###############

    # Adapt the HPL data from excel and store them in a list
    # There are separate lists for a weekday and a weekend day, as well as for the three specified season categories
    HPL = HPL_excel.to_numpy()
    HPL_winter_weekday = HPL[np.ix_([1],np.arange(3, HPL.shape[1]-3))].flatten()
    HPL_summer_weekday = HPL[np.ix_([7],np.arange(3, HPL.shape[1]-3))].flatten()
    HPL_spring_fall_weekday = HPL[np.ix_([13],np.arange(3, HPL.shape[1]-3))].flatten()

    HPL_winter_weekend = HPL[np.ix_([4],np.arange(3, HPL.shape[1]-3))].flatten()
    HPL_summer_weekend = HPL[np.ix_([10],np.arange(3, HPL.shape[1]-3))].flatten()
    HPL_spring_fall_weekend = HPL[np.ix_([16],np.arange(3, HPL.shape[1]-3))].flatten()

    # Initialising the lists with the hourly data for a week
    HPL_winter = []
    HPL_summer = []
    HPL_spring_fall = []
    

    for l in range(7): # Create variables that contain the data of a whole week by combining 5 weekdays and 2 weekend days in a new variable
        if l < 5:
            HPL_winter.extend(list(HPL_winter_weekday))
            HPL_summer.extend(list(HPL_summer_weekday))
            HPL_spring_fall.extend(list(HPL_spring_fall_weekday))
        else:
            HPL_winter.extend(list(HPL_winter_weekend))
            HPL_summer.extend(list(HPL_summer_weekend))
            HPL_spring_fall.extend(list(HPL_spring_fall_weekend))

    # Initialising the list with the hourly data for a year
    HPL_list = []

    for m in range(52): # Check which seasons corresponds to the current week number and add the hourly data for a week to the list for a 
        if m < 8 or 43 <= m < 52:
            HPL_list.extend(HPL_winter)
        elif 17 <= m < 30:
            HPL_list.extend(HPL_summer)
        elif  8 <= m < 17 or 30 <= m < 43:
            HPL_list.extend(HPL_spring_fall)
        else:
            print('something went wrong!') # A check to ensure that all 52 weeks are assigned

    HPL_plot = np.multiply(DPL_plot, HPL_list) / 100 # Divide HPL_list by 100 to get a percentage and multiply it with the DPL_plot values to get the absolute value
    HPL_plot_sorted = np.sort(HPL_plot)[::-1]
    return time, YPL_plot, WPL_plot, DPL_plot, HPL_plot, WPL_plot_sorted, DPL_plot_sorted, HPL_plot_sorted

if __name__ == "__main__": # Only print these graphs if the Load_curves script is ran

    #Define the Yearly Peak Load for the test systems
    YPL_RBTS = 185
    YPL_RTS = 2850

    #Choose which one of the two test systems should be used by uncommenting that entry
    YPL = YPL_RBTS 
    #YPL = YPL_RTS

    time, YPL_plot, WPL_plot, DPL_plot, HPL_plot, WPL_plot_sorted, DPL_plot_sorted, HPL_plot_sorted = CreateLoadCurve(YPL)

    plt.figure(1)
    plt.plot(time, YPL_plot, label='YPL')
    plt.plot(time, WPL_plot, label='WPL')
    plt.plot(time, DPL_plot, label='DPL')
    plt.xlabel('Time [hours]')
    plt.ylabel('Load [MW]')
    plt.title('Load Curves for the RBTS')
    plt.title('IEEE load curves in Per-unit')
    plt.legend()

    plt.figure(2)
    plt.plot(time, YPL_plot, label='YPL')
    plt.plot(time, WPL_plot_sorted, label='WPL')
    plt.plot(time, DPL_plot_sorted, label='DPL')
    plt.plot(time, HPL_plot_sorted, label='HPL')
    plt.xlabel('Time [hours]')
    plt.ylabel('Load [MW]')
    plt.title('Load Duration Curves for the RBTS in descending order')
    plt.title('IEEE Load Duration Curves in Per-unit') 
    plt.legend()
    plt.show()