# Generation-Adequacy-scripts
 Open-Source scripts for Generation Adequacy analysis using an anlalytical method and a non-sequential MCS method can be found in this repository

The mentioned packages can all be installed using "pip install *package*"

The following scripts can be found in this repository:

## Analytical scripts ##

# COPT_derated
description: Takes the Excel file "COPT_example.xlsx" as an input, with the generator capacity outage levels and state probabilities to obtain the COPT of a generation system.
 
 required packages: -pandas
                    -numpy
                    -tabulate
                    -openpyxl

# COPT_wind
description: Takes the Excel file "COPT_wind.xlsx" as an input, with three sheets: 'wind_speed' with the wind data from a site, 'wind_speed_power_curve' with wind speeds between 0 and 30 m/s to plot the turbine power profile and 'turbine_power_profile' with the turbine characteristics. A COPT for a wind generation system can be obtained, and it can be reduced using the apportioning method.
 
 required packages: -pandas
                    -numpy
                    -tabulate
                    -scipy
                    -matplotlib
                    -openpyxl

# Load_Curves
description: Takes the Excel file "Load_curves.xlsx" as an input, which has the weekly, daily and hourly peak load percentages for the IEEE load curve. The output of the script is 1. a chronological load curve and 2. a load duration curve, with the loads sorted in descending order.
 
 required packages: -pandas
                    -numpy
                    -matplotlib
                    -openpyxl

# Reliability_indices
description: Takes the Excel file "COPT_RBTS.xlsx" as an input to create a COPT. It is then combined with a load profile to obtain reliability indices. The COPT_derated  and Load_curves scripts need to be in the same folder as the Reliability_indices file in order to import them.

 required packages: -pandas
                    -openpyxl

## non-sequential MCS scripts ##

# MCS_state_sampling_derated
description: Takes the Excel file "RBTS_MCS_wind_state_sampling_derated.xlsx" to create a generation profile for the whole system for one year with hourly increments. If wind generation is present in the system, the CreatePowerProfile function from the Analytical scripts should be imported and present in the same folder.

 required packages: -pandas
                    -numpy
                    -matplotlib
                    -openpyxl

# MCS_state_sampling_no_derated
description: Takes the Excel file "RBTS_MCS_wind_state_sampling_derated.xlsx" to create a generation profile for the whole system for one year with hourly increments. If wind generation is present in the system, the CreatePowerProfile function from the Analytical scripts should be imported and present in the same folder. The advantage of the MCS_state_sampling_no_derated script over the MCS_state_sampling_derated script is the reduced computation time compared for a system without derated states.

 required packages: -pandas
                    -numpy
                    -tabulate
                    -openpyxl

# Load_Curves
description: Takes the Excel file "Load_curves.xlsx" as an input, which has the weekly, daily and hourly peak load percentages for the IEEE load curve. The output of the script is 1. a chronological load curve and 2. a load duration curve, with the loads sorted in descending order.
 
 required packages: -pandas
                    -numpy
                    -matplotlib
                    -openpyxl

# Reliability_indices_MCS
description: Takes the Excel file "RBTS_MCS_wind_state_sampling.xlsx" as an input to create a generation profile for each iteration. It is then combined with a load profile to obtain reliability indices. The MCS_state_sampling(_no)_derated and Load_curves scripts need to be in the same folder as the Reliability_indices file in order to import them.

 required packages: -pandas
                    -numpy
                    -matplotlib
                    -openpyxl
