# title: "Run the SCE-UA Algorithm for calibration of the run_BRIN_model"
# author: "J. Arturo Torres-Matallana(1); Chenyao Yang(2)"
# organization (1): Luxembourg Institute of Science and Technology (LIST)
# organization (2): ... (UTAD)
# date: 12.02.2022 - 03.03.2022

import time
import os
import getpass
#simport random
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
######################################################################################################################################################################
class Timer():
    # This timer class can be ignored as it is only for test not planned to be implemented
    def start(self):
        print(f"[BENCHMARK] Start Time - {datetime.now()}")
        self._start_time = time.perf_counter()

    def _duration(self):
        duration = timedelta(seconds=time.perf_counter() - self._start_time)
        print(f"[BENCHMARK] Total Duration - {duration}")

    def end(self):
        self._duration()
        print(f"[BENCHMARK] End Time - {datetime.now()}")
######################################################################################################################################################################
def mkdir(dir = None):
  '''
  Creates the given directory.

  Parameters
  ----------
  dir : char
           Directory
  '''
  if not dir is None:
    if not os.path.exists(dir):
      os.makedirs(dir)      
######################################################################################################################################################################  
def my_sce_ua(d_ini, path_data, path_obs, cv_i,**kwargs):
    # importing  all the functions defined in func.py
    #from func import f_data_in, f_subset, f_data_obs, f_subset_obs, f_subset_2    
    folder_output_base = os.path.basename(path_obs)
    folder_output_base = os.path.splitext(folder_output_base)
    folder_output_base = folder_output_base[0]
    # Define output folder
    folder_output = os.path.join(os.path.dirname(path_obs),folder_output_base)
    # 'Create output folder
    try:
        mkdir(folder_output)
    except OSError as error:
        print(error)

    # # ' Setup input data filepaths
    # data_file = path_data
    #
    # # ' Reading dataset Remich
    # # here \s+ for sep means any one or more white space character
    # data = f_data_in(dat_file=data_file, dat_sep=',', dat_day='day', dat_month='month', dat_year='Year')
    # # print(type(data))
    # # print(data.head())
    #
    # # ' Subset data by date
    # if len(d_ini) == 2:
    #     data_subset = f_subset(dat=data, date_ini=d_ini[0], date_end=d_ini[1])
    #     # print(type(data_subset))
    #     # print(data_subset)
    # elif len(d_ini) == 4:
    #     data_subset = f_subset_2(dat=data, date_ini=d_ini[0], date_end=d_ini[1], date_ini_2=d_ini[2], date_end_2=d_ini[3])
    #
    # # ' Export data_subset as csv file
    # data_subset.to_csv(folder_output + '/tmp_data_subset_' + str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) + '.csv')
    #
    # # ' Plotting time series
    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(16, 9))
    # plt.plot(data_subset['date'], data_subset['tas'], linewidth=0.5)
    # # plt.show()
    #
    # # show current monitor dpi https://www.infobyip.com/detectmonitordpi.php
    # plt.savefig(folder_output + '/tas_' + str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) + '.png', dpi=175)
    #
    # # Read observational data
    # data_file_obs = path_obs
    #
    # obs = f_data_obs(dat_file=data_file_obs, dat_sep=',')
    # print(obs)
    # print(data_subset)
    #
    # year = data_subset['Year'].unique() define for generic 'jahr' i.e 'year' instead
    #
    # # obs_subset = f_subset_obs(dat=obs, year_ini=np.int64(d_ini[0:4]), year_end=np.int64(d_end[0:4]))
    # obs_subset = f_subset_obs(dat=obs, year_ini=year[0], year_end=year[len(year) - 1])
    # print(obs_subset)
    # obs_bbch09 = obs_subset[["Years", "budburst.DOY."]]
    # print(obs_bbch09)
    # obs_bbch09.to_csv(folder_output + '/tmp_obs_' + str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) + '.csv')
    assert "model_choice" in kwargs.keys(), "the model choice is not specified"
    model_choice = kwargs["model_choice"]

    f(d_ini, folder_output, path_data, path_obs, cv_i+1, model_selection = model_choice)
######################################################################################################################################################################
# 1. User-specific input
if getpass.getuser() == 'Clim4Vitis':
    script_drive = "H:\\"
    #shape_path = r"H:\Grapevine_model_GridBasedSimulations_study4\shapefile"
elif getpass.getuser() == 'Admin':
    script_drive = "G:\\"
# Define the main directory depending on the device
main_dir = os.path.join(script_drive,"Grapevine_model_VineyardR_packages_SCE_implementations") 
os.chdir(os.path.join(main_dir, "Pyphenology_SCE_UA", "calibration_run")) # Change to the implementation directory where all scripts are ready to be called  
from sce_ua import f

# cv_list = list(np.arange(5,30+5,5)) # Define the potential variability in suited parameter values
cv_list = list(np.arange(10,50+10,10)) # Test in a later fashion 
gridded_climate_path = os.path.join(main_dir, "gridded_climate_data") # ("./calibration_test/Lisboa region_no_NAN.csv") # Path to weather data
study_stage = "Flowering"
study_varieties = ["TF", "TN"]
phenolog_ob_summary = os.path.join(main_dir, "phenology_data", "Douro", "observations.xlsx") # Path to phenology data in Douro
phenolog_ob_df = pd.read_excel(phenolog_ob_summary, header=0, sheet_name=study_stage) # Read the phenology data into df

#phenology_ob_varieties = phenolog_ob_df.loc[phenolog_ob_df["Sites"]=="Lisboa","Varieties"].unique() # Determine number of unique varieties to study
# boundary_year = 1990 # Define the starting year that should be discarded since the weather data in the site does not match. Case-study specific
# Get a list of varieties for the calibration
# phenology_ob_varieties = [variety for variety in phenology_ob_varieties if "Touriga Nacional" not in variety] # TN needs to be excluded as the current calibration framework does not suppor uncontinuous datasets
model_list = ["classic_GDD", "GDD_Richardson", "wang", "triangular", "sigmoid"]

for cv_i in cv_list:
    for phenology_model_choice in model_list:
        time_start = datetime.now()
        time_start_str = time_start.strftime("%Y-%m-%d %H:%M:%S")
        print("Start time =" + time_start_str + "for the model {} under CV experiment {}%".format(phenology_model_choice, str(cv_i)))
        out_dir = os.path.join(main_dir, "Pyphenology_SCE_UA", "calibration_run", str(cv_i), phenology_model_choice)
        mkdir(out_dir)
        for phenology_ob_variety in study_varieties:
            # Define the path to output phenology observation that correspond to a given variety in Lisbon vineyard
            path_obs1 = os.path.join(out_dir, "sample_data_{}.csv".format(phenology_ob_variety)) 
            # Load the phenology observation data for the vineyard located in Lisbon that correspond to a certain variety
            variety_full_data = phenolog_ob_df.loc[:, np.logical_or(phenolog_ob_df.columns.str.contains(phenology_ob_variety, regex=False), 
                                                                    phenolog_ob_df.columns.isin(["Year"]))]
            # Load the subset of phenology observational data for a given variety
            variety_phenology_data =  variety_full_data.loc[:, ["Year","Plots_{}".format(phenology_ob_variety), phenology_ob_variety]]
            # Export the subset of variety phenology data into local disk
            variety_phenology_data.to_csv(path_obs1,na_rep=-999,index=False,index_label=False)
            # # Define an random number to name the output
            # id_cpu = random.randint(100, 999)
            # Define the starting and ending year
            #if variety_budburst_data["Year"].min() == boundary_year: # The begining year needs to be skipped since the weather data does not match
            start_year =  str(int(variety_phenology_data["Year"].min()))
            # Define the end year that corresponds to the study period
            end_year = str(int(variety_phenology_data["Year"].max()))
            # Define the calibration time period according to observed study years that will be supplied into the list
            d_ini = ["{}-01-01".format(start_year), "{}-12-31".format(end_year)] # Only for naming purpose
            # Start optimization of parameters
            my_sce_ua(d_ini, gridded_climate_path, path_obs1, cv_i, model_choice=phenology_model_choice)
            # Print the computation end time
            time_end = datetime.now()
            time_end_str = time_end.strftime("%Y-%m-%d %H:%M:%S")
            print("Start time =" + time_end_str + "for the model {} under CV experiment {}%".format(phenology_model_choice, str(cv_i)))
            # Compute the total time costed
            time_elapsed = time_end - time_start
            print("Elapsed time is " + str(time_elapsed) + "for the model {} under CV experiment {}%".format(phenology_model_choice, str(cv_i)))
