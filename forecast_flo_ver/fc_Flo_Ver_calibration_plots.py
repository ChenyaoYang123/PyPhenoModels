# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:06:06 2022

@author: Chenyao
"""
import os
import os.path
import pandas as pd
import numpy as np
import getpass
from os.path import join,isdir,dirname
import sys
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
# Define the disk drive letter according to the username
if getpass.getuser() == 'Clim4Vitis':
    script_drive = "H:\\"
    #shape_path = r"H:\Grapevine_model_GridBasedSimulations_study4\shapefile"
elif getpass.getuser() == 'Admin':
    script_drive = "G:\\"
elif (getpass.getuser() == 'CHENYAO YANG') or (getpass.getuser() == 'cheny'):
    script_drive = "D:\\"
target_dir1 = r"Mega\Workspace\Study for grapevine\Study6_Multi_phenology_modelling_seasonal_forecast\script_collections" # Specific for a given study
target_dir2 = r"Mega\Workspace\Programming_resources\Python\Functions" # Specific for a given study
###############################################################################
def add_script(script_drive, target_dir):
    '''
    Add script stored in the target directory to the system path

    Parameters
    ----------
    target_dir : str or path-like object,  a string path to the target disk path where the multi-model phenology classess are stored
    
    '''
    #target_dir = r"Mega\Workspace\Study for grapevine\Study6_Multi_phenology_modelling_seasonal_forecast\script_collections" # Specific for a given study
    script_path = join(script_drive, target_dir)
    sys.path.append(script_path)
    sys.path.append(dirname(script_path))  
################################################################################################
add_script(script_drive, target_dir1)
add_script(script_drive, target_dir2)
from Multi_phenology_model_classes import * # Add the multi-phenology model class script
from utilities import get_nearest_hundreds
################################################################################################
# 1. Define user-specified path
main_dir = join(script_drive, dirname(target_dir1), "PMP_sigmoid_run") # Main workspace path
ob_data = join(main_dir, "summary_phenology.csv") # Phenology observation path
weather_data = join(main_dir, "weather.txt") # Phenology observation path
calibration_par_path = join(main_dir, "Calibrated_parameters_target.xlsx") # Calibrated parameter summary
################################################################################################
# 2. Load the weather data
weather_data_df = pd.read_csv(weather_data, sep="\s+", header=0, index_col= False) # Load the weather df
# 2.1 Create the datetime index column
datetime_index = weather_data_df[["Year","DOY"]].apply(lambda x: datetime.strptime(f'{x.Year}-{x.DOY}','%Y-%j'), axis = 1, 
                                          raw = False, result_type = "reduce") 
# 2.2 Extract the underlying column names for each target meteorologicval variable
Tmin_name = [col_name for col_name in weather_data_df.columns if "Tmin" in col_name][0]
Tmean_name = [col_name for col_name in weather_data_df.columns if "Tmean" in col_name][0]
Tmax_name = [col_name for col_name in weather_data_df.columns if "Tmax" in col_name][0]
# 2.3 Obtain the climate series for each variables
T_min = pd.Series(weather_data_df[Tmin_name].values, index = datetime_index, name= "tasmin")  # Ensure the input data is a series
T_mean = pd.Series(weather_data_df[Tmean_name].values, index = datetime_index, name= "tasmean")  # Ensure the input data is a series
T_max = pd.Series(weather_data_df[Tmax_name].values, index = datetime_index, name= "tasmax")  # Ensure the input data is a series
################################################################################################
# 3. Load the calibated parameter settings and observational datasets
calibration_par_TN = pd.read_excel(calibration_par_path, sheet_name = "TN_settings")
calibration_par_TF = pd.read_excel(calibration_par_path, sheet_name = "TF_settings")
# 3.1 Read target parameter settings for each cultivar
Flo_pars = ["SStar<1>", "d_flo", "e_flo"] # Define a list of sigmoid parameters for flowering simulation
Ver_pars = ["SStar<2>", "d_ver", "e_ver"] # Define a list of sigmoid parameters for veraison simulation
# 3.2 Obtain the calibrated parameter sets of each cultivar for both flowering and veraison stage predictions
target_pars = {} 
for var_name, calibration_par_df in {"TN":calibration_par_TN, "TF":calibration_par_TF}.items():
    if var_name not in target_pars.keys():
        target_pars[var_name] = {}
    for target_par in Flo_pars + Ver_pars +["RMSE"]: #Add the RMSE row
        par_row = calibration_par_df.loc[calibration_par_df["Model_parameters"]==target_par, :].dropna(axis=1, how="any")
        par_row_renamed = par_row.rename(columns={col_name:"Repetition_" + str(rep_No+1) for col_name, rep_No in # Rename the columns
                                                    zip(par_row.columns[1:], range(len(par_row.columns[1:])))})
        target_pars[var_name][target_par] = par_row_renamed
# 3.3 Write the target parameters into local .csv file
# 3.3.1 Concatenate parameter df into one object
TN_concat = pd.concat(list(target_pars["TN"].values()), axis= 0, join='inner', ignore_index=False)
TF_concat = pd.concat(list(target_pars["TF"].values()), axis= 0, join='inner', ignore_index=False)
# 3.3.2 Filter out the parameter values so that only those with minimum RMSE are exported
TN_concat.set_index("Model_parameters", inplace=True)
TF_concat.set_index("Model_parameters", inplace=True)
# 3.3.3 Obtain the best-fit parameters (with minimum RMSE) for both varieties
TN_concat_bestfit = TN_concat.iloc[:, pd.Series(TN_concat.loc["RMSE", :].values, dtype=float).argmin()]
TF_concat_bestfit = TF_concat.iloc[:, pd.Series(TF_concat.loc["RMSE", :].values, dtype=float).argmin()]
# 3.3.4 Export the subset df into local disk
TN_concat_bestfit.to_csv(join(main_dir,"TN_para.csv"))
TF_concat_bestfit.to_csv(join(main_dir,"TF_para.csv"))
# 3.4 Load observational datasets
ob_data_phenology = pd.read_csv(ob_data, ",", header=0) # Read the phenology datasets
ob_data_phenology_lisbon = ob_data_phenology.loc[ob_data_phenology["Sites"]=="Lisboa"] # Subset the phenology dataset that only corresponds to the Lisbon dataset                                
################################################################################################
# 4. Run simulation with the best-fit parameter set for each variety and make the respective scatter plot with observation
variety_dict = {"TF": "Touriga Francesa", "TN": "Touriga Nacional"} # Create a dict that contains two variety names
fig, axe = plt.subplots(len(variety_dict), len(variety_dict), figsize=(9,9)) # Create the plot instances
# Iterate over each variety
for index, (var_short_name, var_full_name) in enumerate(variety_dict.items()):
    # Obtain the variety-specific observed phenology data 
    ob_subset_data_phenology = ob_data_phenology_lisbon.loc[ ob_data_phenology_lisbon["Varieties"].str.contains(var_full_name, regex=False) ]
    ob_subset_data_phenology = ob_subset_data_phenology.iloc[:, 1:] # Filter the dataset by removing the first column
    # Compute the calibration repetition with the min RMSE
    min_RMSE_position = pd.Series(np.array(target_pars[var_short_name]["RMSE"].iloc[0, :])[1:], dtype=float).argmin()
    
    
    # Extract the flowering stage parameter set
    flo_thermal = target_pars[var_short_name]["SStar<1>"].set_index("Model_parameters")
    flo_d = target_pars[var_short_name]["d_flo"].set_index("Model_parameters")
    flo_e = target_pars[var_short_name]["e_flo"].set_index("Model_parameters")
    # Extract the veraison stage parameter set
    ver_thermal = target_pars[var_short_name]["SStar<2>"].set_index("Model_parameters")
    ver_d = target_pars[var_short_name]["d_ver"].set_index("Model_parameters")
    ver_e = target_pars[var_short_name]["e_ver"].set_index("Model_parameters")
    # Run the model with prescribed parameter sets for a given cultivar    
    # Subset the meteorological data to be consistent with the observed data of a given variety
    # Subset_Tmean = T_mean.index.year.isin(np.arange(ob_subset_data_phenology["Years"].min(), ob_subset_data_phenology["Years"].max()+1, 1))
    # T_mean_input = T_mean.loc[Subset_Tmean].copy(deep=True)
    # Run the model for flowering stage with presribed input (subset temperature and calibrated parameters)
    flo_SM = phenology_model_run(T_mean.copy(deep=True), thermal_threshold= flo_thermal.iloc[0, min_RMSE_position], module="sigmoid", 
                                        a = flo_d.iloc[0, min_RMSE_position], b = flo_e.iloc[0,min_RMSE_position], 
                                        DOY_format=True, from_budburst=False, T0=1) # Parameter value is adopted whenever it has the lowest RMSE
    flo_SM.name = "Flo_SM" # Re-define the series name 
    # Obtain the observed flowering series
    flo_OB = pd.Series(ob_subset_data_phenology["flowering(DOY)"].values, index= ob_subset_data_phenology["Years"], name= "Flo_OB")
    # Concatenate two series data
    flo_pair = pd.concat([flo_OB, flo_SM], axis= 1, join="inner")
    # Run the model for veraison stage with presribed input (subset temperature and calibrated parameters)
    ver_SM = phenology_model_run(T_mean.copy(deep=True), thermal_threshold= ver_thermal.iloc[0, min_RMSE_position], module="sigmoid", 
                                        a = ver_d.iloc[0,min_RMSE_position], b = ver_e.iloc[0,min_RMSE_position], 
                                        DOY_format=True, from_budburst=False, T0=flo_SM)
    ver_SM.name = "Ver_SM" # Re-define the series name 
    # Obtain the observed veraison series
    ver_OB = pd.Series(ob_subset_data_phenology["veraison(DOY)"].values, index= ob_subset_data_phenology["Years"], name= "Ver_OB")
    # Concatenate two series data with overlapping year only
    ver_pair = pd.concat([ver_OB, ver_SM], axis= 1, join="inner")
    # Create a paired group variable
    pairs_OB_SM = [flo_pair, ver_pair]
    pairs_OB_SM_name = ["flowering_stage", "veraison_stage"]
    # pLot for each two axe per variety
 
        
    for axe_instance, OB_SM_df, variable_name in zip(target_axe_list, pairs_OB_SM, pairs_OB_SM_name):
        # Get the phenology stage in current loop
        if "flowering" in variable_name:
            pheno_stage = "Flo"
            # The boundary value is inferred from two varieties over two stages
            min_bound = 100
            max_bound = 170
        elif "veraison" in variable_name:
            pheno_stage = "Ver"
            min_bound = 170
            max_bound = 240
        # Get the observational and simulational series
        OB =  OB_SM_df["{}_OB".format(pheno_stage)].copy(deep=True)
        SM =  OB_SM_df["{}_SM".format(pheno_stage)].copy(deep=True)
        # Make the scatter plot
        axe_instance.scatter(x = OB, y = SM, marker='o',s=25, facecolors='none',edgecolors='black')
        # Get the minimum and maximum boundary values for the plot
        # min_bound = get_nearest_hundreds(round(np.nanmin(np.concatenate((OB, SM)))))-10
        # max_bound = get_nearest_hundreds(round(np.nanmax(np.concatenate((OB, SM)))))+10
        # Set the x-axis and y-axis boundary values
        axe_instance.set_xlim(left = min_bound, right = max_bound) # Set the x and y limit in the same scale
        axe_instance.set_ylim(bottom = min_bound, top= max_bound)
        # tick_interval =(max(axe_instance.get_xticks()) - min(axe_instance.get_xticks()))/(len(axe_instance.get_xticks())-1) 
        if np.array_equal(axe_instance.get_xticks(), axe_instance.get_yticks()):
            ticks = axe_instance.get_xticks() # With equal ticks in both x- and y-axis, adopt the x-axis ticks
        else:
            raise "x-axis and y-axis ticks are not euqal for the scatter plot"
        # Set up the axis ticks
        axe_instance.set_xticks(ticks)
        axe_instance.set_yticks(ticks)
        # Set up the axis tick parameters
        axe_instance.tick_params(axis='x', length=5, labelsize=8, pad=3)
        axe_instance.tick_params(axis='y', length=5, labelsize=8, pad=3)
        # Set up the plot title
        axe_instance.set_title(var_short_name+ "_" +variable_name, fontdict = {'fontsize':10}, loc = 'center', pad=10, fontweight="bold")
        axe_instance.plot(axe_instance.get_xlim(), axe_instance.get_ylim(), ls="--", c=".3") # Add the diagonal line
        # Make the necessary statistical calculations between OB and SM
        MBE_calculation = round(np.nanmean(OB) - np.nanmean(SM)) # Mean biased error
        MAE_calculation = round(mean_absolute_error(OB, SM)) # Mean absolute error
        MSE_calculation = round(mean_squared_error(OB, SM)) # Mean squared error 
        R_square = round((np.corrcoef(OB, SM)[0][1])**2,2)  # Pearson correlation coefficient
        EF = round(r2_score(OB, SM),2)
         # Place relavant statiscal metrics for each subplot
        axe_instance.text(0.9,0.25,"MBE="+str(MBE_calculation), fontsize=8, color='blue', horizontalalignment='center', transform = axe_instance.transAxes)
        axe_instance.text(0.9,0.2,"MAE="+str(MAE_calculation), fontsize=8, color='blue', horizontalalignment='center', transform = axe_instance.transAxes)
        axe_instance.text(0.9,0.15,"RMSE="+str(int(np.sqrt(MSE_calculation))), fontsize=8, color='blue', horizontalalignment='center',transform = axe_instance.transAxes)
        axe_instance.text(0.9,0.1,"R2="+str(R_square), fontsize=8, color='blue', horizontalalignment='center', transform = axe_instance.transAxes)
        axe_instance.text(0.9,0.05,"EF="+str(EF), fontsize=8, color='blue', horizontalalignment='center', transform = axe_instance.transAxes)

# Save the figure into the local disk
fig.savefig(join(main_dir,"calib_scatter.png"), bbox_inches="tight",pad_inches=0.05,dpi=600)
plt.close()

# 5. Run simulations with the best-fit parameter set for each variety and make the respective line plot with observation
variety_dict = {"TF": "Touriga Francesa", "TN": "Touriga Nacional"} # Create a dict that contains two variety names
# 5.1 Collect simulation series for all calibrated parameter sets
var_sm_flo = {} # Create an empty dictionary to store flowering simulation results for each variety
var_sm_ver = {} # Create an empty dictionary to store veraison simulation results for each variety
for var_short_name in target_pars.keys():
    # Obtain the variety-specific observed phenology data 
    ob_subset_data_phenology = ob_data_phenology_lisbon.loc[ ob_data_phenology_lisbon["Varieties"].str.contains(variety_dict[var_short_name], regex=False) ]
    ob_subset_data_phenology = ob_subset_data_phenology.iloc[:, 1:] # Filter the dataset by removing the first column
    # Obtain the calibration df
    var_df = target_pars[var_short_name].copy()
    RMSE_series = pd.Series(np.array(var_df["RMSE"].iloc[0, :])[1:], dtype=float)
    if var_short_name not in var_sm_flo.keys():
        var_sm_flo[var_short_name]={}
    if var_short_name not in var_sm_ver.keys():
        var_sm_ver[var_short_name]={}
    for repetition_run in range(len(RMSE_series)):
        # Extract the flowering stage parameter set
        flo_thermal = var_df["SStar<1>"].set_index("Model_parameters")
        flo_d = var_df["d_flo"].set_index("Model_parameters")
        flo_e = var_df["e_flo"].set_index("Model_parameters")
        # Extract the veraison stage parameter set
        ver_thermal = var_df["SStar<2>"].set_index("Model_parameters")
        ver_d =var_df["d_ver"].set_index("Model_parameters")
        ver_e = var_df["e_ver"].set_index("Model_parameters")
        # Run the model with prescribed parameter sets for a given cultivar    
        # Subset the meteorological data to be consistent with the observed data of a given variety
        # Subset_Tmean = T_mean.index.year.isin(np.arange(ob_subset_data_phenology["Years"].min(), ob_subset_data_phenology["Years"].max()+1, 1))
        # T_mean_input = T_mean.loc[Subset_Tmean].copy(deep=True)
        # Run the model for flowering stage with presribed input (subset temperature and calibrated parameters)
        flo_SM = phenology_model_run(T_mean.copy(deep=True), thermal_threshold= flo_thermal.iloc[0, repetition_run], module="sigmoid", 
                                            a = flo_d.iloc[0, repetition_run], b = flo_e.iloc[0,repetition_run], 
                                            DOY_format=True, from_budburst=False, T0=1) # Parameter value is adopted whenever it has the lowest RMSE
        # Run the model for veraison stage with presribed input continued from the flowering stage
        ver_SM = phenology_model_run(T_mean.copy(deep=True), thermal_threshold= ver_thermal.iloc[0, repetition_run], module="sigmoid", 
                                            a = ver_d.iloc[0,repetition_run], b = ver_e.iloc[0,repetition_run], 
                                            DOY_format=True, from_budburst=False, T0=flo_SM)
        # Rename the flowering and veraison stage
        flo_SM.name = "Flo_SM" # Re-define the series name 
        ver_SM.name = "Ver_SM" # Re-define the series name
        # Obtain the observed flowering series
        flo_OB = pd.Series(ob_subset_data_phenology["flowering(DOY)"].values, index= ob_subset_data_phenology["Years"], name= "Flo_OB")
        # Concatenate two series data
        flo_pair = pd.concat([flo_OB, flo_SM], axis= 1, join="inner")
        # Only extract simulation series that match the observed data
        flo_SM = flo_pair["Flo_SM"].copy(deep=True)

        # Obtain the observed veraison series
        ver_OB = pd.Series(ob_subset_data_phenology["veraison(DOY)"].values, index= ob_subset_data_phenology["Years"], name= "Ver_OB")
        # Concatenate two series data
        ver_pair = pd.concat([ver_OB, ver_SM], axis= 1, join="inner")
        # Only extract simulation series that match the observed data
        ver_SM = ver_pair["Ver_SM"].copy(deep=True)
        
        # Attach the simulation series to the target dictionary
        if repetition_run == RMSE_series.argmin():  # if the current loop equals to position with the minimum RMSE
            var_sm_flo[var_short_name][str(repetition_run)+"_minRMSE"] = flo_SM.copy(deep=True)
            var_sm_ver[var_short_name][str(repetition_run)+"_minRMSE"] = ver_SM.copy(deep=True)
        else:
            var_sm_flo[var_short_name][str(repetition_run)] = flo_SM.copy(deep=True)
            var_sm_ver[var_short_name][str(repetition_run)] = ver_SM.copy(deep=True)

# 5.2 Collect the target simulation series to be plotted on the line plot
var_sm_flo_plot = {}
var_sm_ver_plot = {}
for (var_name_flo, var_flo_data_dict), (var_name_ver, var_ver_data_dict) in zip(var_sm_flo.items(), var_sm_ver.items()):
    # Compute the 5th percentile value for both varieties
    var_flo_data_5th = pd.Series(np.nanpercentile(list(var_flo_data_dict.values()), 5, 0), index = np.unique([ser.index for ser in list(var_flo_data_dict.values())]))
    var_ver_data_5th = pd.Series(np.nanpercentile(list(var_ver_data_dict.values()), 5, 0), index = np.unique([ser.index for ser in list(var_ver_data_dict.values())]))
    # Compute the median values for both varieties
    var_flo_data_minRMSE = [flo_ser for flo_rep, flo_ser in var_flo_data_dict.items() if flo_rep == "5_minRMSE"][0]
    var_ver_data_minRMSE = [ver_ser for ver_rep, ver_ser in var_ver_data_dict.items() if ver_rep == "5_minRMSE"][0]
    # Compute the 95th percentile values for both varieties
    var_flo_data_95th = pd.Series(np.nanpercentile(list(var_flo_data_dict.values()), 95, 0), index = np.unique([ser.index for ser in list(var_flo_data_dict.values())]))
    var_ver_data_95th = pd.Series(np.nanpercentile(list(var_ver_data_dict.values()), 95, 0), index = np.unique([ser.index for ser in list(var_ver_data_dict.values())]))
    # Attach results to target dictionary
    if var_name_flo not in var_sm_flo_plot.keys():
        var_sm_flo_plot[var_name_flo] = {}
    var_sm_flo_plot[var_name_flo]["5th"] = var_flo_data_5th.copy(deep=True)
    var_sm_flo_plot[var_name_flo]["minRMSE"] = var_flo_data_minRMSE.copy(deep=True)
    var_sm_flo_plot[var_name_flo]["95th"] = var_flo_data_95th.copy(deep=True)
    if var_name_ver not in var_sm_ver_plot.keys():
        var_sm_ver_plot[var_name_ver] = {}
    var_sm_ver_plot[var_name_ver]["5th"] = var_ver_data_5th.copy(deep=True)
    var_sm_ver_plot[var_name_ver]["minRMSE"] = var_ver_data_minRMSE.copy(deep=True)
    var_sm_ver_plot[var_name_ver]["95th"] = var_ver_data_95th.copy(deep=True)

# 5.3 Make the line plot between observation series and simulation band
variety_dict = {"TF": "Touriga Francesa", "TN": "Touriga Nacional"} # Create a dict that contains two variety names
fig, axe = plt.subplots(len(variety_dict), len(variety_dict), figsize=(9,4), tight_layout=True) # Create the plot instances
plot_col_dict = {}
# Iterate over each variety
for index, (var_short_name, var_full_name) in enumerate(variety_dict.items()):
    ob_subset_data_phenology = ob_data_phenology_lisbon.loc[ ob_data_phenology_lisbon["Varieties"].str.contains(variety_dict[var_short_name], regex=False) ]
    ob_subset_data_phenology = ob_subset_data_phenology.iloc[:, 1:] # Filter the dataset by removing the first column
    # Obtain the observed flowering series for a given variety 
    flo_OB = pd.Series(ob_subset_data_phenology["flowering(DOY)"].values, index= ob_subset_data_phenology["Years"], name= "Flo_OB")
    # Obtain the observed veraison series for a given variety 
    ver_OB = pd.Series(ob_subset_data_phenology["veraison(DOY)"].values, index= ob_subset_data_phenology["Years"], name= "Ver_OB")
    # Extract the simulation series for the flowering and veraison stage
    SM_series_flo = var_sm_flo_plot[var_short_name]["minRMSE"]
    SM_series_ver = var_sm_ver_plot[var_short_name]["minRMSE"]
    # Infer study years
    # study_years = np.arange(min(flo_OB.index), max(flo_OB.index), 1) if np.array_equal(flo_OB.index,ver_OB.index) else None# use yearly timeseries for the flowering stage
    study_years = list(flo_OB.index) if np.array_equal(flo_OB.index,ver_OB.index) else None
    for plot_index, axe_instance in enumerate(axe[index, :]): # Plot over each row of subplots
        # Plot for the flowering stage
        if plot_index == 0: # Ensure the first column of first row corresponds to the line plot for observations
            axe_instance.plot(study_years, flo_OB, ls="dashed", lw= 0.5, color="black", marker = "o", ms = 5, mfc = "black", mec = "black", mew = 0.5)
                              #marker='o', mfc =scenario_color, mec='black', markersize=3)
            # Plot for the simulated data by parameters with the lowest RMSE
            axe_instance.plot(study_years, SM_series_flo, ls="dashed", lw= 0.5, color="black",
                              marker = "o", mfc = "none", ms = 4, mec = "black", mew = 0.5)
            # Add the errobar for simulation in each year
            
            # axe_instance.errorbar(study_years, SM_series_flo, fmt="o", yerr= np.vstack((SM_series_flo-var_sm_flo_plot[var_short_name]["5th"], var_sm_flo_plot[var_short_name]["95th"]-SM_series_flo)),
            #                       capsize=2,ecolor="black",elinewidth=1.5,
            #                         marker = "o", mfc = "none", ms = 5, mec = "black", mew = 0.5)#capsize=5, ecolor="black", elinewidth="0.5") # Note here the height is customized empirically
                                  # marker=symbol_marker, mfc = symbol_fc_baseline, mec="black",
                                  # ms=4)
            # Fill the band between the lower (5th) and upper (95th) percentile simulations
            # axe_instance.fill_between(study_years, var_sm_flo_plot[var_short_name]["5th"], var_sm_flo_plot[var_short_name]["95th"], # Provided the x position, ymax, ymin positions to fill 
            #                  facecolor = "grey", # The fill color
            #                  color = "black",   # The outline color
            #                  edgecolors = "white", # The line edge color
            #                  linewidth =1,
            #                  alpha=0.4)
            axe_instance.set_ylabel("Flowering stage (DOY)",fontsize=6, rotation=90,labelpad = 5)
            axe_instance.set_ylim(90, 180, auto=False)
            # Set the subplot labels on mean and std for both observed and simulated series
            axe_instance.text(0.15,0.15,"Mean_OB="+str(round(np.nanmean(flo_OB))), fontsize=4, fontstyle="italic", color='black', horizontalalignment='center', transform = axe_instance.transAxes)
            axe_instance.text(0.15,0.1,"Std_OB="+str(round(np.nanstd(flo_OB))), fontsize=4, fontstyle="italic", color='black', horizontalalignment='center', transform = axe_instance.transAxes)
            axe_instance.text(0.9,0.15,"Mean_SM="+str(round(np.nanmean(SM_series_flo))), fontsize=4, fontstyle="italic", color='black', horizontalalignment='center', transform = axe_instance.transAxes)
            axe_instance.text(0.9,0.1,"Std_SM="+str(round(np.nanstd(SM_series_flo))), fontsize=4, fontstyle="italic", color='black', horizontalalignment='center', transform = axe_instance.transAxes)
        else: # Plot for the veraison stage
            axe_instance.plot(study_years, ver_OB, ls="dashed", lw= 0.5, color="black", marker = "o", ms = 5, mfc = "black", mec = "black", mew = 0.5)
                              #marker='o', mfc =scenario_color, mec='black', markersize=3)
            # Plot for the simulated data by parameters with the lowest RMSE
            axe_instance.plot(study_years, SM_series_ver, ls="dashed", lw= 0.5, color="black",
                              marker = "o", mfc = "none", ms = 4, mec = "black", mew = 0.5)
            # Add the errobar for simulation in each year
            # axe_instance.errorbar(study_years, SM_series_ver, fmt="o", yerr= np.vstack((SM_series_ver-var_sm_ver_plot[var_short_name]["5th"], var_sm_ver_plot[var_short_name]["95th"]-SM_series_ver)),
            #                   capsize=2,ecolor="black",elinewidth=1.5,
            #                     marker = "o", mfc = "none", ms = 5, mec = "black", mew = 0.5)#capsize=5, ecolor="black", elinewidth="0.5")
        
            # axe_instance.errorbar(study_years, var_sm_ver_plot[var_short_name]["minRMSE"], yerr= np.vstack((var_sm_ver_plot[var_short_name]["5th"], var_sm_ver_plot[var_short_name]["95th"])),
            #                  capsize=5, ecolor="black", elinewidth="0.5")
            # Fill the band between the lower (5th) and upper (95th) percentile simulations
            # axe_instance.fill_between(study_years, var_sm_ver_plot[var_short_name]["5th"], var_sm_ver_plot[var_short_name]["95th"], # Provided the x position, ymax, ymin positions to fill 
            #                  facecolor = "grey", # The fill color
            #                  color = "black",   # The outline color
            #                  edgecolors = "white", # The line edge color
            #                  linewidth =1,
            #                  alpha=0.4)
            axe_instance.set_ylabel("Veraison stage (DOY)",fontsize=6, rotation=90,labelpad = 5)
            axe_instance.set_ylim(170, 240, auto=False)
            # Set the subplot labels on mean and std for both observed and simulated series
            axe_instance.text(0.15,0.15,"Mean_OB="+str(round(np.nanmean(ver_OB))), fontsize=4, fontstyle="italic", color='black', horizontalalignment='center', transform = axe_instance.transAxes)
            axe_instance.text(0.15,0.1,"Std_OB="+str(round(np.nanstd(ver_OB))), fontsize=4, fontstyle="italic", color='black', horizontalalignment='center', transform = axe_instance.transAxes)
            axe_instance.text(0.9,0.15,"Mean_SM="+str(round(np.nanmean(SM_series_ver))), fontsize=4, fontstyle="italic", color='black', horizontalalignment='center', transform = axe_instance.transAxes)
            axe_instance.text(0.9,0.1,"Std_SM="+str(round(np.nanstd(SM_series_ver))), fontsize=4, fontstyle="italic", color='black', horizontalalignment='center', transform = axe_instance.transAxes)
        # Add necessary plot decorations that applies to all subplots
        axe_instance.tick_params(axis='x',length=3,width=1,labelsize=6,pad=2) # Set the y-axis parameters
        axe_instance.tick_params(axis='y',length=3,width=1,labelsize=6,pad=2) # Set the y-axis parameters
        # Set the x-axis ticks and tick labels
        axe_instance.set_xticks(study_years)
        axe_instance.set_xticklabels(study_years, rotation=90)
        #axe_instance.yaxis.set_label_coords(1.2,0.5) 
        # Set the axe instance title
        axe_instance.set_title(var_full_name, fontdict= {"fontsize": 7}, loc="center", y=0.8, fontweight="bold")
# Export the figure into local disk
fig.savefig(join(main_dir,"calib_lineplot.png"), bbox_inches="tight",pad_inches=0.05,dpi=600)
plt.close()


