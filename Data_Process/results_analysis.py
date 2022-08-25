import os
import os.path
import pandas as pd
import numpy as np
import re
import copy
# import matplotlib.pyplot as plt 
# # import matplotlib.ticker as mtick
# import plotly.express as px
# import plotly.graph_objects as go
import glob
import os
import matplotlib.pyplot as plt
import skill_metrics as sm
from os.path import join
from matplotlib import rcParams
from scipy.stats import norm
from matplotlib import gridspec
from collections import OrderedDict
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
def color_choice(name): 
    from matplotlib import colors as mcolors
    colors_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS) # unpack the list into one dictionary
    color_name=colors_dict[name]
    return color_name
#####################################################################################################################################################################
def fit_gaussian(mean, std):
    '''
    Fit a normal (gaussian) distribution class instance

    Parameters
    ----------
    mean: float, the mean of the normal distribution 
    std: float, the std of the normal distribution 
    '''
    return norm(loc=mean, scale=std)
#####################################################################################################################################################################
def check_nan_ser(input_ser, fill_val = -999.00 ):
    '''
    Check if a series of values are consecutive in nature
    In case it is not consecutive in some years, filling with supplied values
    
    Parameter
    ----------
    input_ser : series, a sequence of values (int or float)
    '''
    if not isinstance(input_ser, pd.core.series.Series):
        raise TypeError("The input data does not follow a series format, the format of {} is found".format(input_ser.__class__)) 
    input_ser_copy = input_ser.copy(deep=True)
    if len(input_ser_copy.index) != len(np.arange(input_ser_copy.index.min(), input_ser_copy.index.max() +1 ,1)):
        # Fill missing observations in years with -999
        full_years = np.arange(input_ser_copy.index.min(), input_ser_copy.index.max() +1 ,1) # Extract the full study year series
        #variety_budburst_data.set_index("Year", inplace=True)
        input_ser_reindex = input_ser_copy.reindex(full_years, fill_value = fill_val)
        return input_ser_reindex
    else:
        return input_ser_copy
#####################################################################################################################################################################
def Make_Taylor_Diagram(Observations,Simulations,SaveFolder,Variable,RMS_ticks=np.arange(0,12+3,3),STD_ticks=np.arange(0,15+3,3),Axismax=15,MakerLegend=False): #
    '''
  Plot the taylor diagram for a set of observation series and corresponding multiple predicted series

  Mandatory Parameters
  ----------
  Observations: one-dimensional array or list for observed series of data.
  Simulations: A dictionary or list of simulation series, each series will be compared with observed series
  SaveFolder: The save folder to store the output figures.
  Variable: Studied variable for the Taylor diagram, which is mainly used for purpose of saving file
  ----------
  Optional Parameters (options to customize the Taylor diagram)
  ----------
  RMS_ticks: numpy array of RMS values to plot gridding circles from observation point
  STD_ticks: numpy array of STD values to plot gridding circles from observation point
  Axismax: maximum for the radial contours (for the maximum STD contour)
    '''
# https://github.com/PeterRochford/SkillMetrics/issues/19
    # Firstly check if observed and simulated series datatype
    # For observed series
    if isinstance(Observations,dict):
        OB=list(Observations.values())
    elif isinstance(Observations,list):
        OB=np.array(Observations)
    elif np.squeeze(np.array(Observations)).ndim>1:
        raise TypeError("The observational input should be one-dimensional array, but a multi-dimensional array is found")
    # For simulation series
    if isinstance(Simulations,(list, dict)): # Check if the input match the type specified
        SM=Simulations.copy()
    else:
        raise TypeError("The simulation input should be dictionary or list, the {} type is found".format(Simulations.__class__))
    # Computation of required statistics to create Taylor diagram
    if isinstance(SM,dict): # In case SM is a dictionary, most probably it is an ordered dictionary
        taylor_SM=list(SM.values())
        marker_labels=list(SM.keys()) # The marker labels for each simulation series are stored in a separate variable
        #marker_labels_dict= {str(index+1):marker for index, marker in enumerate(marker_labels)}
        #marker_labels_taylor=list(marker_labels_dict.keys()) # The marker labels to be used in taylor diagram
        marker_labels_taylor=["OB"]+["S"+str(number) for number in list(np.arange(1,len(marker_labels)+1,1))] # Note the first integer number is always reserved for observations
    elif isinstance(SM,list): # In case the input simulatin is a list of one-dimensional simulatoin array/list 
        taylor_SM=SM.copy()
    taylor_stats_OB = sm.taylor_statistics(taylor_SM[0],OB) # Indeed, the OB stat can be extracted from all recurring comparisons (the same), but I only choose the first prediction series to obtain the value of metric  
    # Loop over all participating simulation series where the Taylor diagram is plotted
    # Create empty lists to store computed statistics
    sdev=[]
    crmsd=[]
    ccoef=[]
    # Firstly, attached the observed statistics into the required statistical list
    sdev.append(taylor_stats_OB['sdev'][0])
    crmsd.append(taylor_stats_OB['crmsd'][0])
    ccoef.append(taylor_stats_OB['ccoef'][0])
    for SM_series in taylor_SM: # Here the taylor_SM must be a list of one-dimensional array
        taylor_stats = sm.taylor_statistics(SM_series,OB)
        sdev.append(taylor_stats['sdev'][1]) # Append the standard deviation value of each simulation series
        crmsd.append(taylor_stats['crmsd'][1]) # Append the centered RMS error value of each simulation series
        ccoef.append(taylor_stats['ccoef'][1]) # Append the Pearson correlation coefficient value of each simulation series
    # After collecting the required statistics, the Taylor diagram is to be plotted 
    plt.close('all') # Close all precedent graphic handles if any
    # Make the input statistics as numpy array instead of lists
    # Options to customize the Taylor diagram
#    RMS_ticks= np.arange(0,30,3)   # RMS values to plot gridding circles from observation point
#    STD_ticks=np.arange(0,5,35) # STD values to plot gridding circles from observation point
#    axismax=30 # Maximum for the radial contours
    if MakerLegend is True: # If the marker legend should turn on or not
        markerlegend="on"
    else:
        markerlegend="off" 
    sm.taylor_diagram(np.array(sdev),np.array(crmsd),np.array(ccoef),checkStats="on",
                      markercolor="red",alpha=0,markerSize=2,markerLegend=markerlegend,# Marker options
                      titleOBS="OB", colOBS = 'black',styleOBS = '-', widthOBS=0.5,# Observational point options
                      titleSTD="on",colSTD="black",styleSTD = '-.',widthSTD=0.5,showlabelsSTD='on',tickSTD = STD_ticks, rincSTD=STD_ticks[1]-STD_ticks[0],axismax=Axismax,# STD contour options
                      titleRMS="on",colRMS = 'green', styleRMS = ':', widthRMS=0.5,showlabelsRMS="on",tickRMS = RMS_ticks,tickRMSangle=135, titleRMSDangle=170, # RMS contour options
                      titleCOR="on",colCOR = 'blue', styleCOR = '--',widthCOR=0.5,showlabelsCOR="on") # Pearson correlation options
#    markerLabel=marker_labels_taylor, markerLabelColor = 'r'
    # sm.taylor_diagram(np.array(sdev),np.array(crmsd),np.array(ccoef),checkStats="on",
#                  markercolor="red",markerLabel=marker_labels_taylor, markerLabelColor = 'r',alpha=0,markerSize=4,markerLegend=markerlegend,# Marker options
#                  titleOBS="Observation", colOBS = 'black',styleOBS = '-', widthOBS=0.5,# Observational point options
#                  titleSTD="on",colSTD="black",styleSTD = '-.',widthSTD=0.5,showlabelsSTD='on',# STD contour options
#                  titleRMS="on",colRMS = 'green', styleRMS = ':', widthRMS=0.5,showlabelsRMS="on",tickRMSangle=135, titleRMSDangle=170, # RMS contour options
#                  titleCOR="on",colCOR = 'blue', styleCOR = '--',widthCOR=0.5,showlabelsCOR="on") # 

    # Small decorations on the Tarlor diagram
    rcParams["figure.figsize"] = [4, 3.5]
    rcParams['lines.linewidth'] = 1 # line width for plots
    rcParams.update({'font.size': 6})
    mkdir(SaveFolder)
    outputfigure=os.path.join(SaveFolder,"Taylor_Diagram_{}.pdf".format(Variable))
    plt.savefig(outputfigure)
    plt.close("all")  
#####################################################################################################################################################################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. User-specific input
main_dir = r"H:\Grapevine_model_VineyardR_packages_SCE_implementations\Pyphenology_SCE_UA\calibration_run"
os.chdir(main_dir) # Change to the current directory  
model_list = ["classic_GDD", "GDD_Richardson", "sigmoid", "triangular", "wang"] # Define a list of models in use
obj_func_col = "like1" #  The name in use for the objective function name
cv_list = list(np.arange(5,30+5,5)) # Define the potential variability in suited parameter values
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create an empty dict to create the variety-dependent model_pars_dict
# 2. Obtain the target model parameters and objective function value
final_results = {}
for cv_i in cv_list:
    var_model_pars_dict = {} # Define an empty dict to collect results on the estimated parameter values
    model_obj_dict = {}# Define an empty dict to collect results on the calculated objective function, i.e. RMSE
    for phenology_model_choice in model_list:
        out_dir = os.path.join(main_dir, str(cv_i), phenology_model_choice)
        mkdir(out_dir)
        list_folder = [item for item in os.listdir(out_dir) if ".csv" not in item]
        if (phenology_model_choice not in model_obj_dict.keys()):
            model_obj_dict[phenology_model_choice] = {}
        # Iterate over each variety
        for folder in list_folder:
            output_csv = glob.glob(join(out_dir,folder,"*_db_*.csv"))[0] # Obtain the output file path
            output_df = pd.read_csv(output_csv,  header=0, index_col=False) # Read the output csv file into df
            output_df_target = output_df.loc[output_df[obj_func_col]==min(output_df[obj_func_col]),:]
            # Obtain the underlying variety name 
            var_name = re.findall("data\w+",folder)[0].strip("data_") # Use "data" as the matching string to search for target variety name
            # Attach the key (model name) into the target dict
            model_pars_dict = {}  # Define an empty dict to collect results on the estimated parameter values
            if phenology_model_choice not in model_pars_dict.keys():
                model_pars_dict[phenology_model_choice] = {}
            # Obtain the target parameters 
            # Since multiple different values could lead to the same performance, averages of mean and std values are adopted
            par0_gaussian = fit_gaussian(np.nanmean(output_df_target["parpar0_mean"]), 
                                             np.nanmean(output_df_target["parpar0_sd"])) # Create the normal distribution based on the distribution parameters obtained
            model_pars_dict[phenology_model_choice]["par0"] =  {"mean":np.nanmean(output_df_target["parpar0_mean"]),
                                                                                  "std":np.nanmean(output_df_target["parpar0_sd"])}    #  copy.deepcopy(par0_gaussian) # Save the parameter normal distribution into target dict
            # Note par0 is the parameter that every model will have
            if phenology_model_choice == "GDD_Richardson":
                # Since multiple different values could lead to the same performance, averages of mean and std values are adopted
                # Par1
                par1_gaussian = fit_gaussian(np.nanmean(output_df_target["parpar1_mean"]) , 
                                             np.nanmean(output_df_target["parpar1_sd"])) # Create the normal distribution based on the distribution parameters obtained
                model_pars_dict[phenology_model_choice]["par1"] = {"mean":np.nanmean(output_df_target["parpar1_mean"]),
                                                                   "std":np.nanmean(output_df_target["parpar1_sd"])}  # copy.deepcopy(par1_gaussian) # Save the parameter normal distribution into target dict
                # Par3
                par3_gaussian = fit_gaussian(np.nanmean(output_df_target["parpar3_mean"]) , 
                                             np.nanmean(output_df_target["parpar3_sd"])) # Create the normal distribution based on the distribution parameters obtained
                model_pars_dict[phenology_model_choice]["par3"] =  {"mean":np.nanmean(output_df_target["parpar3_mean"]),
                                                                   "std":np.nanmean(output_df_target["parpar3_sd"])} 
                # Save the parameter normal distribution into target dict
            elif phenology_model_choice == "sigmoid":
                # Since multiple different values could lead to the same performance, averages of mean and std values are adopted
                # Par4
                par4_gaussian = fit_gaussian(np.nanmean(output_df_target["parpar4_mean"]) , 
                                             np.nanmean(output_df_target["parpar4_sd"])) # Create the normal distribution based on the distribution parameters obtained
                model_pars_dict[phenology_model_choice]["par4"] =  {"mean":np.nanmean(output_df_target["parpar4_mean"]),
                                                                   "std":np.nanmean(output_df_target["parpar4_sd"])} 
                # Save the parameter normal distribution into target dict
                # Par5
                par5_gaussian = fit_gaussian(np.nanmean(output_df_target["parpar5_mean"]) , 
                                             np.nanmean(output_df_target["parpar5_sd"])) # Create the normal distribution based on the distribution parameters obtained
                model_pars_dict[phenology_model_choice]["par5"] =  {"mean":np.nanmean(output_df_target["parpar5_mean"]),
                                                                   "std":np.nanmean(output_df_target["parpar5_sd"])} 
                # Save the parameter normal distribution into target dict
            elif (phenology_model_choice == "triangular") or (phenology_model_choice == "wang"):
                # Since multiple different values could lead to the same performance, averages of mean and std values are adopted
                # Par1
                par1_gaussian = fit_gaussian(np.nanmean(output_df_target["parpar1_mean"]) , 
                                             np.nanmean(output_df_target["parpar1_sd"])) # Create the normal distribution based on the distribution parameters obtained
                model_pars_dict[phenology_model_choice]["par1"] = {"mean":np.nanmean(output_df_target["parpar1_mean"]),
                                                                   "std":np.nanmean(output_df_target["parpar1_sd"])} 
                # Save the parameter normal distribution into target dict
                # Par2
                par2_gaussian = fit_gaussian(np.nanmean(output_df_target["parpar2_mean"]) , 
                                             np.nanmean(output_df_target["parpar2_sd"])) # Create the normal distribution based on the distribution parameters obtained
                model_pars_dict[phenology_model_choice]["par2"] =  {"mean":np.nanmean(output_df_target["parpar2_mean"]),
                                                                   "std":np.nanmean(output_df_target["parpar2_sd"])} 
                # Save the parameter normal distribution into target dict
                # Par3
                par3_gaussian = fit_gaussian(np.nanmean(output_df_target["parpar3_mean"]) , 
                                             np.nanmean(output_df_target["parpar3_sd"])) # Create the normal distribution based on the distribution parameters obtained
                model_pars_dict[phenology_model_choice]["par3"] =  {"mean":np.nanmean(output_df_target["parpar3_mean"]),
                                                                   "std":np.nanmean(output_df_target["parpar3_sd"])} 
                # Save the parameter normal distribution into target dict
            # Attach the variety-dependent results
            if var_name not in var_model_pars_dict.keys():
                var_model_pars_dict[var_name] = {}
            var_model_pars_dict[var_name].update(copy.deepcopy(model_pars_dict))
            # Obtain the minimized objective function value, i.e. RMSE
            obj_mean = np.nanmin(output_df_target[obj_func_col].unique())
            model_obj_dict[phenology_model_choice][var_name] = obj_mean
# for variety, pars_dict in var_model_pars_dict.items():
#     for model_name, par_dict in  pars_dict.items():
#         for par_name, par_dist in par_dict.items():
#             mean = par_dist["mean"]
#             std = par_dist["std"]
#             print(mean,std )
    final_results["CV" + str(cv_i) + "pars_dist"] = var_model_pars_dict.copy()
    final_results["CV" + str(cv_i) + "obj_func"] = model_obj_dict.copy()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # 3. Line plot on the parameter distribution
# save_path = join(main_dir, "results")
# mkdir(save_path)

# for variety, pars_dict in var_model_pars_dict.items():
#     for model_name, fitted_dist_dict in pars_dict.items():
#         axis_list=[] # append to this empty list a number of grid specifications
#         if len(fitted_dist_dict)==1:
#             fig, ax = plt.subplots(len(fitted_dist_dict), 1, figsize=(6,4)) # Number of rows equal to number of parameters
#             axis_list.append(ax)
#         else:
#             fig, ax = plt.subplots(len(fitted_dist_dict), 1, figsize=(6,4*len(fitted_dist_dict)))
#             for i in range(len(fitted_dist_dict)):
#                 axis_list.append(ax[i])
#         for (par_name, dist_parameters), axe_sub in zip(fitted_dist_dict.items(), axis_list):
#             # Fit the distribution 
#             par_dist = fit_gaussian(dist_parameters["mean"], 
#                                     dist_parameters["std"])
#             # Evenly sample the value from the 1th percentile to 9the 9th percentile
#             x = np.linspace(par_dist.ppf(0.01),
#                             par_dist.ppf(0.99), 100)
#             # Compute the essential parameters of the normaldistributions
#             mean_val= round(par_dist.mean(), 1)
#             std_val = round(par_dist.std(), 1)
#             P5_val = round(par_dist.ppf(0.05), 1)
#             P95_val = round(par_dist.ppf(0.95),1)
#             #print(mean_val)
#             #x = normal_fit.rvs(size=1000)
#             # Make the normal distribution plot
#             axe_sub.plot(x, par_dist.pdf(x),  c ="black", ls= '-', lw=2, alpha=0.6)
#             axe_sub.vlines(P5_val, ymin=0, ymax=par_dist.pdf(P5_val), color = 'black', ls="--", lw=1)
#             axe_sub.vlines(mean_val, ymin=0, ymax=par_dist.pdf(mean_val), color = 'black', ls="--", lw=1)
#             axe_sub.vlines(P95_val, ymin=0, ymax=par_dist.pdf(P95_val), color = 'black', ls="--", lw=1)
#             # Annotate the statistics of the distribution
#             axe_sub.text(0.2, 0.08, "5th_percentile=\n{}".format(str(P5_val)), fontsize=8, color='black', horizontalalignment='center', transform = axe_sub.transAxes)
#             axe_sub.text(0.5, 0.15, "Mean={}".format(str(mean_val)), fontsize=8, color='black', horizontalalignment='center',weight="bold", transform = axe_sub.transAxes)
#             axe_sub.text(0.5, 0.1, "std={}".format(str(std_val)), fontsize=8, color='black', horizontalalignment='center', weight="bold", transform = axe_sub.transAxes)
#             axe_sub.text(0.75, 0.08, "95th_percentile=\n{}".format(str(P95_val)), fontsize=8, color='black', horizontalalignment='center', transform = axe_sub.transAxes)
#             # Annotate the parameter 
#             axe_sub.set_title(par_name, fontdict={"fontsize":12}, loc='right', y=0.9, pad=1)
#         # Save the plot into local disk
#         variety_save_folder = join(save_path, variety)
#         mkdir(variety_save_folder)
#         fig.savefig(join(variety_save_folder, model_name+".png"),dpi=600, bbox_inches="tight", pad_inches=0.1)
#         plt.close("all")   
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 4. Bar plot on the goodness-of-fit
save_path = join(main_dir, "results")
mkdir(save_path)
# 4.1 Extract a list of prior-assumed CV
CV_keys = [cv_key for cv_key in final_results.keys() if "obj" in cv_key]
# 4.2 The goodness-of-fit plot for different CV
for CV_key in CV_keys:
    model_obj_dict = final_results[CV_key] # Access the underlying dict for a given assumed CV variation
    grid = gridspec.GridSpec(nrows = 1, ncols = len(model_obj_dict),wspace=0.15)
    axis_list=[] # append to this empty list a number of grid specifications
    for col in range(len(model_obj_dict)):
        axis_list.append(grid[0, col])      
    fig = plt.figure(figsize=(2.5*len(model_obj_dict),2)) # Create a figure instance class
    #fig, ax = plt.subplots(1, len(model_obj_dict), figsize=(2*len(model_obj_dict),3), gridspec_kw= dict(hspace = 0.1, wspace = 0.5)) 
    bar_width = 1 # Specify the bar width
    y_label = "RMSE (days)"
    y_ticks = np.arange(0, 20+5, 5) # Set a fixed y range to represent the min and maxi possible RMSE values
    # var_abbr = {"Cabernet_Sauvignon_T":"CS", "Chardonnay_B":"CD",  
    #             "Grenache_T":"GN", "Riesling_B":"RS", 
    #             "Touriga_Francesa_T":"TF", "Touriga_Nacional_T":"TN"}
    for index, ((model_name, var_dict), axe_grid) in enumerate(zip(model_obj_dict.items(), axis_list)):
        axe_sub = fig.add_subplot(axe_grid)
        para_values=list(var_dict.values()) # List of values to be used as heights in the bar plot
        para_labels=list(var_dict.keys()) # List of strings to be used as labels in the bar plot
        x = np.arange(0,len(para_values)*2,2) # Set the bar at the x-position for every 2 adjacent value
        axe_sub.bar(x, para_values, bar_width, color=color_choice("lightgrey"),edgecolor='black')
        axe_sub.set_xticks(x) # Set the x-axis ticks for every 2-adjacent sequence 
        #axe_sub.set_xticklabels([var_abbr[para_label] for para_label in para_labels],fontdict={"fontweight":"bold"}) # Label the x-axis for every 2-adjacent sequence 
        axe_sub.set_xticklabels([str(para_label) for para_label in para_labels],fontdict={"fontweight":"bold"}) # Label the x-axis
        axe_sub.set_yticks(y_ticks) # Set the x-axis ticks for every 2-adjacent sequence 
        axe_sub.set_yticklabels([str(round(y_tick,1)) for y_tick in y_ticks], fontdict={"fontweight":"bold"}) # Label the x-axis for every 2-adjacent sequence 
        axe_sub.tick_params(axis='x',length=1,labelsize=6,pad=3.5) # Set the x-axis tick parameters
        axe_sub.tick_params(axis='y',length=1,labelsize=6,pad=3.5)
        if index==0:
            axe_sub.set_ylabel(y_label, fontdict={"fontsize":6})
        #axe_sub.set_ylim(bottom=0,top=20) 
        if model_name == "classic_GDD":
            title_name = "GDD"
        else:
            title_name = model_name
        axe_sub.set_title(title_name,fontdict={'fontsize':7},loc='center',y=0.85,fontweight="bold") 
        #axe_sub.set_aspect(aspect='auto',adjustable='box')
        #plt.rcParams["font.weight"] = "bold" # set every text in bold  
    #fig.tight_layout()
    # Save the plot to the file
    save_dir = join(save_path, CV_key)
    mkdir(save_dir)
    fig.savefig(join(save_dir,"obj_summary.png"),dpi=600, bbox_inches="tight", pad_inches=0.1)
    plt.close("all")      
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
# 5 The parameter value and uncertainty plot
CV_keys = [cv_key for cv_key in final_results.keys() if "pars_dist" in cv_key]
# 5.1 Iterate over each model to collect the plotting data first
plot_dict = {} # Create an empty dict to store plotting data
for model_name in model_list:
    plot_dict[model_name] ={}
    for CV_key in CV_keys:
        if CV_key not in plot_dict[model_name].keys():
            plot_dict[model_name][CV_key] = {}
        para_dist_dict = final_results[CV_key] # Access the underlying parameter distribution dict for a given assumed CV variation
        for variety_name, para_dict in para_dist_dict.items():
            var_model_dict =  para_dict[model_name].copy() # Access the variety-model dictionary
            if variety_name not in plot_dict[model_name][CV_key].keys():
                plot_dict[model_name][CV_key][variety_name] ={}
            for para_name, para_val in var_model_dict.items():
                mean_val= para_val["mean"] # Use the mean key to access the mean parameter value
                std_val = para_val["std"]  # Use the std key to access the std parameter value
                if para_name not in plot_dict[model_name][CV_key][variety_name].keys():
                    plot_dict[model_name][CV_key][variety_name][para_name+"_mean"] = mean_val
                    plot_dict[model_name][CV_key][variety_name][para_name+"_std"] = std_val
# 5.2 Define the parameter definition dict for each model
model_parameter_dict = {"classic_GDD": ["par0"], "GDD_Richardson": ["par0", "par1", "par3"],
                        "sigmoid": ["par0", "par4", "par5"], "triangular": ["par0", "par1", "par2", "par3"],
                        "wang": ["par0", "par1", "par2", "par3"]
                        }
# parameter_def = {"par0":"Thermal Demand", "par1":"Minimum Development Temperature",
#                  "par2":"Optimum Development Temperature", "par3":"Maximum Development Temperature",
#                  "par4":"Curve Sharpness", "par5":"Mid-Response Temperature"
#                  }
parameter_def = {"par0":"CTF", "par1":"MnDT",
                 "par2":"ODT", "par3":"MxDT",
                 "par4":"CS", "par5":"MRT"
                 }
cv_list = list(np.arange(5,30+5,5)) # Define a list of potential variability in suited parameter values 
save_path = join(main_dir, "results") # Save path for the plot
mkdir(save_path)
target_stats = ["mean","std"]
max_figsize = (4, 12) # The maximum figure (width, height)
max_para = max([len(para_list) for model_name, para_list in model_parameter_dict.items()])# The maximum number of parameters from the pre-defined model
bar_width = 0.5 # Specify the bar width
# 5.3 Iterate over each model to make the bar plot on the mean and std of parameters
for target_stat in target_stats:
    writer=pd.ExcelWriter(join(save_path,"parameters_{}.xlsx".format(str(target_stat)))) # Save the mean and std into 2 different excel files
    for model_name, CV_dict_full in plot_dict.items():
        parameter_list = model_parameter_dict[model_name] # Access the underling number of parameter list
        #rows_input = len(parameter_list) # The number of subplot rows depend on the number of parameters
        # rows_all_CV = []
        # for CV_dict_name, var_para_dict in CV_dict_full.items():
        #     nrows= np.array([len(var_para_dict_val) for var_para_dict_val in var_para_dict.values()]) # The number of subplot rows depend on the number of parameters
        #     nrows = np.unique(nrows) /2 # Since every parameter is accompanied by mean and std, it is needed to divide by 2 to get the actual number of parameters
        #     rows_all_CV.append(nrows)
        # rows_input = np.unique(rows_all_CV) # A single unique number of parameter value will be determined
        # Always arrange subplots in a single column with multiple rows
        #grid = gridspec.GridSpec(nrows = int(rows_input), ncols = 1, hspace=0.1)
        fig, ax = plt.subplots(int(max_para), 1, figsize=max_figsize) # Keep all the figures with the same number of subplots
        # Create a figure instance class
        #fig = plt.figure(figsize=max_figsize) # The figure is resized based on number of parameters
        for index, (parameter, axe_sub) in enumerate(zip(parameter_list, ax.flat)): 
            #axe_sub = fig.add_subplot(axe_grid) # Create the subplot axe
            CV_var_stat = {} # Create an empty dict to collect target results to plot
            for cv_i in cv_list:
                CV_key_val = [CV_key for CV_key in CV_dict_full.keys() if str(cv_i) in CV_key][0]
                CV_var_stat[str(cv_i)] = CV_dict_full[CV_key_val].copy()
            para_labels = list(CV_var_stat.keys()) # List of strings to be used as labels in the bar plot
            para_TF_stat_dict = {int(cv_val):val_dict["TF"][parameter+ "_" + str(target_stat)] for cv_val, val_dict in CV_var_stat.items()} # Access the mean parameter values for TF
            para_TN_stat_dict = {int(cv_val):val_dict["TN"][parameter+ "_" + str(target_stat)] for cv_val, val_dict in CV_var_stat.items()} # Access the std parameter values for TF
            para_TF_stat_ser = pd.Series(para_TF_stat_dict, name= "TF_"+ parameter_def[parameter])
            para_TN_stat_ser = pd.Series(para_TN_stat_dict, name= "TN_"+ parameter_def[parameter])
            # Generate an output df to excel file
            target_df = pd.concat([para_TF_stat_ser, para_TN_stat_ser], axis=1, join="inner", ignore_index=False)
            if index==0:
                index_col = True
                index_label= "CV assumptions"
                start_col_id = 0
            else:
                index_col = False
                index_label= None
                start_col_id = 3 * index + 1
            target_df.to_excel(writer, sheet_name= model_name, header=True, startrow=0, 
                                        startcol= start_col_id, index=index_col, index_label=index_label, engine="openpyxl")
            #para_TN_mean = [val_dict["TN"][parameter+"_mean"] for val_dict in CV_var_stat.values()] # Access the mean parameter values for TF
            #para_TN_std = [val_dict["TN"][parameter+"_std"] for val_dict in CV_var_stat.values()] # Access the std parameter values for TF
            x = np.arange(0,len(para_labels)*2,2) # Set the bar at the x-position for every 2 adjacent value
            axe_sub.bar(x-0.5, para_TF_stat_ser.values, bar_width, color=color_choice("orange"),edgecolor='black',align="edge")
            axe_sub.bar(x+0.5, para_TN_stat_ser.values, -bar_width, color=color_choice("green"),edgecolor='black',align="edge")
            axe_sub.set_xticks(x) # Set the x-axis ticks for every 2-adjacent sequence 
            #axe_sub.set_xticklabels([var_abbr[para_label] for para_label in para_labels],fontdict={"fontweight":"bold"}) # Label the x-axis for every 2-adjacent sequence 
            axe_sub.set_xticklabels([str(para_label) for para_label in para_labels],fontdict={"fontweight":"bold"}) # Label the x-axis
            axe_sub.set_ylim(auto=True)
            #axe_sub.set_yticks(y_ticks) # Set the x-axis ticks for every 2-adjacent sequence 
            #axe_sub.set_yticklabels([str(round(y_tick,1)) for y_tick in y_ticks], fontdict={"fontweight":"bold"}) # Label the x-axis for every 2-adjacent sequence 
            axe_sub.tick_params(axis='x',length=1,labelsize=6,pad=3.5) # Set the x-axis tick parameters
            axe_sub.tick_params(axis='y',length=1,labelsize=6,pad=3.5) 
            if parameter=="par0":
                y_label = "Parameter values (degree-day sum)" # Label for the thermal forcing parameter
            else:
                y_label = "Parameter values (°C)" # Label for the structual parameters, e.g. base temperature or maximum limit temperature
            axe_sub.set_ylabel(y_label, fontdict={"fontsize":6})
            # Set the subplot title
            axe_sub.set_title(parameter_def[parameter],fontdict={'fontsize':7},loc='center',y=0.9,fontweight="bold") 
            if min(axe_sub.get_ylim())<0:
                axe_sub.axhline(0, color='black', lw=0.5) # Add the horizontal x-axis line 
            # change all spines line width
            for axis_spine in ['top','bottom','left','right']:
                axe_sub.spines[axis_spine].set_linewidth(0.5) # Set the lined width to 0.5
            
        if len(parameter_list) < max_para: # In case the plotting parameters are smaller than the maximum number of parameters
            axe_id_delete = np.arange(len(parameter_list), max_para,1)
            for axe_id in list(axe_id_delete):
                fig.delaxes(ax[axe_id])
        # Save the plot to the file
        fig.savefig(join(save_path,"parameters_{0}_{1}.png".format(model_name, target_stat)), dpi=600) #bbox_inches="tight", pad_inches=0.1)
        plt.close("all")
    writer.save()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
# 6 Evaluation of calibrated parameters (plots in the Taylor Diagram)
# 6.1 Pre-define essential input variables
os.chdir(main_dir) # Change to the implementation sdirectory  
from Multi_phenology_model_classes import *
# Define the parameter dictionary for the phenology models
model_parameter_dict = {"classic_GDD": ["CTF"], "GDD_Richardson": ["CTF", "MnDT", "MxDT"],
                        "sigmoid": ["CTF", "CS", "MRT"], "triangular": ["CTF", "MnDT", "ODT", "MxDT"],
                        "wang": ["CTF", "MnDT", "ODT", "MxDT"]
                        } 
# Define a dict that links the parameter abb with the actual abb  
para_abb_corresp = {"MnDT":"Tdmin", "ODT":"Tdopt", "MxDT":"Tdmax", 
                    "CS": "a", "MRT":"b"
                    } 
save_path = join(main_dir, "results") # Create the save path where target result plot are stored
grid_weather_input_file = join(main_dir, "evaluation_sce_ua", "grid_weather_input") # Define the weather input file for the Lisboa vineyard extracted from E-OBS 
phenology_OB = join(main_dir, "evaluation_sce_ua", "phenology_data") # Define the path to the observed phenology data
parameter_file = join(save_path, "parameter_summary.xlsx") # Define the path to the parameters obtained from previous executed SCE-UA
para_excel = pd.ExcelFile(parameter_file) # Inquire the target parameter file
para_excel_sheet_names = para_excel.sheet_names # A list of sheet names that should match the list of phenology models
study_variety = "Touriga Francesa" # Define the variety name or Touriga Nacional
study_variety_abb = "TF" # Define the variety name or TN
study_stage ="flowering"
rep = 10000 # Define number of repetitions/folds in the parameter sampling for the evaluation run 
tolerable_sampling = True # Define the bool variable used in the model run during repetitive parameter sampling
# 6.2 Load the weather and phenology observations into df
weather_df = pd.read_csv( glob.glob(join(grid_weather_input_file,"*csv"))[0], header=0, index_col=False)
datetime_index = pd.to_datetime(weather_df.loc[:, weather_df.columns.isin(["day", "month", "Year"])]) # Generate the datetime index
weather_df.set_index(datetime_index, inplace=True) # Set the date time index in the weather df
weather_ser_subset = weather_df["tg"] # Subset the temperature column that will be used as the weather input
phenology_ob_df = pd.read_csv( glob.glob(join(phenology_OB,"*csv"))[0], header=0, index_col=0) # Read the phenology ob csv file into df
phenology_ob_df_subset = phenology_ob_df.loc[phenology_ob_df["Sites"]=="Lisboa", # Subset the phenology ob based on the data in the Lisboa vineyard
                                             phenology_ob_df.columns.isin(["Sites", "Years","Varieties","{}(DOY)".format(study_stage)])]
phenology_ob_df_subset = phenology_ob_df_subset.loc[phenology_ob_df_subset["Varieties"].str.contains(study_variety, regex=False),:] # Subset the observed flowering data
# Access the observed phenology data for a given variety of a given stage in the Lisboa vineyard
phenology_ob_df_subset_var_stage = pd.Series(phenology_ob_df_subset["{}(DOY)".format(study_stage)].values, 
                                             index=phenology_ob_df_subset["Years"], name="OB")
# Check and fill the missing values of observations
phenology_ob_use = check_nan_ser(phenology_ob_df_subset_var_stage) # Final phenology observations that will be use throughout the evalution processs
# Subset the weather data in years where observations are available 
weather_ser_use = weather_ser_subset.loc[weather_ser_subset.index.year.isin(phenology_ob_use.index)] # mean temperature series
# 6.3 Perform the evaluation of calibrated parameter from SCE-UA based on the Lisbon vineyard data
for model_name in  para_excel_sheet_names:
    # Load the parameter excel file into the df
    para_excel_df = pd.read_excel(para_excel, sheet_name= model_name, header=0, index_col=0)
    if any(para_excel_df.isnull().any()):
        para_excel_df.dropna(axis=1,how="any", inplace=True)
    # Select a list of relevant columns for the analysis, i.el for a given variety
    sel_col = para_excel_df.loc["CV assumptions"].str.contains("{}_".format(study_variety_abb), regex=False) # CV assumptions is the fixed length characters to locate columns
    para_excel_df_subset = para_excel_df.loc[:, sel_col]
    # Access the underlying list of parameters for each model
    para_list = model_parameter_dict[model_name]
    for cv_i in cv_list: # Iterate over each cv considered
        # Establish the fitted normal distribution from the calibrated mean and std for each parameter under a given model and a variety 
        para_dist_dict = {} # Create an empty dict to store the fitted distribution of each parameter according to its calibrated mean and std
        for para_name in para_list:
            # Select the two columns that contains both the mean and std of the target parameter    
            sel_para_col = para_excel_df_subset.loc["CV assumptions"].str.contains("_{}".format(para_name), regex=False) # CV assumptions is the fixed length characters to locate columns
            # Selet column that corresponds to the mean parameter value
            para_mean = para_excel_df_subset.loc[cv_i, para_excel_df_subset.columns.str.contains("Mean",regex=False) & sel_para_col]
            # Selet column that corresponds to the std parameter value
            para_std = para_excel_df_subset.loc[cv_i, para_excel_df_subset.columns.str.contains("Std",regex=False) & sel_para_col]
            # Fit a normal distribution according to the mean and std
            para_normal_fit = fit_gaussian(float(para_mean), float(para_std)) # Copy the fitted normal distribution into the target dicts
            para_dist_dict[str(para_name)] = copy.deepcopy(para_normal_fit)
        # Run the simulations with pre-defined number of repetitions using sampled parameter values from the fitted normal distributions
        SM_taylor= OrderedDict()  # Create an empty dictionary to store all simulations from every sampled parameter values from the fitted normal distribution
        for rep_id in range(rep):
            while tolerable_sampling:
                # Define a dict to collect parameter value for parameters other than CTF. For CTF, it will be accessed directly from the dict
                par_dict = {}
                for para_abb, para_dist in para_dist_dict.items():
                    if para_abb=="CTF": # The Critical Thermal Forcing parameter is always present irrespective of the models
                        CTF_par = float(para_dist.rvs(size=1)) # Randomly sample the CTF parameter value from the calibration curve (i.e. fitted normal distribution)
                    else:
                        para_act_abb = para_abb_corresp[str(para_abb)] # Access the actual parameter abbreviation
                        par_dict[para_act_abb] = float(para_dist.rvs(size=1)) # Randomly Sample the parameter value from the fitted distribution
                # Implement the phenology model simulations according to sampled parameter values
                try:
                    if model_name=="classic_GDD":
                
                        phenology_out = phenology_model_run(weather_ser_use, thermal_threshold = CTF_par, module = model_name, 
                                                            DOY_format=True, from_budburst=False, T0=60, Tdmin=0) # Run the phenology model
                    else:
                        phenology_out = phenology_model_run(weather_ser_use, thermal_threshold = CTF_par, module = model_name, 
                                                            DOY_format=True, from_budburst=False, T0=60, **par_dict) # Run the phenology model      
                except:
                    continue  # In case of incorrect parameter sampling that lead to errors during simulations, it will do the resampling of parameters to re-do the simulations
                # In case of any NaN values, it will do the parameter resampling to re-do the simulations despite correct model runs
                if any(phenology_out.isnull()): # but in case of any NaN values despite correct model runs,
                    continue
                # In case of correct model run without NaN values, indicating the parameter sampling fine
                else:
                    break # Break the while loop means the parameter sampling is within the reasonable range, thus can go to the next step
            # Collect the simulation values
            SM_taylor["rep_" + str(rep_id+1)] = phenology_out.copy(deep=True)
        # Make the Taylor diagram plot
        save_path_taylor = join(save_path, "taylor_diagram", model_name) # Create a customized saving path
        mkdir(save_path_taylor)
        Make_Taylor_Diagram(list(phenology_ob_use), SM_taylor, save_path_taylor, "eval_{}".format(cv_i))  
        
            
