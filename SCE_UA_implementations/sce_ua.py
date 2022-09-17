# title: "Implementation of the SCE-UA Algorithm for calibration of the run_BRIN_model"
# author: "J. Arturo Torres-Matallana(1); Chenyao Yang(2)"
# organization (1): Luxembourg Institute of Science and Technology (LIST)
# organization (2): ... (UTAD)
# date: 12.02.2022 - 03.03.2022
# importing  all the functions required
import spotpy
# from spotpy.parameter import Constant
import sys
import os 
import glob
# import random
import numpy
import pandas
import pandas as pd
import re
from collections.abc import Iterable 
from collections import OrderedDict
from spotpy.parameter import Uniform, Normal
from scipy.stats import norm, truncnorm
# from spotpy.parameter import Constant
from spotpy.objectivefunctions import rmse
from Multi_phenology_model_classes import *
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++        
def detect_break_points(my_list, interval= 1, target_break_point="all"):
    '''
    Check and return any break points in a series of continuous or discontinuous numbers (can be float or integer) 
    but with fixed interval between numbers. This is mainly used to check if there are any gap in the phenology data

    Parameters
    ----------
    my_list : list, input list contains a series of continuous or discontinuous numbers with a fixed interval between numbers.
    interval: int or float, the fixed interval between numbers in the input list.
    target_break_point: int, the target ordinal break point specified by the user. For instance, there could have been many break points in a data series, 
    where user can choose a given ordinal number.
    '''
    assert isinstance(my_list, list), "the input is not a list, {} is found".format(my_list.__class__)
    
    if all(a+interval==b for a, b in zip(my_list, my_list[1:])):
        return "The input number in the list is continuous without any breaking points"
    breaking_points ={}
    for i, (a, b) in enumerate(zip(my_list, my_list[1:])):
        if not a + interval==b:
            breaking_points["break_point_"+ str(i+1)] = b # At the break point, using the next number from the list (not the first)
        else:
            continue
    # Define the ordinal number of a target breaking point
    if isinstance(target_break_point, int): # Return a given ordinal break point number 
        target_key = list(breaking_points.keys())[target_break_point-1]
        return breaking_points[target_key]
    elif target_break_point=="all": # Attempt to collect all break points
        output_points = []
        if len(breaking_points)!=0:
            for breaking_point_key in breaking_points.keys():
                output_points.append(breaking_points[breaking_point_key])
            return output_points
        else:
            return None
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++        
def parameter_dist(uniform_low, uniform_high, pre_CV, par_name, fix_bound=False, dis_method= "percentage"):
    '''
    For each calibrated parameter, it is assumed that their parameter values follow a normal
    distribution, with their expected values following a uniform distribution. 
    From the inferred uniform distribution, the low and high values are assumed to follow a normal distribution.    
    The current version of the function only return the sampled parameter values of the expected parameter values 
    
    Parameters
    ----------
    uniform_low : int or float, the pre-defined first guess for the mean of a potential normal distribution for the lower bound of the uniform distribution 
    uniform_high: int or float, the pre-defined first guess for the mean of a potential normal distribution for the upper bound of the uniform distribution 
    pre_CV: int or float, the coefficient of variation for the parameter. The CV applieds to the normal and uniform distribution parameters
    par_name: str, the parameter name in use
    fix_bound: bool, if the boundary of the uniform distribution is fixed (default) or not.
    dis_method: str, the distribution method to draw parameter values during optimization/calibration process. Either "normal" for a normal dsitribution or "percentage" from a fixed percentage (default).  
    '''
    if not fix_bound:
        # 1. Draw flexible boundaries from a normal distribution 
        if dis_method=="normal":
            # Define the uniform distribution for the mean of parameter
            uniform_low_mean_sd = abs(uniform_low * (pre_CV / 100)) # Sd needs to be positive 
            uniform_low_mean_gauss = norm(loc=uniform_low, scale=uniform_low_mean_sd) # Generate random numbers from the pre-defined guassian distribution
            uniform_high_mean_sd = abs(uniform_high* (pre_CV / 100)  ) # Sd needs to be positive 
            uniform_high_mean_gauss = norm(loc=uniform_high, scale=uniform_high_mean_sd) # Generate random numbers from the pre-defined guassian distribution
            # Sample the mean values from the defined Gaussian distribution
            uniform_low_mean_gauss_sample = uniform_low_mean_gauss.rvs(size=1)
            uniform_high_mean_gauss_sample = uniform_high_mean_gauss.rvs(size=1)
            # Avoid situations where uniform_low_mean_gauss_sample>=uniform_high_mean_gauss_sample
            while uniform_low_mean_gauss_sample>=uniform_high_mean_gauss_sample: 
                uniform_low_mean_gauss_sample = uniform_low_mean_gauss.rvs(size=1)
                uniform_high_mean_gauss_sample = uniform_high_mean_gauss.rvs(size=1)
                if uniform_high_mean_gauss_sample>uniform_low_mean_gauss_sample:
                    break
            # In case a fixed boundary of the uniform distribution is assumed, parameter values are directly sampled from the fixed uniform distribution
            par_mean = Uniform(name= par_name, low=uniform_low_mean_gauss_sample, high=uniform_high_mean_gauss_sample)
        # 2. Draw flexible boundaries from a fixed percentage
        elif dis_method=="percentage":
            range_abs =  abs(uniform_high - uniform_low) # Obtain the absolute difference between the high and low boundary
            low_bound_var = uniform_low - range_abs * (pre_CV / 100) # Left bound moves towards left inifinity with prescribed variability 
            high_bound_var = uniform_high + range_abs * (pre_CV / 100) # Right bound moves towards right inifinity with prescribed variability 
            par_mean = Uniform(name= par_name, low=low_bound_var, high=high_bound_var) # Draw the parameter values from a newly formed uniform distribution
    else:
        par_mean = Uniform(name= par_name, low=uniform_low, high=uniform_high)
    return par_mean
    # # Define the uniform distribution for the sd of parameter. Note the lower and upper bound are sampled from a truncated normal distribution (>=0)
    # uniform_sd_low = uniform_low_mean_sd * (pre_CV / 100)
    # uniform_sd_low_gauss =  truncnorm(0, numpy.inf, loc=uniform_low_mean_sd, scale=uniform_sd_low) # A truncated normal distribution to sample std
    # uniform_sd_high = uniform_high_mean_sd * (pre_CV / 100)
    # uniform_sd_high_gauss =  truncnorm(0, numpy.inf, loc=uniform_high_mean_sd, scale=uniform_sd_high) # A truncated normal distribution to sample std
    # # Sample the std from the defined gaussian distribution
    # uniform_sd_low_gauss_sample = uniform_sd_low_gauss.rvs(size=1)
    # uniform_sd_high_gauss_sample = uniform_sd_high_gauss.rvs(size = 1)
    # while uniform_sd_low_gauss_sample>=uniform_sd_high_gauss_sample: # while uniform_sd_low_sample>=uniform_sd_high_sample:
    #     uniform_sd_low_gauss_sample = uniform_sd_low_gauss.rvs(size=1)
    #     uniform_sd_high_gauss_sample = uniform_sd_high_gauss.rvs(size=1)
    #     if uniform_sd_high_gauss_sample>uniform_sd_low_gauss_sample:
    #         break
    # par_sd = Uniform(low=uniform_sd_low_gauss_sample, high=uniform_sd_high_gauss_sample)
    # Avoid negative std
    # if uniform_sd_low_sample<=0:
    #     uniform_sd_low_sample = 0
    # elif uniform_sd_high_sample<=0:
    #     uniform_sd_high_sample= 0
    # # Avoid high is lower than low in the uniform distribution
    # while uniform_sd_low_sample>=uniform_sd_high_sample:
    #return Normal(name=par_name, mean=par_mean.optguess, stddev= par_sd.optguess) # Return a normal distribution of potential parameter value    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def check_file_stat(output_csv, file_type =".csv"):
    """
    Check if a target file is writable or appendable, among others (with different modes).
    
    Parameter    
    ----------
    output_csv: path or file-like object, path pointing out to the target file
    file_type: str, the file suffix indicating the file type
    """
    while True:   # repeat until the try statement succeeds
        try:
            myfile = open(output_csv, "w") # or "a+", whatever you need
            break                             # exit the loop
        except IOError:
            print("Could not open the file! Please close the {} file".format(file_type))
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++        
# import time
# from datetime import datetime, timedelta
def f(d_ini, folder_out, path_data, path_obs, cv_i, **kwargs):
    '''
    Implement the SCE-UA optimization algorithm to estimate the best-fit parameters
    
    Parameter
    ----------
    d_ini : list, list of the calibration time period according to observed phenology study years
    folder_out: str/path-like object, the path to the directory of calibration output
    path_data: str/path-like object, the path to the directory of observed weather data
    path_obs: str/path-like object, the path to the directory of observed phenology data
    kwargs: any additional keyword arguments, mainly composed of key-word parameter pairs
    '''
    class MySpotSetup(object):
        # Define the init attribute for the MySpotSetup class
        def __init__(self, obj_func=None):
            self.obj_func = obj_func
            # Assign the class attribute to get the target parameter list
            #self.params = MySpotSetup.param_list  
            # self.d_ini = d_ini
            # self.id_cpu = id_cpu
            self.path_data = path_data
            self.path_obs = path_obs

            # Beginning of code implemented by Chenyao!
            # 1. Load observed phenology data and weather data
            phenology_raw = pd.read_csv(self.path_obs, header=0, index_col=False)  # Here the sample data only contains the budburst date for a given variety
            # Determine the study variety
            if any(phenology_raw.columns.str.contains("TN",regex=False)):
                study_variety = "TN"
            elif any(phenology_raw.columns.str.contains("TF",regex=False)):
                study_variety = "TF"
            phenology_ob = phenology_raw.set_index("Year") # Set the year index
            #study_years = [min(phenology_ob.index) - 1] + phenology_ob.index.to_list()  # Obtain the inferred study years. Get -1 year so that it includes the preceding year of the first starting year
            study_years =  list(phenology_ob.index.unique())         
            # For weather input data
            weather_input_path = os.path.join(self.path_data, study_variety)
            weather_input_csv_files  = glob.glob(os.path.join(weather_input_path,"*.csv"))
            weather_OB_plots = {}
            re_pattern = re.compile(r"[\\]\w?_gridded") # Pre-define the regular expression search pattern 
            for weather_input_csv in weather_input_csv_files:
                # Extract the plot name
                plot_name = re_pattern.findall(weather_input_csv)[0].strip("\\_gridded")
                # Read the gridded climate dataset for the vineyard plot
                weather_OB = pd.read_csv(weather_input_csv, header=0, index_col=False)
                # Subset the gridded climate dataset for the vineyard plot according to study years 
                weather_OB_subsets =  weather_OB[weather_OB["Year"].isin(study_years)]
                # Create the date time index for the weather input data
                Datetime_index = pd.to_datetime(weather_OB_subsets.loc[:, weather_OB_subsets.columns.isin(["day", "month", "Year"])]) # Generate the datetime index
                # Construct the Tmean daily series over study years
                Tmean_OB = pd.Series(weather_OB_subsets["tg"].values, index = Datetime_index, name="Tmean")  # Obtain the time series of daily minimum temperature, the name should be named as tasmin.
                # Attach the subset weather df into the target dict
                weather_OB_plots[plot_name] = Tmean_OB.copy(deep=True)
            
            # weather_OB.to_csv(
            #     folder_out + '/tmp_weather_OB_' + str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) + '.csv') # ato
            # weather_OB_datetime_index = pd.to_datetime(weather_OB.loc[:, weather_OB.columns.isin(["day", "month", "Year"])]) # Generate the datetime index
            # T_min_series = pd.Series(weather_OB["tasmin"].values, index = weather_OB_datetime_index, name="tasmin")  # Obtain the time series of daily minimum temperature, the name should be named as tasmin.
            # T_max_series = pd.Series(weather_OB["tasmax"].values, index = weather_OB_datetime_index, name="tasmax")  # Obtain the time series of daily maximum temperature,  the name should be named as tasmax.
            # # Subset the series for which only study years are analyzed
            # T_min_series = T_min_series.loc[T_min_series.index.year.isin(study_years)]
            # T_max_series = T_max_series.loc[T_max_series.index.year.isin(study_years)]
            # T_mean_series= (T_min_series + T_max_series)/2 # Compute the mean series
            # Ending of code implemented by Chenyao!
            self.obs_bbch = phenology_ob.copy(deep=True)
            plots_order = phenology_ob["Plots_{}".format(study_variety)].unique() # return values in the order of their appearance and with the same dtype.
            self.plots_order = plots_order
            #self.T_min_series = T_min_series
            #self.T_max_series = T_max_series
            self.T_mean_series = weather_OB_plots.copy()
            self.study_variety = study_variety
        # Define the essential parameter generating function
        def parameters(self):
            param_list = [] # Create an empty list to append sampling parameter values 
            if len(kwargs)!=0:
                if ("classic_GDD" not in kwargs.values()) and ("GDD_Richardson" not in kwargs.values()):
                    # par0: thermal_threshold expressed in GDD, representing the growing degree days 
                    # from a starting date T0 to a date where the stage occurs.
                    # All models but GDD and GDD richardson models accumulate daily degree day between 0 and 1, therefore the critical state of forcing for a target stage is low
                    par0_sample = parameter_dist(20, 100, cv_i, "CTF") # Wide range scenario
                    #par0_sample = parameter_dist(40, 80, cv_i, "CTF") # Narrow range scenario
                else:
                    par0_sample = parameter_dist(700, 1500, cv_i, "CTF") # Wide range scenario
                    #par0_sample = parameter_dist(1100, 1300, cv_i, "CTF") # Narrow range scenario
                # Append the parameter values
                param_list.append(par0_sample)
                if ("wang" in kwargs.values()) or ("triangular" in kwargs.values()):
                    # par1: Tdmin, the base temperature for phenology development
                    par1_sample = parameter_dist(-5, 10, cv_i, "MnDT") # Wide range scenario
                    #par1_sample = parameter_dist(0, 5, cv_i, "MnDT") # Narrow range scenario
                    # Append the parameter values
                    param_list.append(par1_sample)
                    # par2: Tdopt, the optimum temperature at which development rate is optimum.
                    par2_sample = parameter_dist(15, 28, cv_i, "ODT") # Wide range scenario
                    #par2_sample = parameter_dist(22, 25, cv_i, "ODT") # Narrow range scenario
                    # Append the parameter values
                    param_list.append(par2_sample)
                    # par3: Tdmax, the maximum temperature at which development rate is stopped.
                    par3_sample = parameter_dist(32, 42, cv_i, "MxDT") # Wide range scenario
                    # par3_sample = parameter_dist(28, 35, cv_i, "MxDT") # Narrow range scenario
                    # Append the parameter values
                    param_list.append(par3_sample)
                elif "GDD_Richardson" in kwargs.values():
                    # par1: Tdmin, the base temperature for phenology development
                    par1_sample = parameter_dist(-5, 10, cv_i, "MnDT") # Wide range scenario
                    # par1_sample = parameter_dist(0, 5, cv_i, "MnDT") # Narrow range scenario
                    # Append the parameter values
                    param_list.append(par1_sample)
                    # par3: Tdmax, the optimum temperature at which development rate is optimum.
                    par3_sample = parameter_dist(32, 42, cv_i, "MxDT") # Wide range scenario
                    # par3_sample = parameter_dist(28, 35, cv_i, "MxDT") # Narrow range scenario
                    # Append the parameter values
                    param_list.append(par3_sample)
                elif "sigmoid" in kwargs.values():
                    # par4: a, the sharpness of the curve exclusively for the sigmoid model
                    par4_sample = parameter_dist(-30, 30, cv_i, "CS") # Wide range scenario
                    # par4_sample = parameter_dist(-15, 0, cv_i, "CS") # Narrow range scenario
                    # Append the parameter values
                    param_list.append(par4_sample)
                    # par5: b, the mid-response temperature exclusively for the sigmoid model
                    par5_sample = parameter_dist(0, 25, cv_i, "MRT") # Wide range scenario
                    # par5_sample = parameter_dist(10, 20, cv_i, "MRT") # Narrow range scenario
                    # Append the parameter values
                    param_list.append(par5_sample)
            else:
                raise KeyError("The model choice is not specified")
            # Attach the collect parameter list into the class instance attribute
            self.params = param_list
            return spotpy.parameter.generate(self.params)
        def simulation(self, x):
            # Define the parameter vector to be passed
            print('... before running model ...')
            detect_nan = False
            while numpy.invert(detect_nan):
                # d_ini_sim = d_ini
                #id_cpu_sim = self.id_cpu
                # In all cases, the par0 will exist, because all model need to know the target thermal forcing
                #par0 = norm.rvs(loc=x[0], scale=x[1], size=1)
                par0 = round(x[0], 1)
                # For other structural parameters, a dict is defined to collect all values
                par_dict = {}
                if ("wang" in kwargs.values()) or ("triangular" in kwargs.values()):
                    #par1 = norm.rvs(loc=x[2], scale=x[3], size=1)
                    par1 = round(x[1], 1)
                    par_dict["Tdmin"] = par1
                    #par2 = norm.rvs(loc=x[4], scale=x[5], size=1)
                    par2 = round(x[2], 1)
                    par_dict["Tdopt"] = par2
                    #par3 = norm.rvs(loc=x[6], scale=x[7], size=1)
                    par3 = round(x[3], 1)
                    par_dict["Tdmax"] = par3
                    # Concatenate the parameter list
                    pars = pd.DataFrame([par0, par1, par2, par3])
                    print("Parameters chosen are: par0={0}, par1={1}, par2={2}, par3={3}".format(str(par0), str(par1), str(par2), str(par3)))
                elif "GDD_Richardson" in kwargs.values():
                    #par1 = norm.rvs(loc=x[2], scale=x[3], size=1)
                    par1 = round(x[1], 1)
                    par_dict["Tdmin"] = par1
                    #par2 = norm.rvs(loc=x[4], scale=x[5], size=1)
                    par2 = round(x[2], 1)
                    par_dict["Tdmax"] = par2
                    # Concatenate the parameter list
                    pars = pd.DataFrame([par0, par1, par2])
                    print("Parameters chosen are: par0={0}, par1={1}, par2={2}".format(str(par0), str(par1), str(par2)))
                elif "sigmoid" in kwargs.values():
                    #par1 = norm.rvs(loc=x[2], scale=x[3], size=1)
                    par1 = round(x[1], 1)
                    par_dict["a"] = par1
                    #par2 = norm.rvs(loc=x[4], scale=x[5], size=1)
                    par2 = round(x[2], 1)
                    par_dict["b"] = par2
                    # Concatenate the parameter list
                    pars = pd.DataFrame([par0, par1, par2])
                    print("Parameters chosen are: par0={0}, par1={1}, par2={2}".format(str(par0), str(par1), str(par2)))
                elif "classic_GDD" in kwargs.values():
                    # Write the selected parameter vectors to .csv file
                    pars = pd.Series(par0)
                    print("Parameters chosen are: par0={0}".format(str(par0)))
                # Write the selected parameter vectors to .csv file    
                par_csv= os.path.join(folder_out, "tmp_pars_" + '.csv') #str(d_ini[0]) + '_' +str(d_ini[len(d_ini) - 1])
                check_file_stat(par_csv)
                pars.to_csv(par_csv) 
                # Beginning of code implemented by Chenyao!
                # 2. Run BRIN model to generate output for dormancy and budburst date
                
                    # dormancy_out, budburst_out = run_BRIN_model(self.T_min_series, self.T_max_series, CCU_dormancy=par0,
                    #                                         CGDH_budburst=par1, TMBc_budburst=par2, TOBc_budburst=par3,
                    #                                         Richarson_model='daily')
                study_variety = self.study_variety
                OB_phenology  = self.obs_bbch  # Access the observed phenology data
                OB_phenology_data = OB_phenology[str(study_variety)] # pandas.DataFrame(self.obs_bbch09)
                weather_OB_plots_dict = self.T_mean_series.copy() # Assign the input climate data dict into a variable 
                weather_OB_plots_order_dict = OrderedDict() # Create an empty ordered dict
                plots_order = self.plots_order # Access the underlying observations
                for plot_order in list(plots_order):
                    weather_OB_plots_order_dict[plot_order] = weather_OB_plots_dict[plot_order].copy(deep=True) # This ensure the simulation follows the same plot order of observations
                phenology_SM_ser = pandas.Series(dtype="float")
                for plot_name, T_mean in weather_OB_plots_order_dict.items():
                    try:
                        if "classic_GDD" in kwargs.values():
                
                            phenology_out = phenology_model_run(T_mean, thermal_threshold = par0, module = list(kwargs.values())[0], 
                                                                DOY_format=True, from_budburst=False, T0=0, Tdmin=0) # Other parameters
                        else:
                            phenology_out = phenology_model_run(T_mean, thermal_threshold = par0, module = list(kwargs.values())[0], 
                                                                DOY_format=True, from_budburst=False, T0=0, **par_dict) # Other parameters        
                    except:
                        break
                    phenology_SM_ser= pandas.concat([phenology_SM_ser, phenology_out],  axis=0, join="outer")
                
                    
                    # phenology_out.name = "SM"# Set the simulation series name
                    # phenology_ob_input = phenology_OB.loc[phenology_OB["Plots_{}".format(study_variety)]==plot_name, study_variety]
                    # phenology_ob_input.name = "OB" # Set the observation series name
                    # phenology_ob_sm_df = pd.concat([phenology_ob_input, phenology_out], axis=1, join='inner', ignore_index=False)  
                    # # Concatenate the resulting df into the final df
                    # phenology_sum = pd.concat([phenology_sum,phenology_ob_sm_df], axis=0, join='outer', ignore_index=False)
                    
                    #phenology_SM_plot[plot_name] = phenology_out.copy(deep=True)
                
                
                # A list of BRIN model parameters that need to be calibrated (the docst can be retrieved by calling run_BRIN_model.__doc__)
                # CCU_dormancy: float, a BRIN model parameter that represents the cumulative chilling unit to break the dormancy with calculations starting from the starting_DOY.
                # CGDH_budburst: float, a BRIN model parameter that represents the cumulative growing degree hours to reach the budbust onset from the dormancy break DOY.
                # TMBc_budburst: float, a BRIN model parameter that represents the upper temperatue threshold for the linear response function.
                # TOBc_budburst: float, a BRIN model parameter that represents the base temperature for the post-dormancy period.
                
                # here we add the check function of NA values
                if any(phenology_SM_ser.isnull()):
                    detect_nan = True
                    break
                if len(OB_phenology_data)!= len(phenology_SM_ser): # If length of simulations and that of observations are not equivalent, break the iteration loop
                    print("Number of observed phenology records are not equivalent to those of simulations for model {} under parameter0 at {}".format(list(kwargs.values())[0], par0))
                    break
                # Set the new attribute
                #setattr(self, "obs_bbch_copy", phenology_sum.copy(deep=True))
                
                #bbch09_sim = pandas.Series(budburst_pred)
                phenology_out_csv = os.path.join(folder_out, 'tmp_bbch_sim_' + '.csv') #str(d_ini[0]) + '_'+ str(d_ini[len(d_ini) - 1])
                check_file_stat(phenology_out_csv)
                phenology_SM_ser.to_csv(phenology_out_csv, index=False, header=False)  # ato

                #bbch09_sim1 = bbch09_sim['budburst_pred']

                return phenology_SM_ser.values 

        def evaluation(self):
            study_variety = self.study_variety
            obs_bbch_data = self.obs_bbch[str(study_variety)] # pandas.DataFrame(self.obs_bbch09)
            observation_csv = os.path.join(folder_out, 'tmp_obs_bbch' +  '.csv') #str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) +
            obs_bbch_data.to_csv(observation_csv,
                index=False, header=False)  # ato

            return obs_bbch_data.values

        def objectivefunction(self, simulation, evaluation, params=None):
            if isinstance(simulation,Iterable):   
                check_NaN_bool = any(pd.isnull(simulation)) 
            elif ~isinstance(simulation,Iterable):
                check_NaN_bool = pd.isnull(simulation)
            if check_NaN_bool:
                like = numpy.nan
                return like
            
            #simulation.to_csv(folder_out + '/my_sim.csv', mode="a", index=False, header=False)  # ato
            # Convert both evaluation and simulation into arrays
            evaluation = numpy.array(evaluation)
            simulation = numpy.array(simulation)
            # check if -999 values are present in evaluation (obs)
            if -999 in evaluation: # Check if any missing values are found in observations
                # Deal with missing values
                id_missing = numpy.where(evaluation == -999)
                id_missing1 = list(id_missing)
                id_missing1 = pandas.DataFrame(id_missing1)
                id_missing1 = id_missing1.T
                id_missing1.columns = ['id_missing']
                missing_file = os.path.join(folder_out, 'tmp_observations_missing.csv')
                id_missing1.to_csv(missing_file) #str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1])
                # evaluation_df = pandas.Series(evaluation)
                # evaluation.to_csv(folder_out + '/tmp_evaluation_' + str(d_ini[0]) + '_' +
                #                   str(d_ini[len(d_ini) - 1]) + '.csv')
                # simulation.to_csv(folder_out + '/tmp_simulation_' + str(d_ini[0]) + '_' + str(d_ini[1]) + '.csv')
                id_no_missing = numpy.where(evaluation != -999)
                evaluation = evaluation[id_no_missing]
                simulation = simulation[id_no_missing] # No needed after including len(d_ini) = 4

            # Evaluating rmse
            like = rmse(evaluation, simulation)
            like_2csv = pandas.DataFrame([like])
            like_2csv_path = os.path.join(folder_out, 'tmp_like' + '.csv') # str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1])
            check_file_stat(like_2csv_path)
            like_2csv.to_csv(like_2csv_path, mode='a',
                index=False, header=False)  # ato
            print("Successul run. rmse = " +str(like))
            
            return like

    # ' Calibrating with SCE-UA
    log_out = os.path.join(folder_out, 'SCEUA_log' + '_cv' +
                      '.log')  # str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1])
    sys.stdout = open(log_out, 'w')

    my_spot_setup = MySpotSetup(spotpy.objectivefunctions.rmse)
    print('... before sampler...')
    db_out = os.path.join(folder_out, 'SCEUA_db_cv' + str(cv_i)) # str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) 
    sampler = spotpy.algorithms.sceua(my_spot_setup, dbname=db_out,
                                      dbformat='csv') #parallel='mpi')

    # ' Select number of maximum repetitions
    rep = 50000 # 5000

    # ' We start the sampler and set some optional algorithm specific settings
    # ' (check out the publication of SCE-UA for details):
    # sampler.sample(rep, ngs=9, kstop=3, peps=0.1, pcento=0.1) # First setting. The ngs parameter should be greater than number of parameters for calibrations, including the mean and std of each target parameter.
    
    # Place where the SCE-UA is being implemented
    sampler.sample(rep, ngs=40, kstop= 100, peps=0.0000001, pcento=0.0000001, max_loop_inc=None) # Suggested setting
    # sampler.sample(rep, ngs=20, kstop=100, peps=0.0000001, pcento=0.0000001, max_loop_inc=None) # Suggested setting
    # sys.stdout.close()

    # Get the results of the sampler
    results = sampler.getdata()

    # Use the analyser to show the parameter interaction _' + str(d_ini[0]) + '_' + str(d_ini[1]) + '.csv'
    spotpy.analyser.plot_parameterInteraction(results, fig_name=folder_out + '/Parameterinteraction.png')

    # posterior = spotpy.analyser.get_posterior(results, percentage=10)

    # spotpy.analyser.plot_parameterInteraction(posterior, fig_name=folder_out + '/Parameterinteraction_post_' +
    #                                                               str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) +
    #                                                               '_cv' + str(cv_i) + '.png')

    # bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)
    # best_param = spotpy.analyser.get_best_parameterset(results)
    #
    # best_model_run = results[bestindex]
    # print(best_model_run)
    # print(bestobjf)
    # print('Check ' + str(best_param))
    #
    # fw = open(folder_out + '/SCEUA_vineyard_bestmodel_' + str(id_cpu) + '_' + str(d_ini[0]) + '_' +
    #           str(d_ini[1]) + '.txt', 'w+')
    # fw.write('## parameters for bubdbreak calibration (best model) \r\n')
    # fw.write('\r\n')
    # # fw.write(str(best_model_run) + '\r\n')
    # fw.write('month = ' + str(best_param[0][0]) + '\r\n')
    # fw.write('day = ' + str(best_param[0][1]) + '\r\n')
    # fw.write('par1 = ' + str(best_param[0][2]) + '\r\n')
    # fw.write('rmse = ' + str(bestobjf) + '\r\n')
    # fw.close()
# if __name__ == '__main__':
#     print("Don´t execute f function")