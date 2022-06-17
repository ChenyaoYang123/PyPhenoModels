########################################################################## Function or Library Blocks ####################################################################################################################
import xarray as xr
import rioxarray
import os
import os.path
import pandas as pd
import numpy as np
import re
import decimal
import time
import csv
# import matplotlib.pyplot as plt 
# # import matplotlib.ticker as mtick
# import plotly.express as px
# import plotly.graph_objects as go
import glob
import os
import sys
import getpass
import calendar
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
from matplotlib_scalebar.scalebar import ScaleBar
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from itertools import product
from os.path import join,dirname
from datetime import datetime
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from collections.abc import Iterable
from scipy.stats import pearsonr, ks_2samp,variation, iqr, mode
from scipy.interpolate import UnivariateSpline
from scipy.stats.kde import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error

def add_script(script_drive, target_dir):
    '''
    Add script stored in the target directory to the system path

    Parameters
    ----------
    target_dir : str or path-like object,  a string path to the target disk path where the multi-model phenology classess are stored
    
    '''
    script_path = join(script_drive, target_dir)
    sys.path.append(script_path)
    sys.path.append(dirname(script_path))
########################################################################## Function or Library Blocks ####################################################################################################################

########################################################################## Coding Blocks #################################################################################################################################
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Input code session+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. Specify all user-dependent path and other necessary constant input variables
# 1.1 Check for target drive letter, which basically check the device working with.
if getpass.getuser() == 'Clim4Vitis':
    script_drive = "H:\\"
elif getpass.getuser() == 'Admin':
    script_drive = "G:\\"
elif (getpass.getuser() == 'CHENYAO YANG') or (getpass.getuser() == 'cheny'):
    script_drive = "D:\\"
main_script_path_ = r"Mega\Workspace\Study for grapevine" # Specific for a given study
target_dir = join(script_drive, main_script_path_, "Study7_Sigmoid_phenology_modelling_seasonal_forecast", "script_collections")
# 1.2 Add the script path into system path
add_script(script_drive, target_dir) # Add the target folder containing necessary self-defined scripts into the system path
# 1.3 Import all essential functions and classes used
from Multi_phenology_model_classes import * # Add the multi-phenology model class script from myself
from fc_Flo_Ver_func import *
# 1.4 Define essential path for local files that will be used in the analysis
root_path= join(script_drive, "Grapevine_model_SeasonalForecast_FloVer") # Main root path for the data analysis
E_OBS_path = join(script_drive, "E_OBS_V24") # Define the E-OBS climate dataset path
forecast_path = join(root_path, "Forecast_datasets_ECMWF") # Define the forecast dataset path
output_path = join(root_path,"output") # Define the output path
# 1.5 Path for the target shape files
shape_path = join(script_drive, r"Mega\Workspace\SpatialModellingSRC\GIS_root\Root\shape_wine_regions") # Define the main path to target shapefile 
study_shapes = glob.glob(join(shape_path,"*.shp")) # Obtain a list of shape files used for defining the study region
study_region = "PT_outline" # Define the name of file used for study region. 
study_outline = "PT_outline" # Define the name of file used for the background outline
study_region_shape = [shape for shape in study_shapes if study_region in shape][0] # Squeeze the list
study_outline_shape = [shape for shape in study_shapes if study_outline in shape][0] # Squeeze the list
proj = ccrs.PlateCarree() # Define the target projection CRS
GDF_shape= gpd.read_file(study_region_shape).to_crs(proj) # to_crs(proj) # Load the study region shape file into a geographic dataframe
GDF_shape_outline= gpd.read_file(study_outline_shape).to_crs(proj) # Load the background outline shape file into a geographic dataframe
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 2. Define a list of target grid points that comprise the study region
# 2.1 Define essential inputs
determine_lat_lon_vector = "by_shape" # Two options are available: collect data points either from shape file or inferred from forecast dataset itself
forecast_time = [2, 3, 4, 5]  # Define the forecast initialization month, which should be case-specific
forecast_time_data = sort_FCdatasets_by_FCdate(forecast_path, forecast_time) # Obtain the target forecast dict dataset sorted by different forecast initilization dates
metero_var_dict_forecast= {"Tmin":"mn2t24", "Tmax":"mx2t24", "Prec":"tp"} # Define the meteorological variable dictionary
resolution = 0.1 # Define the grid point spatial resolution
begin_year = 1993 # The starting study year
end_year = 2017 # The last study year
study_period = np.arange(begin_year,end_year+1,1) # Define the study period
date_range_output= pd.date_range("{}-12-31".format(begin_year), "{}-12-31".format(end_year), freq="Y") # Define the output datetime(year) range
metero_var_dict_OB = {"Tmin":"tn", "Tmean":"tg", "Tmax":"tx"} # Define the respective meteorological dict used for extracting time series
metero_var = ["tn","tg","tx"] # Define the meteorological variable that corresponds to climate dataset var names
target_column_order = ['day','month','Year', *metero_var, 'lat','lon'] # A pre-specified order of columns for meteorological data
ensemble_members = 25 # Define the number of ensemble members used in the seasonal forecast
resolution = 0.1 # Define the gridding resolution of target grid points
decimal_place = abs(decimal.Decimal(str(resolution)).as_tuple().exponent) # Extract number of decimal places in the input float format
# 2.2 Collect study grid points according to the method specified
if determine_lat_lon_vector=="by_shape": # Provide a shape file that delineate AOI (area of interests) with all grid point information
    #GDF_shape = GDF_shape[(GDF_shape["name"]!="Vinho Verde") & (GDF_shape["name"]!="Douro")] # Sampling the geodataframe so that only Lisbon wine region is studied
    # Extract the boundary rectangle coordinates and define the study resolution 
    minx, miny, maxx, maxy = GDF_shape.total_bounds
    decimal_place = abs(decimal.Decimal(str(resolution)).as_tuple().exponent) # Extract number of decimal places in the input float
    # Round the boundary coordinates into target resolution
    minx = round(minx,decimal_place) 
    maxx = round(maxx,decimal_place) 
    miny = round(miny,decimal_place) 
    maxy = round(maxy,decimal_place)
    # Form a pair of coordinates from existing shape file boundary and check if each point is within the shape
    lon_vector = np.arange(minx, maxx, resolution)
    lat_vector = np.arange(miny, maxy, resolution)
    coordinates = list(product(lon_vector, lat_vector)) # Form the list of coordinates to be studied
    # Collect the target points to do the simulations
    target_points = []
    for coordinate in coordinates:
        grid_point = Point(coordinate)
        if any(GDF_shape.geometry.contains(grid_point, align=False)):
            target_points.append(grid_point)
elif determine_lat_lon_vector=="by_dataset": # Infer the study coordinates from the forecast dataset coordinates
    geo_points = []    
    for fc_time in forecast_time: # It can also supply observed climate datasets to infer the target coordinates
        list_of_points = check_lonlat_ncdatasets(forecast_time_data[str(fc_time)])
        geo_points.append(gpd.GeoSeries(list_of_points))
    for geo_ser_first, geo_ser_second in zip(geo_points[:], geo_points[1:]):
        if not all(geo_ser_first.geom_equals(geo_ser_second)):
            raise ValueError("Grid points are not equal across different forecast datasets")
    # After the equality test, simply return the target list of grid points
    target_points = list(geo_points[0]) # Since any of the geo_points elements are equal, only use the first one
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 3. Define the output xarray templates to store output data
# 3.1 Obtain a list of lon and lat vectors from collected target_points
# Note here the shapely point object will generate the floating points when extracting the longitude and latitude
lon_target_vector = np.unique([round(target_point.x, decimal_place) for target_point in target_points]) # A unique list of longitude
lat_target_vector = np.unique([round(target_point.y, decimal_place) for target_point in target_points]) # A unique list of latitude
coords_xarray = [ ("lat", lat_target_vector), ("lon", lon_target_vector)] # Create the coordinate dimensions ("time",time_vector)
# 3.2 Define the template xarray files that are aimed for storing different kinds of output dataset
output_template_score = xr.DataArray(coords=coords_xarray) # Create a dimensional template xarray object that is going to be used as the output structure
output_template_sm_ob = xr.DataArray(coords=coords_xarray + [("time", date_range_output)]) # Create a dimensional template xarray object object that is going to be used as the output structure
output_template_sm_fc = xr.DataArray(coords=coords_xarray + [("time", date_range_output), ("number", range(ensemble_members))]) # Create a dimensional template xarray object object that is going to be used as the output structure
# 3.3 Create a dictionary to save output template files for reference simulations
forecast_ob_dict ={} # Dictionary to store simulated data with observed weather
forecast_ob_dict["ob_flo"] = output_template_sm_ob.copy(deep=True)
forecast_ob_dict["ob_ver"] = output_template_sm_ob.copy(deep=True)
# 3.4 Create a dictionary of output xarray objects to save forecast scores
forecast_r_dict = {} # Dictionary to store forecast performance scores of r
forecast_r_sig_dict = {} # Dictionary to store forecast performance scores of r with significant analysis
forecast_MAE_dict = {} # Dictionary to store forecast performance scores of MAE
forecast_RMSE_dict = {} # Dictionary to store forecast performance scores of RMSE
forecast_fairRPS_dict = {} # Dictionary to store forecast performance scores of fairRPS
forecast_sm_dict ={} # Dictionary to store the simulation data (in this case phenology DOY)
forecast_yearly_score_dict = {} # Dictionary to store yearly score (compute ensemble forecast scores per year), e.g. RMSE per year. Now discarded since ensemble forecast probability is computed 
# 3.5 Attach to each empty dict the target xarray object for each forecast initialization time
for forecast_month in forecast_time:
    forecast_r_dict[str(forecast_month)+"_flo"] = output_template_score.copy(deep=True)
    forecast_r_dict[str(forecast_month)+"_ver"] = output_template_score.copy(deep=True)
    forecast_r_sig_dict[str(forecast_month)+"_flo"] = output_template_score.copy(deep=True)
    forecast_r_sig_dict[str(forecast_month)+"_ver"] = output_template_score.copy(deep=True)
    forecast_MAE_dict[str(forecast_month)+"_flo"] = output_template_score.copy(deep=True)
    forecast_MAE_dict[str(forecast_month)+"_ver"] = output_template_score.copy(deep=True)
    forecast_RMSE_dict[str(forecast_month)+"_flo"] = output_template_score.copy(deep=True)
    forecast_RMSE_dict[str(forecast_month)+"_ver"] = output_template_score.copy(deep=True)
    forecast_fairRPS_dict[str(forecast_month)+"_flo"] = output_template_score.copy(deep=True)
    forecast_fairRPS_dict[str(forecast_month)+"_ver"] = output_template_score.copy(deep=True)   
    forecast_sm_dict[str(forecast_month)+"sm_flo"] = output_template_sm_fc.copy(deep=True)
    forecast_sm_dict[str(forecast_month)+"sm_ver"] = output_template_sm_fc.copy(deep=True)  
    forecast_yearly_score_dict[str(forecast_month)+"sm_flo"] = output_template_sm_ob.copy(deep=True)
    forecast_yearly_score_dict[str(forecast_month)+"sm_ver"] = output_template_sm_ob.copy(deep=True)    
# 3.6 Create template xarray objects to store GDD data analysis
daily_date_range= pd.date_range("{}-01-01".format(begin_year), "{}-12-31".format(end_year), freq="D") # Define the output daily date range, aligning with selected seasonal span
flo_end_date = "07-31" # A string representation of last possible date of flowering in studied wine regions
ver_end_date =  "09-30" # A string representation of last possible date of veraison in studied wine regions
forecast_GDD_dict_ob = {} # Dictionary to store simulated data with observed GDD
forecast_GDD_dict_fc ={} # Dictionary to store simulated data with forecast data GDD 
for forecast_month in forecast_time:
    # Define the starting month DOY
    init_date = pd.to_datetime("{}-01".format(str(forecast_month)), format="%m-%d")
    # Define the period range from an initial date to a specified date that corresponds to either reference flowering DOY or veraison DOY
    flo_month_ref = pd.period_range(init_date, pd.to_datetime(flo_end_date, format="%m-%d"), freq="M").month
    ver_month_ref = pd.period_range(init_date, pd.to_datetime(ver_end_date, format="%m-%d"), freq="M").month
    # Get the time series dimension
    daily_date_range_flo =  daily_date_range[daily_date_range.month.isin(flo_month_ref)]
    daily_date_range_ver =  daily_date_range[daily_date_range.month.isin(ver_month_ref)]
    # Get the output template for flowering and veraison 
    output_template_GDD_flo = xr.DataArray(coords=coords_xarray + [("time", daily_date_range_flo)]) # Create a dimensional template xarray object object that is going to be used as the output structure
    output_template_GDD_ver = xr.DataArray(coords=coords_xarray + [("time", daily_date_range_ver)]) # Create a dimensional template xarray object object that is going to be used as the output structure
    # Attach the results data array into the target observational dict
    forecast_GDD_dict_ob[str(forecast_month)+"gdd_flo"] = output_template_GDD_flo.copy(deep=True)
    forecast_GDD_dict_ob[str(forecast_month)+"gdd_ver"] = output_template_GDD_ver.copy(deep=True)
    # Attach the results data array into the target simulation dict
    forecast_GDD_dict_fc[str(forecast_month)+"gdd_flo"] = output_template_GDD_flo.copy(deep=True)
    forecast_GDD_dict_fc[str(forecast_month)+"gdd_ver"] = output_template_GDD_ver.copy(deep=True)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 4 Pre-load the meteorological datasets (reference and forecast) into memoery
# 4.1 Pre-load reference meteorological datasets
# 4.1.1 Collect reference meteorological dataset .nc files
target_datasets_OB = [] # Define an empty list to collect target observational gridded datasets. Here it only refers to observational weather data
for ncfile in glob.glob(join(E_OBS_path,"*.nc")):
    if any(varname in ncfile for varname in metero_var):
        target_datasets_OB.append(ncfile)
# 4.1.2 Pre-load the .nc datasets into memory in the form of xarray.dataset object 
xarray_OBdata_dict = {} 
for grid_data in target_datasets_OB:
    if any([var in grid_data for var in metero_var]):
        xarray_data = xr.open_dataset(grid_data, mask_and_scale=True, engine = "netcdf4", decode_times =True) # Open and read the datasets into xarray objects
        # Access the underlying variable name of the dataset                                                
        var_shortname = list(xarray_data.data_vars.keys())[0]
        # Subset the dataset to study region and study time             
        xarray_data_subset_region_time = subset_dataset(xarray_data, lon_target_vector, lat_target_vector, study_period, preceding_year = True)                                 
        # Attach the subset into target dict
        xarray_OBdata_dict[var_shortname] = xarray_data_subset_region_time.copy(deep=True)
        xarray_data.close()
    else:
        continue
# interp_resolution = 1
# # Interpolate the datasets into 1.0 resolution
# xarray_data_interp = xarray_data_subset_region_time.interp(latitude= np.arange(latmin, latmax+interp_resolution, interp_resolution), longitude=np.arange(lonmin, lonmax+interp_resolution, interp_resolution), method="nearest")
# 4.2 Pre-load the forecast meteorological datasets
forecast_data_dict = {} 
# 4.2.1 Define the meteorological variable names used in the forecast dataset variables
Tmin_varname = metero_var_dict_forecast["Tmin"]
Tmax_varname = metero_var_dict_forecast["Tmax"]
# 4.2.2 Pre-load the .nc datasets into memory in the form of xarray.dataset object 
for forecast_month in forecast_time_data.keys():
    if forecast_month not in forecast_data_dict.keys():
        forecast_data_dict[forecast_month] ={}
    forecast_data_var_dict = forecast_time_data[forecast_month]
    # Load the list of path for pre-organized forecast datasets
    Tmin_list_path = forecast_data_var_dict["Tmin"]
    Tmax_list_path = forecast_data_var_dict["Tmax"]
    # Load all the path in the list into xarray and concate them all
    Tmin_da = load_and_concat_da(Tmin_list_path)
    Tmax_da = load_and_concat_da(Tmax_list_path)
    # Subset the datasets to target region and time period
    Tmin_da_subset = subset_dataset(Tmin_da, lon_target_vector, lat_target_vector, study_period, preceding_year = False)
    Tmax_da_subset = subset_dataset(Tmax_da, lon_target_vector, lat_target_vector, study_period, preceding_year = False)
    # Attach the pre-load data into the target dict
    forecast_data_dict[forecast_month][Tmin_varname] = Tmin_da_subset
    forecast_data_dict[forecast_month][Tmax_varname] = Tmax_da_subset
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Input code session++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Implementation code session+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 5. Run the simulations throughout target grid points (defined regions) to evaluate the fit of the sigmoid model with observed and forecast datasets
Bug_dict = {} # Create a bug dictionary to catch the bugs during simulations
## !!! Note the implementation is per variety basis, we have studied two varieties, namely TF and TN.
study_var = "TN" #  The run_model() function needs to be adjusted with parameter sets representing each variety separately

# Simulations by iterating over each target grid point
for index, target_point in enumerate(target_points):
    timer = Timer()
    timer.start()
    # Get the latitude and longitude for the point
    lon1 = round(target_point.x, decimal_place)
    lat1 = round(target_point.y, decimal_place)
    # Create point dimensional dictionaries that are used for storing the simulation data
    point = {output_template_score.dims[0]:lat1, output_template_score.dims[1]:lon1} # Basic point dim that only contains lat1 and lon1
    point_time = point.copy() # Create a new point dim via copying original point dim
    point_time["time"] = date_range_output # Update the point dimension with the time dimensions    
    print("Start running for point No.{} out of total {}".format(str(index+1),str(len(target_points)))) # Print the current loop iteration
    # Extract the daily timeseries of observed weather datasets for the specified grid point
    site_data_df  = collect_weather_data(xarray_OBdata_dict, lon1, lat1, study_period, dataset_loaded= True, 
                                         var_identifiers = metero_var, target_column_order = target_column_order, extract_method="nearest")
    # Extract the minimum, mean and maximum temperature series for reference data
    T_min = site_data_df[metero_var_dict_OB["Tmin"]] # Daily minimum temperature
    T_max = site_data_df[metero_var_dict_OB["Tmax"]] # Daily maximum temperature
    T_mean = site_data_df[metero_var_dict_OB["Tmean"]] # Daily mean temperature
    # Subset the reference temperature data to cover the study period only
    T_min = T_min.loc[T_min.index.year.isin(study_period)]
    T_max = T_max.loc[T_max.index.year.isin(study_period)]
    T_mean = T_mean.loc[T_mean.index.year.isin(study_period)]
    # Run the Sigmoid model to get the reference simulated flowering and veraison dates based on the observed weather data
    try:
        flo_ob, ver_ob = run_models(T_mean) # Perform simulations with observed weather 
        # point_coords = {key:([value] if key not in "time" else value) for key,value in point_time.items()} # The same key used from point time but with different values. Use if else in dict comprehension
        # a = xr.DataArray(np.expand_dims(flo_ob, axis=(0,1)), coords= point_coords, dims=list(forecast_ob_dict["ob_flo"].dims))
    except:
        Bug_dict.update({"lon"+str(lon1)+"_lat"+str(lat1):'Issues in simulated values with observed weather!'}) # Catch the erroneous simulation values
    # Re-set the datetime index in the reference phenology series to be consistent with those in the xarray object 
    flo_ob.index = date_range_output
    ver_ob.index = date_range_output
    # Check NaN values for simulation series. If any, fill NaN for the reference series from the nearest neighbours of non-NaN values
    if any(pd.isnull(flo_ob)):
        try:
            flo_ob = check_NaN(flo_ob)
        except:
            # Fill NaN values of simulations from those of nearest neighbours
            flo_ob = extract_nearest_neighbours(forecast_ob_dict["ob_flo"], flo_ob, lon1, lat1, Bug_dict=Bug_dict)
        # In case this still failed, append it to the Bug_dict and continue to the next iteration
        if any(pd.isnull(flo_ob)):
            Bug_dict.update({"lon"+str(lon1)+"_lat"+str(lat1):'failed to fill NaN for the flowering stage at this point'}) # catch the points that NaN values filling failed
            forecast_ob_dict["ob_flo"].loc[point_time] = -999
            continue
    # Same way following the flowering DOY
    if any(pd.isnull(ver_ob)):
        try:
            ver_ob = check_NaN(ver_ob)
        except:
            ver_ob = extract_nearest_neighbours(forecast_ob_dict["ob_ver"], ver_ob, lon1, lat1, Bug_dict=Bug_dict)
        if any(pd.isnull(ver_ob)):
            Bug_dict.update({"lon"+str(lon1)+"_lat"+str(lat1):'failed to fill NaN for the veraison stage at this point'}) # catch the points that NaN values filling failed
            forecast_ob_dict["ob_ver"].loc[point_time] = -999
            continue
    # Save the timeseries of simulated data into the xarray data array objects with simulation series free of NaN
    save_to_dataarray(forecast_ob_dict["ob_flo"], "time", flo_ob, point_time)
    save_to_dataarray(forecast_ob_dict["ob_ver"], "time", ver_ob, point_time)
    # Iterate over each forecast initialization month to analyze the forecast skills
    for forecast_month in forecast_data_dict.keys():
        # Forecast initialization date
        fc_init_date = pd.to_datetime("{}-01".format(str(forecast_month)),format="%m-%d")
        xr_concat_fc_data = forecast_data_dict[forecast_month] # Access the pre-loaded xarray dataset object at a given forecast date
        # Create empty lists to store the Pearson correlation coefficient computed for each ensemble member at a given forecast month/date
        corre_fc_ensemble_flo = []
        corre_fc_ensemble_ver = []
        # Create empty lists to store the significance test results of Pearson correlation coefficient
        corre_fc_sig_ensemble_flo = []
        corre_fc_sig_ensemble_ver = []
        # Create empty lists to store MAE
        MAE_fc_ensemble_flo = []
        MAE_fc_ensemble_ver = []
        # Create empty lists to store RMSE
        RMSE_fc_ensemble_flo = []
        RMSE_fc_ensemble_ver = []
        # Create empty lists to store simulation arrays for individual forecast member simulations
        Array_fc_ensemble_flo = []
        Array_fc_ensemble_ver = []
        # Create empty lists to store GDD values computed from forecast datasets
        GDD_fc_ensemble_flo = [] # The GDD from each initialized forecast date until a common flowering DOY
        GDD_fc_ensemble_ver = [] # The GDD from each initialized forecast date until a common veraison DOY
        # Iterate over the ensemble members under a given forecast month/date
        for ens_member in range(ensemble_members):
            point_time["number"] = ens_member # Update the point dimension with extra ensemble member dimension
            # Extract the forecast datasets series (with gap filled by using the observation data) at a specific forecast date/time/month for a given point
            Tmin_fc = extract_fc_dataset(xr_concat_fc_data[Tmin_varname], Tmin_varname, T_min, lon1, lat1, ensemble_member=ens_member)
            Tmax_fc = extract_fc_dataset(xr_concat_fc_data[Tmax_varname], Tmax_varname, T_max, lon1, lat1, ensemble_member=ens_member)
            # Ensure the length of extracted series are the same and free of NaN values
            if any(np.logical_or(pd.isnull(Tmin_fc), pd.isnull(Tmax_fc))) or (len(Tmin_fc) != len(Tmax_fc)):
                Bug_dict.update({"lon"+str(lon1)+"_lat"+str(lat1):'Errors in extracted forecast datasets'}) # catch the erroneous instances
                Tmin_fc = check_NaN(Tmin_fc) # Fill NaN values if any
                Tmax_fc = check_NaN(Tmax_fc) # Fill NaN values if any
            # Compute the mean meteorological series 
            T_mean_fc = (Tmin_fc + Tmax_fc)/2
            # Subset the temperature data to cover the study period only
            T_mean_fc = T_mean_fc.loc[T_mean_fc.index.year.isin(study_period)]
            # Compute the GDD for the reference (observational) dataset at a given forecast initialization datetime
            GDD_flo_ob, GDD_ver_ob  = time_depend_GDD(T_mean, fc_init_date, cum_sum=True)    
            # Compute the GDD for the forecast dataset at a given forecast initialization datetime
            GDD_flo_fc, GDD_ver_fc = time_depend_GDD(T_mean_fc, fc_init_date,cum_sum=True)
            # Append the GDD yearly series into target empty lists
            GDD_fc_ensemble_flo.append(GDD_flo_fc)
            GDD_fc_ensemble_ver.append(GDD_ver_fc)
            # Run the Sigmoid model to get the simulated flowering and veraison DOY based on the forecast weather data
            try:
                flo_fc, ver_fc = run_models(T_mean_fc)
            # Check if there are any NaN values in the series
            except:
                Bug_dict.update({"lon"+str(lon1)+"_lat"+str(lat1):'Issues in simulated values with forecast datasets!'}) # catch the erroneous simulation values
                #continue # Skip errorneous ensemble simulation
            # Re-set the datetime index in the forecast phenology series to be consistent with those in the xarray object 
            flo_fc.index = date_range_output
            ver_fc.index = date_range_output
            # Check NaN values for simulation series. If any, fill NaN for the forecast series from the nearest neighbours of non-NaN values
            if any(pd.isnull(flo_fc)):
                try:
                    flo_fc = check_NaN(flo_fc) # If the first fill NaN generates errors, proceed to another one
                except:
                    flo_fc = extract_nearest_neighbours(forecast_sm_dict[str(forecast_month)+"sm_flo"], flo_fc, lon1, lat1, ens_dim=True, ens_member=ens_member, Bug_dict=Bug_dict)
                # Fill NaN values of simulations from those of nearest neighbours. But in case this failed, append it to the Bug_dict and continue to the next iteration
                if any(pd.isnull(flo_fc)):
                    Bug_dict.update({"lon"+str(lon1)+"_lat"+str(lat1):'failed to fill NaN for the flowering stage at this point for the ensemble member {}'.format(ens_member)}) # catch the points that NaN values filling failed
                    forecast_sm_dict[str(forecast_month)+"sm_flo"].loc[point_time] = -999 
                    continue
            # Same way following the flowering DOY
            if any(pd.isnull(ver_fc)):
                try:
                    ver_fc = check_NaN(ver_fc) 
                except:
                    ver_fc = extract_nearest_neighbours(forecast_sm_dict[str(forecast_month)+"sm_ver"], ver_fc, lon1, lat1, ens_dim=True, ens_member=ens_member, Bug_dict=Bug_dict)
                # Fill NaN values of simulations from those of nearest neighbours. But in case this failed, append it to the Bug_dict and continue to the next iteration
                if any(pd.isnull(ver_fc)):
                    Bug_dict.update({"lon"+str(lon1)+"_lat"+str(lat1):'failed to fill NaN for the veraison stage at this point for the ensemble member {}'.format(ens_member)}) # catch the points that NaN values filling failed
                    forecast_sm_dict[str(forecast_month)+"sm_ver"].loc[point_time] = -999 
                    continue
            # Attach the forecast series into target output files
            save_to_dataarray(forecast_sm_dict[str(forecast_month)+"sm_flo"], "time", flo_fc, point_time)
            save_to_dataarray(forecast_sm_dict[str(forecast_month)+"sm_ver"], "time", ver_fc, point_time)
            # Compute the Pearson correlation coefficient
            flo_corr, flo_corr_p = pearsonr(flo_ob, flo_fc)
            ver_corr, ver_corr_p = pearsonr(ver_ob, ver_fc)
            # Assign integer values for the significance test of the Pearson correlation coefficient
            flo_sig_result = sig_test_p(flo_corr_p)
            ver_sig_result = sig_test_p(ver_corr_p)
            # Compute the MAE
            flo_MAE = mean_absolute_error(flo_ob, flo_fc)
            ver_MAE = mean_absolute_error(ver_ob, ver_fc)
            # Compute the RMSE
            flo_RMSE = np.sqrt(mean_squared_error(flo_ob, flo_fc))
            ver_RMSE = np.sqrt(mean_squared_error(ver_ob, ver_fc))
            ## Append the computed forecast skill scores per ensemble member into the result list
            # Append the Pearson correlation coefficient
            corre_fc_ensemble_flo.append(flo_corr)
            corre_fc_ensemble_ver.append(ver_corr)
            # Append the significance test result of correlation coefficient
            corre_fc_sig_ensemble_flo.append(flo_sig_result)
            corre_fc_sig_ensemble_ver.append(ver_sig_result)
            # Append the MAE
            MAE_fc_ensemble_flo.append(flo_MAE)
            MAE_fc_ensemble_ver.append(ver_MAE)
            # Append the RMSE
            RMSE_fc_ensemble_flo.append(flo_RMSE)
            RMSE_fc_ensemble_ver.append(ver_RMSE)
            # Append the simulation arrays of forecast DOY into the result list
            Array_fc_ensemble_flo.append(flo_fc)
            Array_fc_ensemble_ver.append(ver_fc)
        ## Compute the ensemble mean of all ensemble members for the forecast skill score
        # Compute the ensemble mean of Pearson correlation coefficient
        ens_corre_flo = np.nanmean(corre_fc_ensemble_flo)
        ens_corre_ver = np.nanmean(corre_fc_ensemble_ver)
        # Compute the significance of Pearson correlation coefficient by determing if a majority of ensemble members give significant results
        corre_fc_sig_ensemble_flo_arry = np.array(corre_fc_sig_ensemble_flo)
        corre_fc_sig_ensemble_ver_arry = np.array(corre_fc_sig_ensemble_ver)
        # Compute the number of ensemble members with significant pearson correlation coefficient
        corre_fc_sig_flo_p = len(corre_fc_sig_ensemble_flo_arry[corre_fc_sig_ensemble_flo_arry==1])/ len(corre_fc_sig_ensemble_flo_arry)
        corre_fc_sig_ver_p = len(corre_fc_sig_ensemble_ver_arry[corre_fc_sig_ensemble_ver_arry==1])/ len(corre_fc_sig_ensemble_ver_arry)
        # Evaluate if a majority of ensemble members that hold significant results for the Pearson correlation coefficient
        if corre_fc_sig_flo_p >= 2/3:
            corre_fc_sig_flo_sig = 1
        else:
            corre_fc_sig_flo_sig = 0
        # Indicate a majority of ensemble members having significant results
        if corre_fc_sig_ver_p >= 2/3: 
            corre_fc_sig_ver_sig = 1
        else:
            corre_fc_sig_ver_sig = 0
        # Compute the ensemble mean of MAE
        ens_MAE_flo = np.nanmean(MAE_fc_ensemble_flo)
        ens_MAE_ver = np.nanmean(MAE_fc_ensemble_ver)
        # Compute the ensemble mean of RMSE
        ens_RMSE_flo = np.nanmean(RMSE_fc_ensemble_flo)
        ens_RMSE_ver = np.nanmean(RMSE_fc_ensemble_ver)
        # Concatenate arrays into a single df
        Array_fc_ensemble_flo_df = pd.concat(Array_fc_ensemble_flo, axis=1, join="inner", ignore_index=False)
        Array_fc_ensemble_ver_df = pd.concat(Array_fc_ensemble_ver, axis=1, join="inner", ignore_index=False)
        # Compute the fairRPS for the flowering and veraison stages
        flo_fairRPS = compute_fairRPS(flo_ob, Array_fc_ensemble_flo_df, study_period)
        ver_fairRPS = compute_fairRPS(ver_ob, Array_fc_ensemble_ver_df, study_period)
        # Compute yearly RMSE from all ensemble members
        flo_ser_RMSE = compute_df_RMSE(Array_fc_ensemble_flo_df, flo_ob)
        ver_ser_RMSE = compute_df_RMSE(Array_fc_ensemble_ver_df, ver_ob)
        ## Handing the GDD 
        # Concatenate GDD series into a single df
        GDD_fc_ensemble_flo_df = pd.concat(GDD_fc_ensemble_flo, axis=1, join="inner", ignore_index=False)
        GDD_fc_ensemble_ver_df = pd.concat(GDD_fc_ensemble_ver, axis=1, join="inner", ignore_index=False)
        # Compute the ensemble mean of all ensemble members of the forecast datasets
        GDD_fc_ensemble_flo_ser = GDD_fc_ensemble_flo_df.apply(np.nanmean, axis=1, raw=False, result_type= "reduce")
        GDD_fc_ensemble_ver_ser = GDD_fc_ensemble_ver_df.apply(np.nanmean, axis=1, raw=False, result_type= "reduce")
        ## Save point data (save point score data without time dimension but with only lat lon dimension)
        # Attach the computed ensemble mean of correlation coefficient into the target xarray data array object
        forecast_r_dict[str(forecast_month)+"_flo"].loc[point] = ens_corre_flo # Since here it is already a scaler-based saving, it is not necessary to use save_to_dataarray()
        forecast_r_dict[str(forecast_month)+"_ver"].loc[point] = ens_corre_ver # Since here it is already a scaler-based saving, it is not necessary to use save_to_dataarray()
        # Attach the computed significance of correlation coefficien into target dict
        forecast_r_sig_dict[str(forecast_month)+"_flo"].loc[point] = corre_fc_sig_flo_sig # Since here it is already a scaler-based saving, it is not necessary to use save_to_dataarray()
        forecast_r_sig_dict[str(forecast_month)+"_ver"].loc[point] = corre_fc_sig_ver_sig # Since here it is already a scaler-based saving, it is not necessary to use save_to_dataarray()
        # Attach the computed ensemble mean of MAE into the target xarray data array object
        forecast_MAE_dict[str(forecast_month)+"_flo"].loc[point] = ens_MAE_flo # Since here it is already a scaler-based saving, it is not necessary to use save_to_dataarray()
        forecast_MAE_dict[str(forecast_month)+"_ver"].loc[point] = ens_MAE_ver # Since here it is already a scaler-based saving, it is not necessary to use save_to_dataarray()
        # Attach the computed ensemble mean of RMSE into the target xarray data array object
        forecast_RMSE_dict[str(forecast_month)+"_flo"].loc[point] = ens_RMSE_flo # Since here it is already a scaler-based saving, it is not necessary to use save_to_dataarray()
        forecast_RMSE_dict[str(forecast_month)+"_ver"].loc[point] = ens_RMSE_ver # Since here it is already a scaler-based saving, it is not necessary to use save_to_dataarray()
        # Attach the fairRPS score into the target xarray data array object
        forecast_fairRPS_dict[str(forecast_month)+"_flo"].loc[point] = flo_fairRPS
        forecast_fairRPS_dict[str(forecast_month)+"_ver"].loc[point] = ver_fairRPS
        ## Save time series data with dimnension (lat, lon, time). Related variables: yearly DOY, seasonal daily GDD series
        # Update the point dimension by deleting the ensemble member dimension, so that it can be used to save the data array
        del point_time["number"] # Update the point dimension by deleting the ensemble member dimension, so that it can be used to save the data array
        save_to_dataarray(forecast_yearly_score_dict[str(forecast_month)+"sm_flo"], "time", flo_ser_RMSE, point_time)
        save_to_dataarray(forecast_yearly_score_dict[str(forecast_month)+"sm_ver"], "time", ver_ser_RMSE, point_time)
        # Save the timeseries of forecast GDD values into target output files
        save_to_dataarray(forecast_GDD_dict_fc[str(forecast_month)+"gdd_flo"], "time", GDD_fc_ensemble_flo_ser, point_time)
        save_to_dataarray(forecast_GDD_dict_fc[str(forecast_month)+"gdd_ver"], "time", GDD_fc_ensemble_ver_ser, point_time)
        # Save the timeseries of observed GDD values into target output files
        save_to_dataarray(forecast_GDD_dict_ob[str(forecast_month)+"gdd_flo"], "time", GDD_flo_ob, point_time)
        save_to_dataarray(forecast_GDD_dict_ob[str(forecast_month)+"gdd_ver"], "time", GDD_ver_ob, point_time)
    # Print the current loop iteration 
    print("Finish processing for point No.{} out of total {}".format(str(index+1),str(len(target_points))))  
    timer.end()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
# 6 Export the results to files (mostly in .nc files) in local disks
# 6.1 Export the GDD outputs
# 6.1.1 Define the GDD output saving path
GDD_save = "GDD_output"
output_path_GDD_ob = join(output_path, "out_flo_ver", GDD_save, "ob")
output_path_GDD_fc = join(output_path, "out_flo_ver", GDD_save, "fc")
mkdir(output_path_GDD_ob)
mkdir(output_path_GDD_fc)
# 6.1.2 Save the computed growing season daily cumulative GDD values into .nc files in local disk
for (key_ob, value_ob), (key_fc, value_fc) in zip(forecast_GDD_dict_ob.items(), forecast_GDD_dict_fc.items()):
    # Convert the xarray Dataarray into xarray Dataset
    output_da_ob = value_ob.to_dataset(name = key_ob)
    output_da_fc = value_fc.to_dataset(name = key_fc)
    # Wite the CF convention to the datasets
    output_da_ob_cf = write_CF_attrs(output_da_ob) 
    output_da_fc_cf = write_CF_attrs(output_da_fc)
    # Save to disk as .nc file
    output_da_ob_cf.to_netcdf(join(output_path_GDD_ob,"{}.nc".format(key_ob)), mode='w', format="NETCDF4", engine="netcdf4")
    output_da_fc_cf.to_netcdf(join(output_path_GDD_fc,"{}.nc".format(key_fc)), mode='w', format="NETCDF4", engine="netcdf4")
# 6.2 Export the forecast skill score outputs
# 6.2.1 Create saving path
output_path_sm = join(output_path, "out_flo_ver", study_var, "simulation_fc")
output_path_ob = join(output_path, "out_flo_ver", study_var, "simulation_ob")
output_path_score = join(output_path, "out_flo_ver", study_var, "simulation_score")
output_path_score_sig = join(output_path_score, "r", "sig")
output_path_RMSE_year = join(output_path, "out_flo_ver", study_var, "simulation_RMSE_ens")
# 6.2.2 Make directories if not exist to save files
mkdir(output_path_sm)
mkdir(output_path_ob)
mkdir(output_path_score)
mkdir(output_path_score_sig)
mkdir(output_path_RMSE_year)
# 6.2.3 Save the score arrays into .nc files at local disk
for stat_name, dict_output in zip(["r", "MAE", "RMSE","fairRPS"], [forecast_r_dict, forecast_MAE_dict, forecast_RMSE_dict, forecast_fairRPS_dict]):
    output_saving_path = join(output_path_score, stat_name)
    mkdir(output_saving_path)
    for key ,value in dict_output.items():
        output_da = value.to_dataset(name = key)
        output_da = write_CF_attrs(output_da) # Wite the CF convention to the datasets
        # Save to disk as .nc file
        output_da.to_netcdf(join(output_saving_path, "{}.nc".format(key)), mode='w', format="NETCDF4", engine="netcdf4")
# 6.2.4 Save the simulation arrays with significance test results of Pearson correlation coefficient into .nc files at local disk
for key, value in forecast_r_sig_dict.items():
    output_da = value.to_dataset(name = key)
    output_da = write_CF_attrs(output_da) # Wite the CF convention to the datasets
    # Save to disk as .nc file
    output_da.to_netcdf(join(output_path_score_sig,"{}.nc".format(key)), mode='w', format="NETCDF4", engine="netcdf4")
# 6.2.5 Save the simulated forecast phenology DOY with forecast datasets into .nc files at local disk
for key ,value in forecast_ob_dict.items():
    output_da = value.to_dataset(name = key)
    output_da = write_CF_attrs(output_da) # Wite the CF convention to the datasets
    # Save to disk as .nc file
    output_da.to_netcdf(join(output_path_ob,"{}.nc".format(key)), mode='w', format="NETCDF4", engine="netcdf4")
# 6.2.6 Save the simulated reference phenology DOY with observed weather datasets into .nc files at local disk
for key ,value in forecast_sm_dict.items():
    output_da = value.to_dataset(name = key)
    output_da = write_CF_attrs(output_da) # Wite the CF convention to the datasets
    # Save to disk as .nc file
    output_da.to_netcdf(join(output_path_sm,"{}.nc".format(key)), mode='w', format="NETCDF4", engine="netcdf4")
# 6.2.7 Save the simulation data with yearly RMSE based on all ensemble members
for key ,value in forecast_yearly_score_dict.items():
    output_da = value.to_dataset(name = key)
    output_da = write_CF_attrs(output_da) # Wite the CF convention to the datasets
    # Save to disk as .nc file
    output_da.to_netcdf(join(output_path_RMSE_year,"{}.nc".format(key)), mode='w', format="NETCDF4", engine="netcdf4")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Implementation code session+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Output analysis code session++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 7 Read and load the output .nc datasets and extract each point data series into the target excel files
# 7.1 Define the essential input path
data_input_path = join(script_drive, "Grapevine_model_SeasonalForecast_FloVer", "output", "out_flo_ver") # Specify the main data input path for the analysis
output_path = join(root_path,"output") # Define the output path
# 7.2 Specify the input path for the two different varieties
TF_path = join(data_input_path, "TF")
TN_path = join(data_input_path, "TN")
TF_path_SM = join(TF_path,  "simulation_fc")
TF_path_OB = join(TF_path,  "simulation_ob")
TN_path_SM = join(TN_path,  "simulation_fc")
TN_path_OB = join(TN_path,  "simulation_ob")
# 7.3 Obtain a list of .nc files for the two different varieties
TF_SM_nc = glob.glob(join(TF_path_SM,"*.nc"))
TF_OB_nc = glob.glob(join(TF_path_OB,"*.nc"))
TN_SM_nc = glob.glob(join(TN_path_SM,"*.nc"))
TN_OB_nc = glob.glob(join(TN_path_OB,"*.nc"))
# 7.4 Define two essential dictionaries used for data output analysis
fc_names_dict = {"2sm": "Feb_1st",
                 "3sm": "Mar_1st",
                 "4sm": "Apr_1st"}
var_data = {"TF":{"OB":TF_OB_nc, "SM":TF_SM_nc},
            "TN":{"OB":TN_OB_nc, "SM":TN_SM_nc},
            }
# 7.5 Implementation of extracting simulated phenology DOY from both reference and foreacst datasets into excel files (iterations should be per variety basis)
## !! Note this represents the step to extract the raw datasets where all forecast skill analysis are based 
for variety_name in var_data.keys():
    var_data_dict = var_data[variety_name]
    # Obtain the underlying variety nc files
    var_OB_nc_list = var_data_dict["OB"]
    var_SM_nc_list = var_data_dict["SM"]
    for var_OB_nc in var_OB_nc_list: # The top level iteration should start with observations
        # Load the .nc file into xarray dataset object
        var_OB_nc_xr = xr.open_dataset(var_OB_nc, mask_and_scale=True, engine = "netcdf4", decode_times =True)
        # List the embedded variables within the dataset
        data_var_list = list(var_OB_nc_xr.data_vars)
        # Obtain the underlying variable name from the dataset
        OB_var_name = [data_var_name for data_var_name in data_var_list if ("_flo" in data_var_name) or ("_ver" in data_var_name)][0] # Unpack the list
        # Access the underlying xarray dataarray object
        var_OB_nc_xr_arr = var_OB_nc_xr[OB_var_name]
        # Obtain the studied variable name
        var_name = "flo" if "_flo" in OB_var_name else "ver" 
        # Get the main saving path
        save_file = join(data_input_path, variety_name, "{}_{}_pheno_data.xlsx".format(variety_name, var_name))
        # if "TF" in var_OB_nc:
        #     save_file = join(data_input_path, "TF", "TF_{}_pheno_data.xlsx".format(var_name))
        # elif "TN" in var_OB_nc:
        #     save_file = join(data_input_path, "TN", "TN_{}_pheno_data.xlsx".format(var_name))
        # Get the saving excel file
        excel_writer = pd.ExcelWriter(save_file)
        for var_SM_nc in var_SM_nc_list: # With observations read in place, forecast datasets initialized in different dates will be applied
            if "5sm" in var_SM_nc: # Skip the forecast dataset that is initialized on May 1st
                continue
            var_SM_nc_xr = xr.open_dataset(var_SM_nc, mask_and_scale=True, engine = "netcdf4", decode_times =True)
            if any(var_name in data_var_name for data_var_name in list(var_SM_nc_xr.data_vars)): # Only iterate through the forecast simulation files that share the same variable name as those of observations
                SM_var_name = [data_var_name for data_var_name in list(var_SM_nc_xr.data_vars) if var_name in data_var_name][0] # Unpack the list
                # Access the underlying xarray dataarray object for forecast dataset
                var_SM_nc_xr_arr = var_SM_nc_xr[SM_var_name]
                # Obtain the saving sheet_name for the excel files
                for key in fc_names_dict.keys():
                    if key in SM_var_name:
                        save_sheet_name = fc_names_dict[key] # The sheet name is based on the varname in the forecast dataset
                # Get the lon and lat names from the xarray object
                lon_name_ob, lat_name_ob = get_latlon_names(var_OB_nc_xr_arr)
                lon_name_fc, lat_name_fc = get_latlon_names(var_SM_nc_xr_arr)
                if lon_name_ob==lon_name_fc:
                    lon_name = lon_name_ob
                if lat_name_ob==lat_name_fc:
                    lat_name = lat_name_ob
                # Compute the lon and lat vectors 
                lon_vector = var_OB_nc_xr_arr[lon_name].data if np.array_equal(var_OB_nc_xr_arr[lon_name],var_SM_nc_xr_arr[lon_name],equal_nan=True) else np.nan
                lat_vector = var_OB_nc_xr_arr[lat_name].data if np.array_equal(var_OB_nc_xr_arr[lat_name],var_SM_nc_xr_arr[lat_name],equal_nan=True) else np.nan
                # Form the list of coordinates to be studied
                coordinates = list(product(lon_vector, lat_vector)) 
                grid_id = 0 # Define the grid ID with 0 based
                # Iterete over all coordinate grid points to extract the point timeseries values
                for coordinate in coordinates:
                    # Unpack the coordinate tuple to obtain the longitude and latitude of a given coordinate
                    lon1 = round(coordinate[0], 1) # Round to 1 decimal digit
                    lat1 = round(coordinate[1], 1) # Round to 1 decimal digit
                    # Extract a full-timeseries of data at the specified grid point for the reference phenology series
                    ob_data = var_OB_nc_xr_arr.sel({lon_name:lon1, lat_name:lat1}, method="nearest")
                    # Obtain the observational series
                    ob_data_ser = pd.Series(ob_data, index = ob_data.time.dt.year.data, name ="obs")
                    # Set the -999/0 values as the np.nan
                    ob_data_ser = fill_na(ob_data_ser, -999)
                    ob_data_ser = fill_na(ob_data_ser, 0)
                    if all(np.isnan(ob_data_ser)):
                        continue # Skip the point with all values NaN
                    elif any(np.isnan(ob_data_ser)): # In case of containing some missing values
                        ob_data_ser = check_NaN(ob_data_ser, fill_extra=False)
                    fc_data_dict = {} # Create an empty dict to store forecast datasets
                    # Extract a full-timeseries of data at the specified grid point for the forecast phenology series
                    for ens_member in var_SM_nc_xr_arr.coords["number"].data:
                        # Obtain the full-series data of grid point for the forecast phenology data
                        sm_ens_data = var_SM_nc_xr_arr.sel({lon_name:lon1, lat_name:lat1, "number":ens_member}, method="nearest").to_series()
                        # Set the -999/0 values as the np.nan
                        sm_ens_data = fill_na(sm_ens_data, -999)
                        sm_ens_data = fill_na(sm_ens_data, 0)
                        if all(np.isnan(sm_ens_data)):
                            continue # Skip the point with all values NaN
                        elif any(np.isnan(sm_ens_data)): # In case of containing some missing values
                            sm_ens_data = check_NaN(sm_ens_data, fill_extra=False)
                        # Attach the forecast dataset from a given forecast ensemble member into the target dict
                        fc_data_dict["ens_"+str(ens_member+1)] = sm_ens_data
                    grid_id = grid_id +1
                    # Convert the fc_data_dict into dataframe
                    fc_data_df = pd.DataFrame(fc_data_dict)
                    # Re-set the df index using the year information
                    fc_data_df.index = fc_data_df.index.year
                    # Concatenate with observational series to have the final df
                    output_df = pd.concat([ob_data_ser, fc_data_df], axis=1, join='inner', ignore_index=False)
                    output_df.index.name = "year" # Set the index name
                    # Reset the index  
                    output_df_intermediate= output_df.reset_index()
                    # Add the grid ID and coordinate lable columns
                    output_df_intermediate["grid_id"] = grid_id
                    # Add the coordinate lable columns
                    output_df_intermediate["lon"] = lon1
                    output_df_intermediate["lat"] = lat1
                    # Re-order the columns
                    output_df_final = output_df_intermediate.reindex(columns = ["grid_id", "lon", "lat"] + 
                                      [ col_name for col_name in output_df_intermediate.columns if col_name not in ["grid_id", "lon", "lat"] ], copy=True)
                    # Save the df to excel file
                    if grid_id == 1: # Only Write the first gridID header
                        header = True
                        start_row = 0
                    else:
                        header = False
                        start_row =  ((grid_id-1) * output_df_final.shape[0]) +1
                    output_df_final.to_excel(excel_writer, sheet_name=save_sheet_name, startrow=start_row, startcol=0, header=header, index=False, engine= "openpyxl")
        # Save the excel file into local disk
        excel_writer.save()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 8. Read and load the fairRPS score data and export it into .nc files
# 8.1 Define the path to fairRPS scores that are pre-computed
data_input_path = join(script_drive, "Grapevine_model_SeasonalForecast_FloVer", "output", "out_flo_ver") # Specify the main data input path for the analysis
study_vars = ["TF","TN"] # Define a list of study varieties
study_stages = ["flo", "ver"] # Define two study phenology stages
study_fc_init_month = ["Feb", "Mar", "Apr"] # Define a list of foreacst initialization months
study_fc_init_month_int = { fc_init_month: (i +2) for i, fc_init_month in enumerate(study_fc_init_month) } # Define a dict of forecast initialization month integers
# 8.2 Define the coordinate array. Note the target_points are inferred from the section 2 
lon_target_vector = np.unique([round(target_point.x, decimal_place) for target_point in target_points]) # A unique list of longitude
lat_target_vector = np.unique([round(target_point.y, decimal_place) for target_point in target_points]) # A unique list of latitude
coords_xarray = [ ("lat", lat_target_vector), ("lon", lon_target_vector)] # Create the coordinate dimensions ("time",time_vector)
# 8.3 Define the template xarray files that are aimed for storing different kinds of output dataset
fairRPS_output = xr.DataArray(coords=coords_xarray) # Create a dimensional template xarray object that is going to be used as the output structure
lon_name, lat_name = get_latlon_names(fairRPS_output) # Get the lon and lat names of the xarray object 
decimal_place = 1 # Define the number of decimal place
#output_path = join(root_path,"output") # Define the output path
# 8.4 Iterate over each study variety
for study_var in study_vars:
    # Access the fairRPS data path for a given variety
    study_var_data_path = join(data_input_path, study_var, "simulation_score", "fairRPS")
    # Iterate over each phenology stage
    for study_stage in study_stages:
        study_var_data_stage_path = join(study_var_data_path, study_stage) # Access the stage-variety specific fairRPS data
        # Collect a list of .csv files that contain the computed fairRPS scores
        fairRPS_csv_list = glob.glob(join(study_var_data_stage_path,"*.csv"))
        # Iterate over each forecast initialization month
        for fc_init_mont in study_fc_init_month:
            # Define the variable name by combining the phenology stage and forecast initilization month  
            var_name_arr = str(study_fc_init_month_int[fc_init_mont]) +"_" + study_stage
            # Have a deep copy of template fairRPS xarray object
            fairRPS_output_copy = fairRPS_output.copy(deep=True)
            # Name the copied fairRPS xarray object first
            fairRPS_output_copy.name = var_name_arr
            # Access the variety-stage data path
            fairRPS_var_stage = [fairRPS_csv_file for fairRPS_csv_file in fairRPS_csv_list if fc_init_mont + "_1st" in fairRPS_csv_file][0]
            # Load the .csv file into dataframe
            fairRPS_var_stage_df = pd.read_csv(fairRPS_var_stage, sep=";", header=0)
            # Get the lon and lat names # It is assumed the lon and lat colnames in the df are identical to xarray lon and lat names
            if any(lon_name==col_name for col_name in fairRPS_var_stage_df.columns) & any(lat_name == col_name for col_name in fairRPS_var_stage_df.columns): 
                lon_col_name = [col_name for col_name in fairRPS_var_stage_df.columns if lon_name in col_name][0]
                lat_col_name = [col_name for col_name in fairRPS_var_stage_df.columns if lat_name in col_name][0]
            else:
                raise ValueError("The longitude and latitude names are not identical between the csv files and the xarray object")
            # Iterate over each grid point to extract the fairRPS score
            for row_idx, row_ser in fairRPS_var_stage_df.iterrows():
                # Obtain the grid point longitude and latitude
                lon1 = round(row_ser[lon_col_name], decimal_place)
                lat1 = round(row_ser[lat_col_name], decimal_place)
                # Obtain the fairRPS score value for a given grid point
                fairRPS_score = row_ser["score"]
                # Assign the fairRPS value for the particular grid point
                fairRPS_output_copy.loc[{lon_name:lon1,lat_name:lat1}] = float(fairRPS_score)
            # Export the xarray data array object into local disk
            output_da = fairRPS_output_copy.to_dataset(name = fairRPS_output_copy.name)
            output_da_crs = write_CF_attrs(output_da) # Wite the CF convention to the datasets
            # Save to disk as .nc file
            output_da_crs.to_netcdf(join(study_var_data_path,"{}.nc".format(fairRPS_output_copy.name)), mode='w', format="NETCDF4", engine="netcdf4")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Output analysis code session++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
########################################################################### Coding Blocks #################################################################################################################################