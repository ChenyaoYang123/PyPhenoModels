import xarray as xr
import os
import os.path
import pandas as pd
import numpy as np
import re
import decimal
import time
# import matplotlib.pyplot as plt 
# # import matplotlib.ticker as mtick
# import plotly.express as px
# import plotly.graph_objects as go
import glob
import os
import sys
import getpass
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib_scalebar.scalebar import ScaleBar
from itertools import product
from os.path import join,dirname
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from shapely.geometry import Point
#%matplotlib inline
# Append the script path to system path
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define the disk drive letter according to the username
if getpass.getuser() == 'Clim4Vitis':
    script_drive = "H:\\"
    #shape_path = r"H:\Grapevine_model_GridBasedSimulations_study4\shapefile"
elif getpass.getuser() == 'admin':
    script_drive = "G:\\"
elif (getpass.getuser() == 'CHENYAO YANG') or (getpass.getuser() == 'cheny'):
    script_drive = "D:\\"
target_dir = r"Mega\Workspace\Study for grapevine\Study6_Multi_phenology_modelling_seasonal_forecast\script_collections" # Specific for a given study
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
add_script(script_drive, target_dir)
from Multi_phenology_model_classes import * # Add the multi-phenology model class script

class Timer():
    
    def start(self):
        print(f"[BENCHMARK] Start Time - {datetime.now()}")
        self._start_time = time.perf_counter()

    def _duration(self):
        duration = timedelta(seconds=time.perf_counter() - self._start_time)
        print(f"[BENCHMARK] Total Duration - {duration}")

    def end(self):
        self._duration()
        print(f"[BENCHMARK] End Time - {datetime.now()}")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def mkdir(dir = None):
    #'''
    #Creates the given directory.

    #Parameters
    #----------
    #dir : char
    #       Directory
    #'''
    if not dir is None:
        if not os.path.exists(dir):
            os.makedirs(dir)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def extract_dim_name(ds_nc):
    '''
    Extract the dimensional names from the xarray object
    
    Parameters
    ----------
    ds_nc : xarray dataset/dataarray, the input xarray dataset to work with
    '''
    for dimension in list(ds_nc.coords.keys()):  
        if "lon" in dimension: # The minimal naming convenction for longitude is lon, but can be possible with full name as longitude
            lon_name="".join(re.findall(r"lon\w*",dimension)) # Get the exact longitude name 
        elif "lat" in dimension: # The minimal naming convenction for latitude is lat, but can be possible with full name as latitude
            lat_name="".join(re.findall(r"lat\w*",dimension)) # Ger the exact latitude name
    return lon_name, lat_name
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def extract_site_data_series(ncfile,nc_var_identifier,lon1,lat1,time_period,method="nearest", _format="netCDF"):
    '''
    Extract site-specific time-series data. Note this function only works for nc file with one variable each

    Parameters
    ----------
    ncfile : path-like object
    nc_var_identifier: str, to identify the actual variable name in nc file
    lon1: longitude in deicmal degree
    lat1: latitude in deicmal degree
    time_period: a target period to extract time-series data
    method: optional, the built-in xarray method to approximate the target grid
    '''
    if _format == "netCDF":
        ds_nc=xr.open_dataset(ncfile,mask_and_scale=True) # Always process the original data if scale_factor and add_offset exists
    elif _format == "GRIB":
        ds_nc=xr.open_dataset(ncfile,mask_and_scale=True, engine="cfgrib") # Always process the original data if scale_factor and add_offset exists
        
    # Obtain the actual coordinate/dimension names, e.g. the latitude cooridnate variable could be "latitude" or "lat" 
    # Note list(ds.dims.keys()) would yield the same results
    lon_name, lat_name = extract_dim_name(ds_nc)
    # Retrive the lat and lon vectors
    #lon_vector=ds_nc[lon_name].data
    #lat_vector=ds_nc[lat_name].data
    # Obtain the underlying variable name contained in the nc file
    var_nc_name=[item for item in list(ds_nc.data_vars) if nc_var_identifier in item][0]
    # Retrive the underly DataArray given the variable name
    var_dict=ds_nc[var_nc_name] 
    # Extract the climate data given the site coordinates and study period
    site_data=var_dict.sel({lon_name:lon1,lat_name:lat1,"time":time_period},method=method).values
    
    # Check units and convert unit to standard one
    if "units" in ds_nc.attrs.keys():
        unit_attr = ds_nc.attrs["units"]
        if unit_attr in ["K","k","Kelvin","kelvin","KELVIN"]: 
            site_data = site_data-273.15 # Transform the kelvin to celcius
        elif "s-1" in unit_attr:
            site_data = site_data*3600*24
    elif "standard_name" in ds_nc.attrs.keys():
        standard_name_attr =  ds_nc.attrs["standard_name"]
        if standard_name_attr== 'precipitation_flux':
            site_data = site_data*3600*24 # Transform the precipitation flux to precipitation amount/day
    # To further process precipitation datasets
    if ("rr" in var_nc_name) or ("pr" in var_nc_name):
        site_data = np.array(site_data)
        site_data[site_data<0.5] = 0 # For any precipitation amount that is below 0.5 mm, set it to 0 mm.    
    # Convert the output in one-D array into pd.Series
    site_data_ser = pd.Series(site_data,index=time_period,name=nc_var_identifier)
    
    return site_data_ser
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def as_text(value):
    '''
    Convert any cell value into text
    Parameter
    ----------
    value : cell value
    '''
    if value is None:
        return ""
    else:
        value=str(value)
    return value
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def consecutive_merge_df(iterable,how="outer",on=None,copy=None):
    '''
    Merge df consecutively by choosing a set of common columns (used as join keys) that are available in both df 
    
    Parameter
    ----------
    iterable : list or tuple, an iterable of df to merge. Not implemented for dict type 
    how: str, method of merging df. See panda documentations
    on: str,  a common set of columns available in all underlying df that are used as join keys
    copy: bool, if make a copy or not.
    '''
    # Check if the input typ is a list or tuple
    if not isinstance(iterable,(list,tuple)):
        raise TypeError("The input iterable is not a required data type, only list or tuple is supported, but {} is found".format(type(iterable))) 
    for i in range(len(iterable)):
        if i != (len(iterable)-1):
            iterable[i+1] = iterable[i].merge(iterable[i+1], how=how, on=on, copy=copy) # Update the next df in place from the input iterable
        else:
            continue
    # Obtain the final df 
    final_df = iterable[-1] # With this method, only the last element from the iterable is required.
    return final_df
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def collect_weather_data(target_datasets, lon, lat, time_period, dataset_loaded= False, var_identifiers = ['tx', 'tg', 'tn',"pr"], target_column_order = ['day','month','Year', 'tx', 'tg', 'tn', "pr", 'Lat','Lon'], **kwargs):
    '''
    Extract and collect the site-specific weather data from supplied gridded datasets with well-formated data columns
    
    Parameter
    ----------
    target_datasets : list/dict, a list or dictionary of target gridded datasets in netCDF format
    lon: float, the longitude of a grid point
    lat: float, the latitude of a grid point
    time_period: 1-D array, the study years 
    var_identifiers: list, a list of variable short name that stand for each variable
    target_column_order: list, the target order of column in the final dataframe  
    kwargs: any addtional key word arguments
    '''
    target_data_list =[] # Define an empty list to which site-specific series data is appended
    if not dataset_loaded: # If the datasets have not been pre-loaded
        # The .nc datasets are not pre-loaded
        for grid_data in target_datasets:
            if any([var in grid_data for var in var_identifiers]):
                var_shortname = [var for var in var_identifiers if var in grid_data][0] # Extract the variable short name
                timestamp_arr = xr.open_dataset(grid_data, mask_and_scale=True, engine = "netcdf4").time # Extract the full timestamp from the supplied dataset. Note timestamp_arr is an xarray dataarray object (1-D with time dim only)
                timestamp_select = timestamp_arr.where(timestamp_arr.time.dt.year.isin(time_period),drop=True) # Drop values in years that are not within the study years
                data_series = extract_site_data_series(grid_data,var_shortname,lon,lat,timestamp_select.data) # Extract the time series of data for a given site
                target_data_list.append(data_series) # Append to an existing empty list
            else:
                continue
    else: # In case the datasets have been pre-loaded, it must be a dictionary with key given for the var name, value for xarray dataarray
        assert isinstance(target_datasets, dict), "input meteorological datasets do not follow a dictionary format"
        assert all([isinstance(target_dataset, (xr.core.dataset.Dataset, xr.core.dataarray.DataArray)) for target_dataset_name, target_dataset in target_datasets.items()]), "one of the input meteorological dataset format is wrong"
        for target_dataset_name, target_dataset_array in target_datasets.items():
            # Extract the target longitude and latitude names
            lon_name, lat_name = extract_dim_name(target_dataset_array)
            # Drop years that are not within the study years
            timestamp_select = target_dataset_array.time.where(target_dataset_array.time.dt.year.isin(time_period),drop=True)
            # Select site-specific timeseries for a given pair of lon1 and lat1
            # Confirm the extract_method is specified
            if "extract_method" in kwargs.keys():
                method = kwargs["extract_method"]
            else:
                raise KeyError("The additional keyword argument 'extract_method' not found") 
            site_data = target_dataset_array[target_dataset_name].sel({lon_name:lon, lat_name:lat, "time":timestamp_select.data}, method=method).data
            # Check if there is any empty dimension
            if site_data.ndim != 1:
                site_data=np.squeeze(site_data)
            # Check units and convert unit to standard one
            if "units" in target_dataset_array.attrs.keys():
                unit_attr = target_dataset_array.attrs["units"]
                if unit_attr in ["K","k","Kelvin","kelvin","KELVIN"]: 
                    site_data = site_data-273.15 # Transform the kelvin to celcius
                elif "s-1" in unit_attr:
                    site_data = site_data*3600*24
            elif "standard_name" in target_dataset_array.attrs.keys():
                standard_name_attr =  target_dataset_array.attrs["standard_name"]
                if standard_name_attr== 'precipitation_flux':
                    site_data = site_data*3600*24 # Transform the precipitation flux to precipitation amount/day
            # To further process precipitation datasets
            if ("rr" in target_dataset_name) or ("pr" in target_dataset_name):
                site_data = np.array(site_data)
                site_data[site_data<0.5] = 0 # For any precipitation amount that is below 0.5 mm, set it to 0 mm.
            #### To be futher added if additional attributes area found in the nc files that are needed to transform the data
            site_data_ser = pd.Series(site_data,index=timestamp_select.data,name=target_dataset_name)
            # Append to an existing empty list
            target_data_list.append(site_data_ser) 
    # Concatenate all list datasets 
    merged_df = pd.concat(target_data_list, axis=1, join='inner', ignore_index=False)
    # Create a df with all desired columns
    merged_df = merged_df.assign(day=lambda x:x.index.day, month=lambda x:x.index.month, Year= lambda x:x.index.year,
                     lon=lon, lat=lat)
    # Reorder the columns
    merged_df_final = merged_df.reindex(columns = target_column_order, copy=True)
    # Check for missing values
    if merged_df_final.isnull().values.any():
        merged_df_final.fillna(method="ffill",inplace=True)
    return merged_df_final
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    import matplotlib.pylab as plt
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap,N)
    # Colormap always map values between 0 and 1 to a bunch of colors.
    # base(0.5). The colormap object is callable, which will return a RGBA 4-element tuple
    return base
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def determine_vmin_vmax(data_arrays,p_low=5,p_high=95):
    '''
    # Iterate the input xarray DataArray and determine the vmin and vmax

    # Parameters
    #----------
    # data_arrays : list of input DataArray to determine the vmin and vmax
    # 
    '''
    cbar_vmin = np.inf # Set the positive infinity
    cbar_vmax = np.NINF # Set the negative inifinity
    assert isinstance(data_arrays,list), "the input list of array is not found" 
    # Iterate over each array stored in the list of arrays
    for data_array in data_arrays:
        # Calculate the vmin and vamx from the data array
        vmin = float(np.nanpercentile(data_array.data.flatten(),p_low)) 
        vmax = float(np.nanpercentile(data_array.data.flatten(),p_high))
        # Compare the cbar_vmin with extracted vmin
        if cbar_vmin >= vmin:
            cbar_vmin = vmin
        # Compare the cbar_vmax with extracted vmax
        if cbar_vmax <= vmax:     
            cbar_vmax = vmax
    return cbar_vmin,cbar_vmax
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_latlon_names(ds_array):
    """
    Extract the latitude and longitude vectors from the supplied Xarray DataArray
    
    Parameters
    ----------
    ds_array: input Xarray dataarray
    """
    # Define the regular expression patterns
    lat_regex=re.compile(r"lat\w*") 
    lon_regex=re.compile(r"lon\w*") 
    # Search for the actual lat and lon name in the datasets
    for dimension in list(ds_array.coords.keys()):  
        if "lon" in dimension: # The minimum name convention is "lon"
            lon_name=lon_regex.findall(dimension)[0]  # Get the exact longitude name 
        elif "lat" in dimension: # The minimum name convenction is lat
            lat_name=lat_regex.findall(dimension)[0]  # Ger the exact latitude name
    # Retrieve the lat and lon vectors
    # lat_vec=ds_array[lat_name]
    # lon_vec=ds_array[lon_name]
    return (lon_name,lat_name)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def extract_fc_dataset(data_path, fc_var, ob_ser, lon1, lat1, ensemble_method=None, reduction_dim = None, **kwargs):
    '''
     Extract forecast datasets for a given point specified by lon and lat, while complement the gap period with observed data
      
     Mandatory Parameters
     ----------
     data_path : list of data path, a list of path pointing to target forecast dataset at a specific forecast time/date/month.
     fc_var: str, the meteorological variable name of the forecast dataset
     ob_ser: series, a series of extracted observed weather for this point
     lon: float, longitude of the site
     lat: float, latitude of the site
     
     Optional Parameters
     ----------
     ensemble_method: str, indicate how to compute the required statistical metric for all the ensemble members of seasonal forecast dataset 
     reduction_dim: str, indicate the dimension (ensemble memeber) that will be reduced by a specific ensemble method.     
    '''      
    assert isinstance(data_path, list), "a list of input data path is required, but the format {} is found".format(type(data_path))
    # Read the list of data path into xarray object that varied with different time span
    list_datasets = [xr.open_dataset(var_path, mask_and_scale=True, engine = "netcdf4") for var_path in data_path]
    # Concatenate all the xarray dataset objects into a signle object with the full time-series
    xr_concat = xr.concat(list_datasets, dim = "time", data_vars ="all",
                             coords = "minimal", join="inner", compat = "equals", combine_attrs = "override")
    # Get the longitude and latitude name from the supplied dataset
    lon_name, lat_name = get_latlon_names(xr_concat) 
    # Calculate the ensemble statistics of all ensemble members for the forecast dataset
    if ensemble_method in ["mean", "MEAN", "Mean","Average", "average"]:
        xr_concat_ensemble_stat = xr_concat.mean(dim = reduction_dim, keep_attrs = True, skipna = True) # Compute the ensemble mean over all ensemble members
    elif ensemble_method in ["MEDIAN", "median", "Med","MED"]:
        xr_concat_ensemble_stat = xr_concat.quantile(0.5, dim = reduction_dim, keep_attrs = True, skipna = True) # Compute the ensemble median over all ensemble members
    elif (ensemble_method is None) & (reduction_dim is None):
        xr_concat_ensemble_stat = xr_concat.copy(deep=True)
    # Extract the full time series of forecast data for a given point and convert it to series
    if reduction_dim != None:
        fc_ser = xr_concat_ensemble_stat[fc_var].sel({lon_name:lon1, lat_name:lat1}, method="nearest").to_series()
    else:
        target_dim_name = [dim_name for dim_name in xr_concat_ensemble_stat.dims.keys() if dim_name not in [lon_name, lat_name, "time"]][0] # Determine the dimension name for the ensemble memeber
        assert "ensemble_member" in kwargs.keys(), "missing additional keyword argument 'ensemble_member'"
        fc_ser = xr_concat_ensemble_stat[fc_var].sel({lon_name:lon1, lat_name:lat1, target_dim_name:kwargs["ensemble_member"]}, method="nearest").to_series()
    # Subset observed weather to years compatible with seasonal forecast data
    ob_ser_subset = ob_ser.loc[ob_ser.index.year.isin(fc_ser.index.year.unique())] 
    # Concatenate two series into one df with matching datetime index
    ob_fc_df = pd.concat([ob_ser_subset, fc_ser], axis=1, join="outer", ignore_index=False) 
    # Fill the forecast data gap values from observed data
    fc_ser_filled = ob_fc_df[fc_var].fillna(ob_fc_df[ob_ser.name]) # Provide this column to fillna, it will use those values on matching indexes to fill:
    
    return fc_ser_filled
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_models(T_input):
    """
    Run the model with prescribed temperature inputs and return simulated flowering and veraison DOY
    
    Parameter    
    ----------
    T_input: series, series of temperature input to drive the model simulations
    # Bug_dict: dict, a dictionary to store the simulated data with NaN values
    # lon1: float, longitude of the location
    # lat1: float, latitude of the location
    """
    # Starting from the first day of year
    DOY_point_flo = phenology_model_run(T_input, thermal_threshold=26.792, module="sigmoid", a=-0.15058, b=23.72417, 
                                            DOY_format=True, from_budburst= False, T0 =1) 
    # Starting from the simulated flowering DOY
    DOY_point_ver = phenology_model_run(T_input, thermal_threshold=75.419, module="sigmoid", a=-39.99993, b=15.4698, 
                                            DOY_format=True, from_budburst= False, T0 = DOY_point_flo.copy(deep=True))

    return DOY_point_flo, DOY_point_ver 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def sort_FCdatasets_by_FCdate(fc_path, fc_times, metero_var_dict_forecast= {"Tmin":"mn2t24", "Tmax":"mx2t24", "Prec":"tp"}, fc_type="monthly", data_format=".nc"):
    """
    Sort the forecast datasets by different forecast month/date
    
    Parameter    
    ----------
    fc_path: str, path or path-like objects pointing to the target forecast datasets files with format specified by data_format
    fc_times: list, a list of different forecast time (month/date) 
    metero_var_dict_forecast: dict, a dict of meteorological variables with full name and abbreviation names
    fc_type: str, str indicating the forecast data type. For seasonal forecast, the type should be monthly or seasonal; For sub-seasonal forecast, this can be daily.
    data_format: str, forecast dataset format
    """

    # Obtain a list of total seasonal forecast datasets
    total_datasets = glob.glob(join(fc_path,"*"+data_format)) # Collect all datasets
    # Sort the datasets by variable name that is embedded in the .nc file 
    total_datasets_by_var = {}  
    for metero_var, metero_var_abb in metero_var_dict_forecast.items():
        total_datasets_by_var[metero_var] = [dataset for dataset in total_datasets if metero_var_abb in dataset]
    # Sort the datasets according to different forecast time/date
    # Create an empty forecast time dictionary to store the sorted results   
    forecast_time_data = {} 
    # Iterate over different forecast time
    for fc_time in fc_times:
        if str(fc_time) not in forecast_time_data.keys():
            forecast_time_data[str(fc_time)] = {}
        for metero_var, datasets in total_datasets_by_var.items():
            target_data = [] # Create an empty list to store the target dataset
            for data in datasets: # Iterate over each dataset and test if the dataset belongs to a specific month 
                da = xr.open_dataset(data, mask_and_scale=True, engine = "netcdf4")
                if fc_type=="monthly": # Specify the fc type
                    fc_init_time = da.time[0].dt.month.data # Check if the month of the first forecast date is within the pre-specified forecast month, attach the data path into target dictionary
                    if fc_init_time==fc_time:
                    # Append the target dataset path to the empty list
                        target_data.append(data)
                        da.close() # Close the current dataset handler
                    else:
                        da.close() # Close and continue into the next iteration
                        continue
            # Attach the list of collected data path into the target dictionary
            forecast_time_data[str(fc_time)][metero_var] = target_data 
    return forecast_time_data
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def check_NaN(ser, fill_method1="quadratic",fill_method2="bfill",fill_method3="ffill"):
    """
    Check if there are any NaN values in the simulated series, if yes, fill NaN
    """
    if any(pd.isnull(ser)):
        ser.interpolate(method=fill_method1, limit_direction="both", inplace=True)
        if any(pd.isnull(ser)):
            ser.fillna(method=fill_method2, inplace=True)
        if any(pd.isnull(ser)):
            ser.fillna(method=fill_method3, inplace=True)
        return ser
    else:
        #print("No NaN values are deteced, thereby no filling NaN is performed")
        return ser
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. Specify all user-dependent path 
# 1.1 Path for local meteorological files
root_path=r"H:\Grapevine_model_SeasonalForecast_Budburst_study6" # Main path
E_OBS_path = r"H:\E_OBS_V24" # Define the E-OBS dataset path
forecast_path = r"H:\Grapevine_model_SeasonalForecast_Budburst_study6\Forecast_datasets_ECMWF" # Define the forecast dataset path
output_path = join(root_path,"output") # Define the output path
meta_data_path = join(root_path,"metadata") # Metadata path
var_identifiers_EOBS = ['tx', 'tg', 'tn'] # Define a list of variable short names for E-OBS dataset 
target_dataset = [] # Define an empty list to collect target gridded datasets
for ncfile in glob.glob(join(E_OBS_path,"*.nc")):
    if any(varname in ncfile for varname in var_identifiers_EOBS):
        target_dataset.append(ncfile)
# 1.2 Path for target shape file
shape_path = join(script_drive, r"Mega\Workspace\SpatialModellingSRC\GIS_root\Shape") # Define the path to target shapefile 
study_shapes = glob.glob(join(shape_path,"*.shp")) # Obtain a list of shape files used for defining study region
study_region = "PT_continent" # Define the name for study region. 
study_shape = [shape for shape in study_shapes if study_region in shape][0] # Squeeze the list
proj = ccrs.PlateCarree() # Define the target projection CRS
GDF_shape= gpd.read_file(study_shape).to_crs(proj) # Load the shape file into a geographic dataframe
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 2. Collect a list of points that comprise the study region
determine_lat_lon_vector = "by_shape" # Collect data points from shape file or from forecast dataset itself
fc_times = [2, 3, 4]  # Specify the forecast time either by month or by day. But this should be case-specific
forecast_time_data = sort_FCdatasets_by_FCdate(forecast_path, fc_times) # Obtain the target forecast dict
metero_var_dict_forecast= {"Tmin":"mn2t24", "Tmax":"mx2t24", "Prec":"tp"} # Define the mteorological variable dictionary

if determine_lat_lon_vector=="by_shape":
    # GDF_shape = GDF_shape[(GDF_shape["name"]!="Vinho Verde") & (GDF_shape["name"]!="Douro")] # Sampling the geodataframe so that only Lisbon wine region is studied
    # Extract the boundary rectangle coordinates and define the study resolution 
    minx, miny, maxx, maxy = GDF_shape.total_bounds
    resolution = 0.1 # Define the study resolution
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
elif determine_lat_lon_vector=="by_dataset": # Infer the study coordinates from the dataset coordinate
    # Obtain a list of total seasonal forecast datasets
    total_datasets = glob.glob(join(forecast_path,"*.nc")) # Collect all datasets
    # Sort the datasets by variable name that is embedded in the .nc file 
    total_datasets_by_var = {}  
    for metero_var, metero_var_abb in metero_var_dict_forecast.items():
        total_datasets_by_var[metero_var] = [dataset for dataset in total_datasets if metero_var_abb in dataset]
    forecast_time_lonlat = {} #  Create an empty forecast time dictionary to store the lat lon vector
    for forecast_month in fc_times:
        if str(forecast_month) not in forecast_time_lonlat.keys():
            forecast_time_lonlat[str(forecast_month)] = {}
        for metero_var, dataset in total_datasets_by_var.items():
            target_lonlat = [] # Create an empty list to store the target longitude and latitude
            for data in dataset: # Iterate over each dataset and test if the dataset belongs to a specific month 
                da = xr.open_dataset(data, mask_and_scale=True, engine = "netcdf4")
                if da.time[0].dt.month.data == forecast_month: # If the first forecast date is within the forecast month, attach the data path into target dictionary
                    # Check the lat, lon coordinates
                    lon_name, lat_name = get_latlon_names(da) # Extract the longitude and latitude name
                    lon_vector, lat_vector = da[lon_name].data, da[lat_name].data
                    # Append the lon, lat vector to the empty list
                    target_lonlat.append((lon_vector, lat_vector))
                    da.close() # Close the current dataset handler
                else:
                    da.close() # Close and continue into the next iteration
                    continue
            # Attach the list of collected data path into the target dictionary
            forecast_time_lonlat[str(forecast_month)][metero_var] = target_lonlat
    # Check if all lat lon vectors are the same across different datasets
    for forecast_month in forecast_time_lonlat.keys():
        for metero_var in forecast_time_lonlat[forecast_month].keys():
            list_of_arrays = forecast_time_lonlat[forecast_month][metero_var]
            list_of_lon = [tuple_array[0] for tuple_array in list_of_arrays] # The first tuple element is the longitude vector
            list_of_lat = [tuple_array[1] for tuple_array in list_of_arrays] # The second tuple element is the latitude vector
            # Check equality 
            for coordinate_vector in [list_of_lon, list_of_lat]:
                check_bool = np.diff(np.vstack(coordinate_vector).reshape(len(coordinate_vector),-1), axis=0)==0
                if not check_bool.all():
                    print("Coordinate vectors are not equal across datasets")
                else:
                    print("Equal coordinate vectors are found across datasets")
    # Randomly take one dataset (first one) to calculate the studied longitude and latitude vectors (as all datasets share the same coordinates)
    first_data_path = forecast_time_data["2"]["Tmin"][0] # Access the first foreacst data path
    da = xr.open_dataset(first_data_path, mask_and_scale=True, engine = "netcdf4") # Open the dataset
    lon_name, lat_name = get_latlon_names(da) # Extract the longitude and latitude name
    first_data_lonlat = forecast_time_lonlat["2"]["Tmin"][0] # Obtain the first longitude and latitude coordinate
    coordinates = list(product(first_data_lonlat[0], first_data_lonlat[1])) # Form the list of coordinates to be studied
    target_points = []
    for coordinate in coordinates:
        # Check and only use data points with no NaN values
        if not any(pd.isnull(da[list(da.data_vars.keys())[0]].sel({lon_name:coordinate[0], lat_name: coordinate[1]}).data).flatten()):
            grid_point = Point(coordinate)
            target_points.append(grid_point)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 3. Define essential inputs and output locations
# 3.1 Define the study years and desired column order for the weather data
begin_year = 1993 # The starting year needs to be one year before the actual starting year  
end_year = 2017
study_period = np.arange(begin_year,end_year+1,1)
date_range_output= pd.date_range("{}-12-31".format(begin_year), "{}-12-31".format(end_year), freq="Y") # Date index written into output file
metero_var = ["tn","tg","tx"] # Define the meteorological variable that is going to be the input for the phenology model
metero_var_dict_OB = {"Tmin":"tn", "Tmean":"tg", "Tmax":"tx"}
target_column_order = ['day','month','Year', *metero_var, 'lat','lon'] # Pre-specified an order of columns for meteorological data
forecast_time = [2, 3, 4]  # Define the forecast month; This should be case-specific
ensemble_members = 25 # Define the number of ensemble used in the seasonal forecast
# 3.2 Create the annual output template to save results
#time_vector = pd.date_range("{}-01-01".format(begin_year), "{}-12-31".format(end_year), freq="D")
#time_vector = pd.date_range("{}-12-31".format(str(begin_year)), "{}-12-31".format(str(end_year)), freq="Y")
# 3.2.1 Generate vector values for each dimensional coordinate
time_vector = study_period
lon_target_vector = np.unique([target_point.x for target_point in target_points]) # A unique list of longitude
lat_target_vector = np.unique([target_point.y for target_point in target_points]) # A unique list of latitude
coords_xarray = [ ("lat", lat_target_vector), ("lon", lon_target_vector)] # Create the coordinate dimensions ("time",time_vector)
# Randomly generate a 3-D dataset to pre-populate the dataset
#random_data = np.random.rand(len(time_vector), len(lat_target_vector), len(lon_target_vector))
#random_data[:] = np.nan
# 3.2.2 Generate the output template for saving the results
outputvars_list = ["flowering_pred","veraison_pred"]
output_template_score = xr.DataArray(coords=coords_xarray) # Create a dimensional template xarray object that is going to be used as the output structure
output_template_sm_ob = xr.DataArray(coords=coords_xarray + [("time", date_range_output)]) # Create a dimensional template xarray object object that is going to be used as the output structure
output_template_sm_fc = xr.DataArray(coords=coords_xarray + [("time", date_range_output), ("number", range(ensemble_members))]) # Create a dimensional template xarray object object that is going to be used as the output structure

# 3.2.3 Create a dictionary of output xarray object
forecast_score_dict = {} # Dictionary to store performance scores
for forecast_month in forecast_time:
    forecast_score_dict[str(forecast_month)+"_flo"] = output_template_score.copy(deep=True)
    forecast_score_dict[str(forecast_month)+"_ver"] = output_template_score.copy(deep=True)
forecast_sm_dict ={} # Dictionary to store simulation data
for forecast_month in forecast_time:
    forecast_sm_dict[str(forecast_month)+"ob_flo"] = output_template_sm_ob.copy(deep=True)
    forecast_sm_dict[str(forecast_month)+"ob_ver"] = output_template_sm_ob.copy(deep=True)
    forecast_sm_dict[str(forecast_month)+"sm_flo"] = output_template_sm_fc.copy(deep=True)
    forecast_sm_dict[str(forecast_month)+"sm_ver"] = output_template_sm_fc.copy(deep=True)
# output_dict = {var : output_template.copy(deep=True) for var in outputvars_list}
# output_dict_sm = {var : output_template_sm.copy(deep=True) for var in outputvars_list}
# output_flower = output_dict["flowering_pred"] # Access the template xarray object for flowering stage
# output_veraison = output_dict["veraison_pred"] # Access the template xarray object for veraison stage
# output_flower_sm = output_dict_sm["flowering_pred"]
# output_veraison_sm = output_dict_sm["veraison_pred"]    
    
# 3.2.5 Pre-load the required datasets into python variables
xarray_data_dict = {}
resolution = 1
# Read historical weather datasets and interpolate them into target resolutions
if determine_lat_lon_vector=="by_dataset":
    for grid_data in target_dataset:
        if any([var in grid_data for var in metero_var]):
            xarray_data = xr.open_dataset(grid_data, mask_and_scale=True, engine = "netcdf4") # Open and read the datasets into xarray objects
            # Subset the dataset to study region                                                  
            xarray_data_subset_region = xarray_data.where(((xarray_data.latitude >= min(lat_target_vector)-2) & (xarray_data.latitude <= max(lat_target_vector)+2)) &
                                                                     ((xarray_data.longitude >= min(lon_target_vector)-2) & (xarray_data.longitude <= max(lon_target_vector)+2)),
                                                                      drop=True)                           
            # Then subset the dataset to study time period
            xarray_data_subset_region_time = xarray_data_subset_region.where(xarray_data_subset_region.time.dt.year.isin([begin_year-1]+list(study_period)),  drop=True)                                     
            # Obtain the lat lon boundary
            latmin, latmax, lonmin, lonmax = int(min(xarray_data_subset_region_time["latitude"]).data),  int(max(xarray_data_subset_region_time["latitude"]).data), int(min(xarray_data_subset_region_time["longitude"]).data), int(max(xarray_data_subset_region_time["longitude"]).data)
            # Interpolate the datasets into 1.0 resolution
            xarray_data_interp = xarray_data_subset_region_time.interp(latitude= np.arange(latmin, latmax+resolution, resolution), longitude=np.arange(lonmin, lonmax+resolution, resolution), method="nearest")
            var_shortname = list(xarray_data_interp.data_vars.keys())[0] # Access the underlying data array variable name 
            xarray_data_dict[var_shortname] = xarray_data_interp.copy(deep=True)
            xarray_data.close()
        else:
            continue
else:
    for grid_data in target_dataset:
        if any([var in grid_data for var in metero_var]):
            xarray_data = xr.open_dataset(grid_data, mask_and_scale=True, engine = "netcdf4") # Open and read the datasets into xarray objects
            # Access the underlying variable name of the dataset                                                
            var_shortname = list(xarray_data.data_vars.keys())[0]
            xarray_data_dict[var_shortname] = xarray_data.copy(deep=True)
            xarray_data.close()
        else:
            continue
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 4. Run the simulations throughout target regions and compute the forecast score for each point
Bug_dict = {} # Create a bug dictionary to catch the bugs during simulations
# Iterate over each target grid point
for index, target_point in enumerate(target_points):
    timer = Timer()
    timer.start()
    #lon1 = round(target_point.x, decimal_place)
    #lat1 = round(target_point.y, decimal_place)
    lon1 = target_point.x
    lat1 = target_point.y
    point = {output_template_score.dims[0]:lat1, output_template_score.dims[1]:lon1} #Get the current point dict
    print("Start running for point No.{} out of total {}".format(str(index+1),str(len(target_points))))  
    # Extract the daily timeseries of observed weather datasets for the specified grid point
    site_data_df  = collect_weather_data(xarray_data_dict, lon1, lat1, study_period, dataset_loaded= True, 
                                         var_identifiers = metero_var, target_column_order = target_column_order, extract_method="nearest")
    # Extract the minimum temperature series
    T_min = site_data_df[metero_var_dict_OB["Tmin"]] # Ensure the input data is a series
    T_min.name ="tasmin" # Reset the Tmin series name into "tasmin"
    # Extract the mean temperature column 
    #T_mean = site_data_df[metero_var_dict["Tmean"]] # Ensure the input data is a series
    # Extract the maximum temperature series 
    T_max = site_data_df[metero_var_dict_OB["Tmax"]] # Ensure the input data is a series
    T_max.name ="tasmax" # Reset the Tmax series name into "tasmax"
    # Calculate the Tmean series
    T_mean = (T_min + T_max)/2
    T_mean.name ="tas" # Reset the Tmean series name into "tas"
    T_mean_ob = T_mean.loc[T_mean.index.year.isin(study_period)] # Subset the temperature data to cover the study period only
    try: # # Run the model to get the simulated flowering and veraison date based on the observed weather data
        flo_ob, ver_ob = run_models(T_mean_ob)
        point_time = point.copy()
        point_time["time"] = date_range_output # Update the point dimension with extra time dimension
        forecast_sm_dict[str(forecast_month)+"ob_flo"].loc[point_time] = flo_ob
        forecast_sm_dict[str(forecast_month)+"ob_ver"].loc[point_time] = ver_ob
    except:
        Bug_dict.update({"lon"+str(lon1)+"_lat"+str(lat1):'Issues in simulated values with observed weather!'}) # catch the erroneous simulation values
        continue
    # 3.3.2 Extract the seasonal forecast dataset
    for forecast_month in forecast_time_data.keys():
        forecast_data_var_dict = forecast_time_data[forecast_month]
        Tmin_list_path = forecast_data_var_dict["Tmin"]
        Tmax_list_path = forecast_data_var_dict["Tmax"]
        # Compute the P-correlation coefficient for all forecast ensemble members and adopts the mean value
        corre_fc_ensemble_flo = []
        corre_fc_ensemble_ver = []
        for ens_member in range(ensemble_members):
            point_time["number"] = ens_member # Update the point dimension with extra ensemble member dimension
            # Extract the forecast Tmin dataset (with gap filled) at a specific forecast date/time/month for a given point
            Tmin_fc = extract_fc_dataset(Tmin_list_path, metero_var_dict_forecast["Tmin"], T_min, lon1, lat1, ensemble_member=ens_member)
            Tmax_fc = extract_fc_dataset(Tmax_list_path, metero_var_dict_forecast["Tmax"], T_max, lon1, lat1, ensemble_member=ens_member)
            # Ensure the length of extracted series are the same and free or NaN values
            if any(np.logical_or(pd.isnull(Tmin_fc), pd.isnull(Tmax_fc))) or (len(Tmin_fc) != len(Tmax_fc)):
                Bug_dict.update({"lon"+str(lon1)+"_lat"+str(lat1):'Errors in extracted forecast data'}) # catch the erroneous instances
            # Compute the mean series 
            T_mean_fc = (Tmin_fc + Tmax_fc)/2
            # Subset the temperature data to cover the study period only
            T_mean_fc = T_mean_fc.loc[T_mean_fc.index.year.isin(study_period)]
            # Obtain the prediction for dormancy break and budburst dates
            # dormancy_out, budburst_out = run_BRIN_model(T_min, T_max, CCU_dormancy = 144.19, T0_dormancy = 213, CGDH_budburst = 823.9, 
            #                    TMBc_budburst= 25, TOBc_budburst = 0.42, Richarson_model="daily") # The parameter set is applied for TN calibrated from Luisa. L. et al. 2020 
            # if any(budburst_out.isnull()):
            #     budburst_out.fillna(method="ffill", inplace=True)
            # Run the sigmoid model that is calibrated for TN
            # Obtain the predicted phenology DOY over study period for a given point using a specific phenology module # Mean parameter settings for TN
            try:
                flo_fc, ver_fc = run_models(T_mean_fc) # Run the model to get the simulated flowering and veraison date based on the forecast weather data
                # Attach unmodified simulation series into target output files
                forecast_sm_dict[str(forecast_month)+"sm_flo"].loc[point_time] = flo_fc
                forecast_sm_dict[str(forecast_month)+"sm_ver"].loc[point_time] = ver_fc
                # Check all the simulated series if there are any NaN values
                target_dict = {} # Create an empty dictionary to store results
                for key, data_ser in {"flo_ob_fillNaN": flo_ob, "ver_ob_filllNaN": ver_ob, "flo_sm_fillNaN": flo_fc, "ver_sm_fillNaN": ver_fc}.items():
                    if any(pd.isnull(data_ser.index)):
                        data_ser.index = pd.Index(study_period) # The index column shuold already conform to the study period
                    data_ser_noNaN = check_NaN(data_ser) # Fill NaN values if any
                    target_dict[key] = data_ser_noNaN          
                # Compute the correlation coefficient
                flo_corr = np.corrcoef(np.array(target_dict["flo_ob_fillNaN"]), np.array(target_dict["flo_sm_fillNaN"]),  rowvar=False)[0][1]
                ver_corr = np.corrcoef(np.array(target_dict["ver_ob_filllNaN"]), np.array(target_dict["ver_sm_fillNaN"]), rowvar=False)[0][1]
                # Append the correlation coefficient into the result list
                corre_fc_ensemble_flo.append(flo_corr)
                corre_fc_ensemble_ver.append(ver_corr)
            # # Check if there are any NaN values in the series
            except:
                Bug_dict.update({"lon"+str(lon1)+"_lat"+str(lat1):'Issues in simulated values!'}) # catch the erroneous simulation values
                continue # Skip errorneous ensemble simulation
        # Compute the ensemble mean of correlation coefficient
        ens_corre_flo = np.nanmean(corre_fc_ensemble_flo)
        ens_corre_ver = np.nanmean(corre_fc_ensemble_ver)
        # Attach the computed ensemble mean of correlation coefficient into the dictionary
        forecast_score_dict[str(forecast_month)+"_flo"].loc[point] = ens_corre_flo
        forecast_score_dict[str(forecast_month)+"_ver"].loc[point] = ens_corre_ver
        # Attach the results to taget .nc file
        # Write the timeseries of DOY of a given point into the data array
        # for index, (DOY_flo, DOY_ver) in enumerate(zip(flo_ob, ver_ob)):
        #     #target_year = pd.to_datetime("{}-12-31".format(str(int(DOY_point.index[index]))), format="%Y-%m-%d")# Obtain the year information
        #     if flo_ob.index[index] == ver_ob.index[index]: # Test that the flowering and veraison DOY are reached in the same year
        #         target_year = flo_ob.index[index] # Obtain the target year by accessing the year of simulated flowering DO
        #     elif flo_ob.index[index] != ver_ob.index[index]: # In case the flowering and veraison stage are not reached in the same year
        #         Bug_dict.update({"lon"+str(lon1)+"_lat"+str(lat1):'NaN'}) # catch the erroneous simulation values
        #         continue
    print("Finish processing for point No.{} out of total {}".format(str(index+1),str(len(target_points))))  
    timer.end()
# 3.4 Write the output into the target 3-D dataset
output_path = join(output_path,"forecast_performance")
mkdir(output_path)
for key ,value in forecast_score_dict.items():
    output_da = value.to_dataset(name = key)
    # Save to disk as .nc file
    output_da.to_netcdf(join(output_path,"{}.nc".format(key)), mode='w', format="NETCDF4", engine="netcdf4")