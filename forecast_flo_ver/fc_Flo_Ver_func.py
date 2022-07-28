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
#%matplotlib inline
# Append the script path to system path
###############################################################################################################################################################################################################
########################################################################## Function Blocks ####################################################################################################################
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
###############################################################################################################################################################################################################
#  Specify all user-dependent path and other necessary constant input variables
# Check for target drive letter, which basically check the device working with.
if getpass.getuser() == 'Clim4Vitis':
    script_drive = "H:\\"
elif getpass.getuser() == 'Admin':
    script_drive = "G:\\"
elif (getpass.getuser() == 'CHENYAO YANG') or (getpass.getuser() == 'cheny'):
    script_drive = "D:\\"
main_script_path_ = r"Mega\Workspace\Study for grapevine" # Specific for a given study
target_dir = join(script_drive, main_script_path_, "Study7_Sigmoid_phenology_modelling_seasonal_forecast", "script_collections")
# Add the script path into system path
add_script(script_drive, target_dir) # Add the target folder containing necessary self-defined scripts into the system path
# Import all essential functions and classes used
from Multi_phenology_model_classes import * # Add the multi-phenology model class script from myself 
###############################################################################################################################################################################################################
class Timer():
    '''
    Define a timer class that calculate the duration between a block of codes.
    '''
    def start(self):
        print(f"[BENCHMARK] Start Time - {datetime.now()}")
        self._start_time = time.perf_counter()

    def _duration(self):
        duration = timedelta(seconds=time.perf_counter() - self._start_time)
        print(f"[BENCHMARK] Total Duration - {duration}")

    def end(self):
        self._duration()
        print(f"[BENCHMARK] End Time - {datetime.now()}")
    #### %%timeit
###############################################################################################################################################################################################################
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
###############################################################################################################################################################################################################          
def subset_list_from_list(list_of_list,filter_list):
    '''
    Select subset of a list-of-list with selection criteria provided by filter list.

    Parameters
    ----------
    list_of_list : a list of list, user-specified input
    filter_list: a list contains elements that are contained within original sublist elements
    '''
    List_of_List_Target=[] # This is the refined list of list from input
    for sub_list in list_of_list:
        sub_list_new=[]
        for sub_list_ele in sub_list:
            if any(ele in sub_list_ele for ele in filter_list): # Check each sublist, if any elements are desired
                sub_list_new.append(sub_list_ele)
        if len(sub_list_new)!=0:
            List_of_List_Target.append(sub_list_new)
    return List_of_List_Target
###############################################################################################################################################################################################################
def modify_cmap(original_cmap,insert_position=None,select_bins=None,extend_bins=None,remove_bins=False):
    '''
    Modify original cmap to return a new cmap

    Parameters
    ----------
    original_cmap : cmap instance, supplied cmap
    insert_position : int or sequence, indicate the position of cmap bin array to be inserted. Only used if remove_bins=False 
    select_bins : int or sequence, indicate the position to select color bins from extended cmap or delected if remove_bins is True
    extend_bins: int, indicate the extended number of bins. Only used if remove_bins=False
    remove_bins: bool, if to remove selected bins from supplied cmap. In this case, insert_position will be the position for deletion
    '''
    if remove_bins is False:
        # Obtain number of bins for the original cmap
        bins_number = original_cmap.N
        # Determine the target number of bins for the new cmap
        target_bin_number = bins_number+extend_bins
        # Create an extended cmap based on original cmap name and extended bins
        extend_cmap = discrete_cmap(target_bin_number, original_cmap.name)
        # Access the original cmap color bin arrays
        original_cmap_bins = original_cmap(np.linspace(0,1,bins_number))
        # Access the newly created cmap color bin arrays
        extend_cmap_bins = extend_cmap(np.linspace(0,1,target_bin_number))
        # Select target color bins to add into original color map bins
        compiled_bins = np.insert(original_cmap_bins, insert_position, extend_cmap_bins[select_bins,:], axis=0) # Here the axis is always 0 to insert for a given row
    else:
        # Access the original cmap color bin arrays
        original_cmap_bins = original_cmap(np.linspace(0,1,original_cmap.N))
        # Remove selected color bins to make a new cmap bins
        compiled_bins = np.delete(original_cmap_bins, select_bins, axis=0) 
        
    # Return the new colormap
    return ListedColormap(compiled_bins)
###############################################################################################################################################################################################################
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
###############################################################################################################################################################################################################
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
    # if site_data.ndim != 1:
    #     site_data=np.squeeze(site_data)
    # # Check units and convert unit to standard one
    # if var_dict.attrs["units"] in ["K","k","Kelvin","kelvin"]: 
    #     site_data=site_data-273.15 # Transform the kelvin to celcius
    # elif var_dict.attrs['standard_name']== 'precipitation_flux' or "s-1" in var_dict.attrs['units']:
    #     site_data=site_data*3600*24 # Transform the precipitation flux to precipitation amount/day
    # if ("rr" in nc_var_identifier) or ("pr" in nc_var_identifier):
    #     site_data=np.array(site_data)
    #     site_data[site_data<0.5]=0 # For any precipitation amount that is below 0.5 mm, set it to 0 mm.
    # #### To be futher added if additional attributes area found in the nc files that are needed to transform the data
    
    # Convert the output in one-D array into pd.Series
    site_data_ser = pd.Series(site_data,index=time_period,name=nc_var_identifier)
    
    return site_data_ser
###############################################################################################################################################################################################################
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
###############################################################################################################################################################################################################
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
###############################################################################################################################################################################################################
def find_nearest(array, value, index_search=True):
    '''
    Find the closest array element to value""
    Parameter
    ----------
    array : array/sequence/list, input array-like to search 
    value: the target value to match
    index_search: bool, if return index of the value or the value out of the array
    '''
   # Find out the index of the array that correspond to the closest array element
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if index_search:
        return idx
    else:
        return array[idx]
###############################################################################################################################################################################################################
def get_nearest_hundreds(x, round_decimals=10):
    # Firstly round the number to the nearest integer, and then split the number into tenth and ones digit.
    # Secondly, for the ones digit below 2.5 assign 0 
    '''
    Calculate the nearest hundred from the input data

    Parameters
    ----------
    x : int/float/iterable objects, input data to be used to infer the nearest hundred integer values. 
    One common application scenario is to calculate the boundary value for plot. 
    '''
    if isinstance(x, (int, float)):
        # if ((round(x)%10)<2.5) or ((round(x)%10)>7.5):
        #     target_hundred = int(round(x/10))*10
        # else:
        #     target_hundred = (round(x)//10)*10 +5
        
        if (round(x)%round_decimals) != 5:
            target_hundred = round(round(x/round_decimals))*round_decimals
        else:
            target_hundred = round(round(x/round_decimals))*round_decimals +round_decimals
            
        return target_hundred
    elif isinstance(x, Iterable): # If the x is iterable, compute the nereaset hundred for all elements inside x 
        target_hundred = [round(round(x/round_decimals))*round_decimals if (round(x)%round_decimals) != 5 else round(round(x/round_decimals))*round_decimals +5 for ele in list(x)]
        # target_hundred = [int(round(ele/10))*10 if (round(ele)%10<2.5) or (round(ele)%10>7.5) else (round(ele)//10)*10 +5 for ele in list(x)]
        #nearest_hundred = [int(np.round(ele/10))*10 if (round(ele)%10<2.5)or(round(ele)%10>7.5) else (round(ele)//10)*10 +5 for ele in list(x)]
        return target_hundred
###############################################################################################################################################################################################################
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
            # Obtain the underlying data array
            da = target_dataset_array[target_dataset_name].copy(deep=True)
            #lon_idx = find_nearest(da.longitude, lon1)
            # lat_idx = find_nearest(da.latitude,  lat1)
            # startTime_idx = find_nearest(da.time.data, timestamp_select.isel(time=0).data)
            # endTime_idx = find_nearest(da.time.data, timestamp_select.isel(time=len(timestamp_select.time)-1).data)
            # site_data = a[startTime_idx:(endTime_idx+1), lat_idx, lon_idx]
            site_data = da.sel({lon_name:lon, lat_name:lat, "time":timestamp_select.data}, method=method).data
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
    # Check for missing values and fill NaN if any
    merged_df_final = check_NaN(merged_df_final)

    return merged_df_final
###############################################################################################################################################################################################################
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
###############################################################################################################################################################################################################
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
###############################################################################################################################################################################################################
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
###############################################################################################################################################################################################################
def extract_fc_dataset(xr_concat, fc_var, ob_ser, lon1, lat1, ensemble_method=None, reduction_dim = None, **kwargs):
    '''
     Extract forecast datasets for a given point specified by lon and lat, while complement the gap period with observed data
      
     Mandatory Parameters
     ----------
     xr_concat : xarray dataset, a pre-concatenated xarray dataset object (concatenate along the time dimension) to extract the timeseries for lon1 and lat1 
     fc_var: str, the meteorological variable name of the forecast dataset
     ob_ser: series, a series of extracted observed weather for this point
     lon: float, longitude of the site
     lat: float, latitude of the site
     
     Optional Parameters
     ----------
     ensemble_method: str, indicate how to compute the required statistical metric for all the ensemble members of seasonal forecast dataset 
     reduction_dim: str, indicate the dimension (ensemble memeber) that will be reduced by a specific ensemble method.     
    '''      
    # assert isinstance(data_path, list), "a list of input data path is required, but the format {} is found".format(type(data_path))
    # # Read the list of data path into xarray object that varied with different time span
    # list_datasets = []
    # for var_path in data_path:
    #     xr_data = xr.open_dataset(var_path, mask_and_scale=True, engine = "netcdf4") # Load and open the xarray dataset
    #     list_datasets.append(xr_data)
    #     xr_data.close()
    # #list_datasets = [xr.open_dataset(var_path, mask_and_scale=True, engine = "netcdf4") for var_path in data_path]
    # # Concatenate all the xarray dataset objects into a signle object with the full time-series
    # xr_concat = xr.concat(list_datasets, dim = "time", data_vars ="all",
    #                          coords = "minimal", join="inner", compat = "equals", combine_attrs = "override")
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
    # Concatenate two series into one df with matching datetime index
    ob_fc_df = pd.concat([ob_ser, fc_ser], axis=1, join="outer", ignore_index=False) 
    # Fill the forecast data gap values from observed data
    fc_ser_filled = ob_fc_df[fc_var].fillna(ob_fc_df[ob_ser.name]) # Provide this column to fillna, it will use those values on matching indexes to fill:
    # Subset seasonal forecast data to observed weather in case the study period is not consistent 
    fc_ser_filled_subset = fc_ser_filled.loc[fc_ser_filled.index.year.isin(ob_ser.index.year.unique())]
    
    return fc_ser_filled_subset
###############################################################################################################################################################################################################
def run_models(T_input):
    """
    Run the model with prescribed temperature inputs and return simulated flowering and veraison DOY.
    By default, it now only runs for flowering and veraison stage using the sigmoid model
    
    Parameter    
    ----------
    T_input: series, series of temperature input to drive the model simulations
    Bug_dict: dict, a dictionary to store the simulated data with NaN values
    lon1: float, longitude of the location
    lat1: float, latitude of the location
    """
    # Starting from the first day of year
    DOY_point_flo = phenology_model_run(T_input, thermal_threshold=26.792, module="sigmoid", a=-0.15058, b=23.72417, 
                                            DOY_format=True, from_budburst= False, T0 =1) 
    # Starting from the simulated flowering DOY
    DOY_point_ver = phenology_model_run(T_input, thermal_threshold=75.419, module="sigmoid", a = -39.99993, b = 15.4698, 
                                            DOY_format=True, from_budburst= False, T0 = DOY_point_flo.copy(deep=True))

    return DOY_point_flo, DOY_point_ver
###############################################################################################################################################################################################################
def load_and_concat_da(data_path, concat_dim ="time"):
    """
    Load all .nc file into xarray dataset objects and concatenate them along the concat_dim
    
    Parameter    
    ----------
    data_path: iterable, an iterable of path or path-like object that points to the target .nc files 
    concat_dim: str, the dimension to concatenate
    """
    # xr.open_mfdataset() can be an effective alternative
    # Create an empty list to append loaded xarray data array
    list_datasets = []    
    for var_path in data_path: # Iterate over each data path
        xr_data = xr.open_dataset(var_path, mask_and_scale=True, engine = "netcdf4") # Load and open the xarray dataset
        list_datasets.append(xr_data) # Append to the lsit
        xr_data.close() # Close the data handler
    # Concatenate over the specified dimension: concat_dim.        
    xr_concat = xr.concat(list_datasets, dim = concat_dim, data_vars ="all",
                                 coords = "minimal", join="inner", compat = "equals", combine_attrs = "override")
    return xr_concat 
###############################################################################################################################################################################################################
def save_to_dataarray(data_arr, save_dim, output_data, point_dims):
    """
    Save the simulation data into the created xarray dataarray object. Currently, it only supports time-based dimension vector to save the data
    
    Parameter    
    ----------
    data_arr: xr.core.dataarray.DataArray, xarray data array where the target simulations are saved
    save_dim: str, the dimension of DataArray to load the simulation data
    output_data: pd.series, a timeseries of data in the form of panda series to be saved into the xarray data array object 
    point_dims: dict, the point dimension used as indexing object when saving the output_data
    """
    assert isinstance(data_arr, (xr.core.dataset.Dataset, xr.core.dataarray.DataArray)), "the input xarray data array is not a required data array"
    assert isinstance(output_data, pd.core.series.Series), "the data to be saved is not a panda series"
    # The saving dimension must not in the indexing dimension of xarray data array
    save_point = {key:value for key, value in point_dims.items() if key not in save_dim} 
    # Iterate over the dimensional vector and output data
    # The output data index should be the dimension looked up by the xarray
    for dim_ele, data_ele in zip(output_data.index, output_data): # Here the output data and save dimension should be consistent in length
        save_point_dim = save_point.copy()
        save_point_dim.update({save_dim:dim_ele}) # Update the dict with each saving dimmensional element 
        data_arr.loc[save_point_dim] = data_ele # Save a scaler value one at a time for the xarray object
    # Not return anything because the data array object is loaded into the dictionary
###############################################################################################################################################################################################################
def sort_FCdatasets_by_FCdate(fc_path, fc_times, metero_var_dict_forecast= {"Tmin":"mn2t24", "Tmax":"mx2t24", "Prec":"tp"}, 
                              fc_type="monthly", data_format=".nc"):
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
###############################################################################################################################################################################################################
def check_lonlat_ncdatasets(dataset_dict, test_second_layer=True, return_lonlat_list= True):
    """
    Check if lon and lat vectors are equal across different levels and types of datasets.
    
    Parameter    
    ----------
    dataset_dict: dict, input dict with key corredponing to the dataset variable name and value associated with the path to the file
    test_second_layer: bool, if True test across different types of datasets, otherwise only test different levels of the same dataset
    return_lonlat_list: bool, whether to return the collected lon lat vectors or only carry out the testing
    """
    # Another way of checking equality 
    # for coordinate_vector in [list_of_lon, list_of_lat]:
    #     check_bool = np.diff(np.vstack(coordinate_vector).reshape(len(coordinate_vector),-1), axis=0)==0
    #     if not check_bool.all():
    #         print("Coordinate vectors are not equal across datasets")
    #     else:
    #         print("Equal coordinate vectors are found across datasets")
    
    assert isinstance(dataset_dict, dict), "input does not follow a dict format, the {} type is found".format(type(dataset_dict))
    # Testing in the first layer of equality, i.e. testing equality of latlon under internally organized datasets (value)
    collection_dict = {} # Define an output collection dict to collect target points
    for key, value in dataset_dict.items():
        if not isinstance(value, Iterable):
            da = xr.open_dataset(value, mask_and_scale=True, engine = "netcdf4") # Open and load the dataset
            lon_name, lat_name = get_latlon_names(da) # Extract the longitude and latitude name
            lon_vector, lat_vector = da[lon_name].data, da[lat_name].data # Get the longitude and latitude vector from the supplied .nc file datasets
            coordinates = list(product(lon_vector, lat_vector)) # Form the list of coordinates to be studied
            target_points = [] # Create an empty list to collect target points
            for coordinate in coordinates: # Iterate over each coordinate to get the target point
                grid_point = Point(coordinate) 
                target_points.append(grid_point) 
            collection_dict[key] = target_points # Attach to target dict
            da.close() # Close the current dataset handler
        elif isinstance(value, Iterable):
            geo_ser_list = []
            for ele in value:
                #utput_key = "itemNo.{}".format(index+1) # Define the output key
                da = xr.open_dataset(ele, mask_and_scale=True, engine = "netcdf4") # Open and load the dataset
                lon_name, lat_name = get_latlon_names(da) # Extract the longitude and latitude name
                lon_vector, lat_vector = da[lon_name].data, da[lat_name].data # Get the longitude and latitude vector from the supplied .nc file datasets
                coordinates = list(product(lon_vector, lat_vector)) # Form the list of coordinates to be studied
                target_points = [] # Create an empty list to append target grid points
                for coordinate in coordinates:
                    grid_point = Point(coordinate)
                    target_points.append(grid_point) 
                # Append the converted geoseries results to the empty list
                geo_ser_list.append(gpd.GeoSeries(target_points))
                da.close() # Close the current dataset handler
            # Test if all geo-series are equal over sucessive paired geo-series coordinates are equal or not 
            for geo_ser_first, geo_ser_second in zip(geo_ser_list[:], geo_ser_list[1:]):
                if not all(geo_ser_first.geom_equals(geo_ser_second)):
                    raise ValueError("Grid points are not equal across input datasets")
            collection_dict[key] = list(geo_ser_list[0])  # Attach to target dict
    # Testing in the second layer, i.e. testing equality of latlon under value
    if test_second_layer: # This must be true, if dataset_dict value is not iterable
        geo_ser_list_second = [ gpd.GeoSeries(list_points) for list_points in list(collection_dict.values()) ] 
        for geo_ser_first, geo_ser_second in zip(geo_ser_list_second[:], geo_ser_list_second[1:]):
            if not all(geo_ser_first.geom_equals(geo_ser_second)):
                raise ValueError("Grid points are not equal across different variables of datasets")
    # Return the collected grid points or not
    if return_lonlat_list:
        return list(collection_dict.values())[0] # Return the collected grid points from the first collection element (equally apply for any element)
###############################################################################################################################################################################################################
def check_NaN(ser, fill_method1="quadratic", fill_method2="bfill",fill_method3="ffill", fill_extra=True):
    """
    Check if there are any NaN values in the simulated series, if yes, fill NaN
    
    Parameter    
    ----------
    ser: pd.Series, the input panda series to fill NaN
    fill_method1: str, the specified interpolatio method to fill the NaN. As recommended by the developers, the quadratic function is the appropriate choice for time-series based analysis
    fill_method2: str, the backward filling method used to fill NaN.
    fill_method3: str, the forward filling method used to fill NaN.
    fill_extra: bool, if an extra filling NaN operation should be implemented.
    """
    if any(pd.isnull(ser)):
        ser.interpolate(method=fill_method1, limit_direction="both", inplace=True)
        if fill_extra: # In case the interpolation fill does not completely fill the gap, use built-in forward and backward fill to fill the series 
            if any(pd.isnull(ser)):
                ser.fillna(method=fill_method2, inplace=True)
            if any(pd.isnull(ser)):
                ser.fillna(method=fill_method3, inplace=True)
            return ser # This double fillna will ensure no NaN values go unfilled
        else:
            return ser # Series that is only filled once with the specified interpolation method
    else:
        #print("No NaN values are deteced, thereby no filling NaN is performed")
        return ser
###############################################################################################################################################################################################################
def fill_na(ser, NA_val=None):
    """
    Replace and fill specified values as NaN
    
    Parameter    
    ----------
    ser: pd.Series, the input panda series to replace values (e.g. -999) into NaN
    NA_val: float/int, the specified values to be filled as NaN
    """
    assert isinstance(ser, pd.core.series.Series), "the input series does not follow a panda series format, the {} is found".format(type(ser))
    # Fill the serie with specified values as NaN
    ser.loc[ser==NA_val] = np.nan
    return ser
###############################################################################################################################################################################################################
def great_circle(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between two points using their geographic coordinates 
    """
    from math import radians, sin, cos, acos
    # Modify the supplied input longitude and latitude
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])    
    distance = 6371 * (acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2)))
    # Perform the The Great-Circle Distance calculation
    return distance
###############################################################################################################################################################################################################
def GDD_simple(x, GDD_threshold):
    """
    Apply the simple GDD calculation method on individual value
    """
    if x <= GDD_threshold:
        dd_day = 0
    else:
        dd_day = x - GDD_threshold
    return dd_day
###############################################################################################################################################################################################################
def categorize_variable(x, ser):
    """
    Transform the continuous variables into a category variable by computing its percentile categoty overy a timeseries. Intended to be used in series.apply() function
    """
    if x <= ser.quantile(q=0.1):
        x = 1 # The binary value represents the anomalously early/less
    elif (ser.quantile(q=0.1)< x) &  ( x <= ser.quantile(q=0.25)):
        x = 2  # The binary value represents the anomalously late/more
    elif (ser.quantile(q=0.25)< x) & ( x <= ser.quantile(q=0.5)):
        x = 3 # The binary value represents the normal 
    elif (ser.quantile(q=0.5)< x) &  ( x <= ser.quantile(q=0.75)): 
        x =4
    elif (ser.quantile(q=0.75)< x) & ( x <= ser.quantile(q=0.9)): 
        x =5
    elif ser.quantile(q=0.9) < x:
        x = 6
    return x
###############################################################################################################################################################################################################
def tercile_var(x, ser, ref_categorize=False, **kwargs):
    """
    Transform the continuous variables into a category variable by computing its tercile group overy a timeseries. Intended to be used in series.apply() function
    x: the element of ser, intends to be used in ser.apply() function
    ser: series, the input series that contain actual values of a variable, e.g. DOY for phenology
    ref_categorize: bool, if use reference data to convert the continuous variable into a categorical variable
    kwargs: any additionaly keyword arguments
    """
    if not ref_categorize:
        # Categorize the data based on the input data itself
        Q1 = ser.quantile(q=0.33) 
        Q2= ser.quantile(q=0.67)
    else:
        assert ("Q1" in kwargs.keys()) and ("Q2" in kwargs.keys()), "Need to manually supply Q1 and Q2 parameters"
        Q1, Q2  = kwargs["Q1"], kwargs["Q2"] # Access the underlying Q1 and Q2 parameter values
    
    if x <= Q1:
        x = 1 # Value lower than the Quantile 33%, being assigned a value of 1
    elif (Q1< x) &  ( x <= Q2):
        x = 2  # Value lower than the Quantile 67%, but higher than 33%, being assigned a value of2
    elif Q2 < x:
        x = 3 # Value higher than the Quantile 67%, assign the value of 3
    else:
        x = np.nan # Assign nan values in case none of above situation matches
    return x
###############################################################################################################################################################################################################
def categorize_array(data_arr, threshold1=0.33, threshold2=0.67, dim_name="time", 
                     tercile_1=1, tercile_2=2, tercile_3=3, reference_tercile=False, **kwargs):
    """
    Categorize the data array into the tercile group, where data array values are converted into tercile group values 
    
    Parameters
    ----------
    data_arr: xarray DataArray, the input data array with continuous variables (here it denotes phenology)
    threshold1: float, the quantile threshold that denotes the tercile group 1
    threshold2: float, the quantile threshold that denotes the tercile group 2
    dim_name: str, the dimension name along which the quantile values are computed
    tercile_1: int, the integer value assigned to the tercile group1  
    tercile_2: int, the integer value assigned to the tercile group2  
    tercile_3: int, the integer value assigned to the tercile group3  
    reference_tercile: bool, if the categorization should be based on reference/forecast
    kwargs: any additional keyword arguments
    """
    assert isinstance(data_arr, (xr.core.dataarray.DataArray, xr.core.dataset.Dataset)), "the input array is not of the required data types, as the type {} is found".format(type(data_arr))
    # Set any values of -999 to np.nan
    data_arr_func =  data_arr.copy(deep=True)
    data_arr_func = data_arr_func.where(~(data_arr_func==-999), np.nan, drop=False) # Set any values of -999 to np.nan
    if not reference_tercile: # tercile categorization should be based on simulation time series itself
        # Compute the tercile group threhold values
        tercile1_thresh = data_arr_func.quantile(threshold1, dim = dim_name, keep_attrs = True, skipna = True)
        tercile2_thresh = data_arr_func.quantile(threshold2, dim = dim_name, keep_attrs = True, skipna = True)
    else: # tercile categorization should be based on the reference time series 
        assert "ref_arr" in kwargs.keys(), "reference array is not provided"
        ref_data_arr = kwargs["ref_arr" ].copy(deep=True) # Access the reference data array
        tercile1_thresh = ref_data_arr.quantile(threshold1, dim = dim_name, keep_attrs = True, skipna = True)
        tercile2_thresh = ref_data_arr.quantile(threshold2, dim = dim_name, keep_attrs = True, skipna = True)
    # Compute the tercile group values
    #var_OB_nc_xr_arr_sel = var_OB_nc_xr_arr.where(var_OB_nc_xr_arr.notnull(),drop=True)
    # Categorize the data array over the time period, where the tercile group1 is assigned value of 1, tercile 2 being assigned 2, tercile assigned the value of 3
    tercile_filter1 = data_arr_func.where((data_arr_func>tercile1_thresh) |  (data_arr_func.isnull()), tercile_1, drop=False) # Keep the data array values that is larger than tercile1_thresh, and values that is not null
    tercile_filter2 = tercile_filter1.where( (tercile_filter1<= tercile2_thresh) |  (tercile_filter1.isnull()), tercile_3, drop=False)
    tercile_final = tercile_filter2.where(tercile_filter2.isin([tercile_1,tercile_3]) | (tercile_filter2.isnull()) , tercile_2,  drop=False)
    
    return tercile_final
# def tercile_var_xr(xarray, dim_name="time"):
#     """
#     Transform the continuous variables into a category variable by computing its tercile group overy a timeseries of xarray object. 
#     Intended to be used in xr.apply_ufunc() 
#     """
#     if (xarray <= xarray.quantile(q=0.33, dim=dim_name, keep_attrs = True, skipna = True)) & (xarray.notnull()):
#         xarray = 1 # Value lower than the Quantile 33%, being assigned a value of 1
#     elif (xarray.quantile(q=0.33, dim=dim_name, keep_attrs = True, skipna = True)< xarray) &  ( xarray <= xarray.quantile(q=0.67, dim=dim_name, keep_attrs = True, skipna = True)) & (xarray.notnull()):
#         xarray = 2  # Value lower than the Quantile 67%, but higher than 33%, being assigned a value of2
#     elif (xarray > xarray.quantile(q=0.67, dim=dim_name, keep_attrs = True, skipna = True)) & (xarray.notnull()):
#         xarray = 3 # Value higher than the Quantile 67%, assign the value of 3
#     else:
#         xarray = np.nan # Assign nan values in case none of above situation matches
#     return xarray
###############################################################################################################################################################################################################
def subset_two_year_climate(T_series, twoyear_list, initial_date, lead_time):
    """
    Subset the consecutively 2-year temperature series to a range specified by the starting and the ending date 
    
    Parameters
    ----------
    T_series: panda core series, the full temperature series covering the study period (intend to be subset), can be either minimum, maximum and mean temperature.
    twoyear_list: list, list of two consecutive years
    initial_date: Timestamp, the starting date of calculation/forecast
    lead_time: int, number of months from the initial_date
    """
    assert isinstance(T_series, pd.core.series.Series), "temperature series does not follow panda series format, the {} is found".format(type(T_series))
    assert isinstance(twoyear_list, list), "the twoyear_list is not a list format, the {} is found".format(type(twoyear_list))
    assert isinstance(initial_date, (pd._libs.tslibs.timestamps.Timestamp)), "the initial_date is not a Timestamp format, the {} is found".format(type(twoyear_list))

    # Define the target period spanning lean_time in months.
    target_period = pd.period_range(initial_date,initial_date+DateOffset(months=lead_time-1),freq="M") # The DateOffset needs to minus one to get correct number of offset months
    # Subset the temperature series into seasonal temperature spanning each 2-year combo
    seasonal_climate = T_series[T_series.index.year.isin(twoyear_list)]
    # Subset the seasonal climate that starts from the initialization date spanning each 2-year combo
    seasonal_climate= seasonal_climate[~np.logical_and(seasonal_climate.index.dayofyear<initial_date.dayofyear,
                                                      seasonal_climate.index.year == min(twoyear_list))]
    # Subset the seasonal climate that starts from the initialization date and up to the target period length spanning each 2-year combo
    seasonal_climate = seasonal_climate[np.logical_and(seasonal_climate.index.month.isin(target_period.month),
                            seasonal_climate.index < datetime(year=max(twoyear_list), month=target_period.month[-1]+1, day=1))]
    return seasonal_climate
###############################################################################################################################################################################################################
def simulation_maps(file_name, data_arrays, shape_path, savepath, 
                    colormap, cbar_label, plot_dim= "space", origin=ccrs.CRS("EPSG:4326"), proj=ccrs.PlateCarree(), # ccrs.Geodetic()
                    subplot_row=10, subplot_col=3, fig_size=(6,21), specify_bound= False, extend="both", add_scalebar=False,
                    fig_format=".png", subplot_title_font_size=5, dpi=600, **kwargs):    
    """
    Make the spatial map plot for a specific variable from the input Xarray dataarray with prescribed statistics calculated
    # StackOverflow_example_1: https://xarray.pydata.org/en/stable/examples/visualization_gallery.html?highlight=colorbar&fbclid=IwAR3kSbcf3MMPBiEkdGRXNATwKwjFepukxDqY7UYgD4XHHAlaip7BiRADM3E#Control-the-plot%E2%80%99s-colorbar

    Mandatory Parameters
    ----------    
    file_name: str, file name to be used to save the plot.
    data_arrays: input Xarray DataArray, it can be a multi-dimensional array with suplot corresponding to one size of the dimension.
    shape_path: str, shape path for the target shapefile
    colormap : matplotlib cmap, the input cmap to be used. 
    cbar_label : str, indicate the cmap label to be used.
    savepath : str, indicate the save path.
    
    Optional parameters
    ----------
    plot_dim: str, indicate the target dimension for the plot. Defaults plot along the time dimension.
    origin: str/ccrs instance, indicate the crs for the original data.
    proj: str, indicate the crs for projection.
    subplot_row: int, number of rows prescribed in the figure.
    subplot_col: int, number of columns prescribed in the figure.
    fig_size: tuple, the figure size to create the figure class instance.
    fig_format: str, the target file format.
    kwargs: any additional keyword arguments
    """
    # Colorbrewer: https://colorbrewer2.org/#type=sequential&scheme=YlGn&n=5
    # Normalization techniques: https://matplotlib.org/stable/tutorials/colors/colormapnorms.html
    # Read the target shapefile into geodataframe
    if isinstance(shape_path, gpd.geodataframe.GeoDataFrame):
        GDF_shape= shape_path.to_crs(proj)
    else:
        GDF_shape= gpd.read_file(shape_path).to_crs(proj)
    # Assert that the input data array is a xarray object
    assert isinstance(data_arrays, (xr.core.dataarray.DataArray, xr.core.dataset.Dataset, list)), "the input array is not of the required data types"
    if "BS" not in file_name:
        # Create a grid system to apply for subplots
        grid = gridspec.GridSpec(nrows = subplot_row, ncols = subplot_col, hspace = 0.05, wspace = 0.05)
        fig = plt.figure(figsize=fig_size) # Create a figure instance class
        axis_list=[] # append to this empty list a number of grid specifications
        if (subplot_row>1) and (subplot_col>1):
            for row in range(subplot_row):
                for col in range(subplot_col):
                    axis_list.append(grid[row,col])
        elif subplot_row==1:
            for col in range(subplot_col):
                axis_list.append(grid[0,col])
        elif subplot_col==1:
            for row in range(subplot_row):
                axis_list.append(grid[row,0])
    else:
        fig, axe = plt.subplots(subplot_row,  subplot_col, figsize=fig_size, gridspec_kw={"hspace": 0.05, "wspace":0.05}, sharey='row',subplot_kw={'projection': proj})
        axis_list=[] # Append to this empty list a number of axe subplot
        for axe_sub in axe.flat:
            axis_list.append(axe_sub)
    # Obtain the lat and lon names in the Xarray Datasets with the underlying lat,lon vectors
    if isinstance(data_arrays,  list): # If the supplied data array is a list, each data array dimensional names should share the identical names
        lon_names = []
        lat_names = []
        for data_array in data_arrays:
            lon_name, lat_name= get_latlon_names(data_array)
            lon_names.append(lon_name)
            lat_names.append(lat_name)
        lon_name = str(np.unique(lon_names)[0])
        lat_name = str(np.unique(lat_names)[0])
    else:
        lon_name, lat_name= get_latlon_names(data_arrays)
    # Determine the vmin and vmax for the plot
    if not specify_bound:
        cbar_vmin, cbar_vmax = determine_vmin_vmax(kwargs["list_array"])
        bounds = np.linspace(cbar_vmin, cbar_vmax, colormap.N)
    else:
        bounds = kwargs["bounds"]
    # Determine the type of plot
    if plot_dim in ["Space", "space", lon_name, lat_name]: # If the dimension to plot is over the "space" dimension, it is to collect a list of yearly array
        list_array = [data_arrays.sel({"time":time_step}) for time_step in data_arrays.time.data ]  # lon= data_arrays.coords[lon_name],lat=data_arrays.coords[lat_name])]        
    # If the dimension to plot is over the "time" dimension, averaged over the space is performed
    elif plot_dim in ["Time", "tim", "TIME", "time"]:        
        fig = plt.figure(1, (15, 6), dpi=300) # The figure size is empirically defined according to the requirement for the timeseries plot
        plot_axe = fig.add_subplot()
        if kwargs["region_agg"] in ["mean", "Mean", "MEAN", "average"]: # This "region_agg" additional keyword argument must be supplied 
            timeseries_plot = data_arrays.mean(dim=[lon_name, lat_name],skipna=True).plot.line(ax=plot_axe, xticks = data_arrays.time.data,
                                                                             yticks = bounds, c = "black", lw = 1, marker = "o", mfc= "white", mec ="black",  ms=5)
        elif kwargs["region_agg"] in ["median", "Median", "MEDIAN"]: # Different regional aggregation methods
            timeseries_plot = data_arrays.median(dim=[lon_name, lat_name],skipna=True).plot.line(ax=plot_axe, xticks = data_arrays.time.data,
                                                                             yticks = bounds, c = "black", lw = 1, marker = "o", mfc= "white", mec ="black", ms=5)
        # Make some decorations on the time-series plot
        plot_axe.set_xticklabels(data_arrays.time.data, rotation="vertical") # Re-adjust the x-tick labels
        # Set the title of the plot 
        plot_axe.set_title(kwargs["plot_title"], fontdict={"fontsize":subplot_title_font_size,"fontweight":'bold'},
                                   loc="center",x=0.5,y=0.95,pad=0.05)
        # Set the y-axis label
        plot_axe.set_ylabel(kwargs["y_axis_label"], fontdict={"fontsize":7,"fontweight":'bold'}, labelpad = 0.05, loc = "center")
        # Save the plot to a local disk
        mkdir(savepath)
        fig.savefig(join(savepath,file_name+fig_format))
        plt.close(fig)
    elif plot_dim in ["corre", "CORRE", "correlation", "CORRELATION", "Correlation", "r", "scores","MAE","RMSE", "aggregate"]:
        list_array = data_arrays
    # Construct the bounds for the plot
    # if extend =="both":
    #     n_colors = colormap.N +2
    # elif extend in ["min", "max"]:
    #     n_colors = colormap.N +1
    # elif extend =="neither":
    #     n_colors = colormap.N
    # Generate a colormap index based on discrete intervals 
    # Remember that the first and the last bin of supplied cmap will be used for ploting the colorbar extension
    norm_bound =  mpl.colors.BoundaryNorm(boundaries=bounds, ncolors = colormap.N, clip = False, extend=extend)
    # Plot simulation results under each model into the subplot
    for index, (ax_grid, data_array) in enumerate(zip(axis_list, list_array)):
        if "BS" not in file_name:
            # Create a subplot axe in the figure class instance
            subplot_axe=fig.add_subplot(ax_grid,projection=proj)
        else:
            subplot_axe = ax_grid # A pre-created geo-axe
        # Obtain the longitude and latitude vectors
        lonvec = data_array[lon_name].data
        latvec = data_array[lat_name].data
        # Get the porjection latitude and longitude vectors
        dlon = np.mean(lonvec[1:] - lonvec[:-1])
        dlat = np.mean(latvec[1:] - latvec[:-1])
        x, y = np.meshgrid( np.hstack((lonvec[0] - dlon/2., lonvec + dlon/2.)),
                        np.hstack((latvec[0] - dlat/2., latvec + dlat/2.)) )
        # Make the plot
        subplot_axe.pcolormesh(x, y, data_array,
            norm=norm_bound, cmap=colormap #,shading= "nearest" 
            )
        # data_plot = data_array.plot(x=lon_name, y=lat_name,ax=subplot_axe, 
        #     norm=norm_bound, cmap=colormap,extend = extend,
        #     add_colorbar=False, add_labels=False
        #     ) 
        # Set the extent for each subplot map
        extent=[np.min(lonvec)-0.5,np.max(lonvec)+0.75, np.min(latvec)-0.5,np.max(latvec)+0.5] # Minor adjustment with 0.5 degree in each direction
        subplot_axe.set_extent(extent, crs=origin)
        # Add the geometry of shape file for each subplot map
        if ("col_name" in kwargs.keys()) & ("list_geometry" in kwargs.keys()):
            geo_df_col_name = kwargs["col_name"] # Access the underlying column name of supplied geodataframe where it stores information about target geometries that need to be modified
            geo_df_col = GDF_shape[geo_df_col_name] # Access the target geo-dataframe column
            geo_list = kwargs["list_geometry"] # Access the underlying list of geometry ID (ID is the integer ID) that needs to be modified
            # Iterate over each 
            for geo_item in geo_df_col:
                # Access the geo-dataframe data row-by-row
                geo_item_data = GDF_shape.loc[GDF_shape[geo_df_col_name] == geo_item, :]
                if geo_item in geo_list:
                    shp_linewidth = 0.3
                else:
                    shp_linewidth = 0.7
                # Add the target geometry with specified linewidth
                subplot_axe.add_geometries(geo_item_data.geometry, proj,
                            facecolor='none', edgecolor='black',  linewidth = shp_linewidth)
        else:
            subplot_axe.add_geometries(GDF_shape.geometry, proj,
                   facecolor='none', edgecolor='black',  linewidth=0.7)
        # Add the geometry of shape file as the outline for each subplot map
        if "outline" in kwargs.keys():
           GDF_outline = kwargs["outline"] # A geodataframe that is already being read
           subplot_axe.add_geometries(GDF_outline.geometry, proj,
            facecolor='none', edgecolor='black', linewidth=0.7)
        # Set the the yearly data as the subplot title
        if plot_dim in ["Space", "space", lon_name, lat_name]:
            subplot_name = str(int(data_array.time.dt.year)) if isinstance(data_array.time.data, (np.datetime64,datetime)) or ("_pred" in file_name) else str(int(data_array.time.data))
        elif plot_dim in ["corre", "CORRE", "correlation", "CORRELATION", "Correlation", "r", "scores","MAE","RMSE", "aggregate"]:
            if "forecast_month_var" in kwargs.keys(): # "missing 'forecast_month_var' in the funcation call"
                subplot_name = kwargs["forecast_month_var"][data_array.name]
            elif "temporal_agg_var" in kwargs.keys():
                subplot_name = kwargs["temporal_agg_var"][index]
        # Set the subplot title
        subplot_axe.set_title(subplot_name, fontdict={"fontsize":subplot_title_font_size, "fontweight":'bold'},
                                   loc="right", x=0.95, y=0.95, pad=0.05)
        # Add the grid line locators 
        if (len(kwargs["grid_lon"]) != 0) or (len(kwargs["grid_lat"]) != 0):
            assert isinstance(kwargs["grid_lon"] ,list) and isinstance(kwargs["grid_lat"] ,list), "input longitude and latitude gridlines are not in the form of list"
            if ("BS" in file_name) or (plot_dim == "aggregate"):
                if index==0:
                    add_gridlines(subplot_axe, kwargs["grid_lon"],  kwargs["grid_lat"], fontsize=subplot_title_font_size -1, top_labels=True, left_labels=True)
                else:
                    add_gridlines(subplot_axe, kwargs["grid_lon"],  kwargs["grid_lat"], fontsize=subplot_title_font_size -1, top_labels=True, left_labels=False)
            elif "_pred" in file_name:
                if (index % subplot_row)==0:
                    add_gridlines(subplot_axe, kwargs["grid_lon"],  kwargs["grid_lat"], fontsize=subplot_title_font_size -1, top_labels=True, left_labels=True)
                else:
                    add_gridlines(subplot_axe, kwargs["grid_lon"],  kwargs["grid_lat"], fontsize=subplot_title_font_size -1, top_labels=True, left_labels=False)
            
            else:
                add_gridlines(subplot_axe, kwargs["grid_lon"],  kwargs["grid_lat"], fontsize=subplot_title_font_size)
        #gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
        # Set the scale bar # Scalebar_package from mathplotlib https://pypi.org/project/matplotlib-scalebar/
        if add_scalebar is True:
            add_scale_bar(subplot_axe, lonvec, latvec)
            #scale_bar = AnchoredSizeBar(subplot_axe.transData, 10000, '10 km', 'upper right',frameon=False,size_vertical = 100) # ccrs.PlateCarree() unit is meter
        #scale_bar = ScaleBar(13.875, units="km", dimension= "si-length", length_fraction=0.25,location="upper right",
        #                     pad=0.2,label_loc="bottom", width_fraction= 0.05
                             
                             #) #width_fraction= 0.01, label= '50 km',
                             #location= 'upper right',pad=0.2,) # The spatial resolution for each grid point is at 0.125° x 0.125° (roughly 13.875 Km x 13.875 Km)
        #if "_BS"  in file_name:  # suppress auto sizing
        subplot_axe.set_autoscale_on(False) # Set the auto scaling effect for the subplot
        if kwargs["label_features"] is not None:
            if kwargs["label_features"] is True:
                if "label_features_font_size" in kwargs.keys():
                    add_gdf_labels(subplot_axe, GDF_shape, fontsize= kwargs["label_features_font_size"])
                else:
                    add_gdf_labels(subplot_axe, GDF_shape)
        # Append the subplot axe to an empty list
        #axe_list.append(subplot_axe)
    # Set one colorbar for the whole subplots
    if "BS" not in file_name:
        fig.subplots_adjust(right=0.85,wspace=0.05,hspace=0.05) # Make room for making a colorbar
        # Add the colorbar axe
        cbar_ax = fig.add_axes([0.9, 0.45, 0.02, 0.25])
        # Set the colorbar label size
        cbar_label_size = 12
        # Add the colorbar to the figure class instance
        cb  = mpl.colorbar.ColorbarBase(cbar_ax, cmap = colormap,
                                        norm = norm_bound,
                                        extend = extend,
                                        orientation = "vertical")
        # Set the colorbar label
        cb.ax.set_ylabel(cbar_label, rotation=270,fontdict={"size":cbar_label_size})
        # Set the padding between colorbar label and colorbar    
        cb.ax.get_yaxis().labelpad = 15
    else: # For BS plot, a special design on cmap bar label will be applied
        # Make room for making a colorbar
        fig.subplots_adjust(right=0.85, wspace=0.05, hspace=0.05) 
        # Add the colorbar axe
        cbar_ax = fig.add_axes([0.88, 0.25, 0.01, 0.4])
        # Set the colorbar label size
        cbar_label_size = 4
        # Add the colorbar to the figure class instance
        cb  = mpl.colorbar.ColorbarBase(cbar_ax, cmap = colormap,
                                        norm = norm_bound,
                                        extend = extend,
                                        orientation = "vertical")
        # Set the colorbar label
        cb.ax.set_ylabel(cbar_label, rotation=270,fontdict={"size":cbar_label_size})
        # Set the colorbar axis tick parameters
        cb.ax.tick_params(labelsize=cbar_label_size)
        # Set the padding between colorbar label and colorbar    
        cb.ax.get_yaxis().labelpad = 5
        # Additionally, set the subplot invisible in case of extra subplots
        if len(list_array) !=  len(axe.flat):
            for i, axe_sub in enumerate(axe.flat):
                if i+1 >len(list_array):
                    axe_sub.axis('off') # Turn off the axis where no data is there
    # Save the plot to a local disk
    mkdir(savepath)
    fig.savefig(join(savepath,file_name+fig_format), bbox_inches="tight",pad_inches=0.05, dpi=dpi)
    plt.close(fig)  
###############################################################################################################################################################################################################
def arrange_layout(num, cols = np.arange(2,7,1) , first_match=True):
    """
    Arrange map layout for the simulated results.
    Mandatory Parameters
    ---------- 
    num: int, total number of subplots to be used
    cols: array-like, test different possible columns to arrange the subplots (Consider appropriate if num%col ==0)
    first_match: bool, if return the first matching element from the computed list of cols.
    """
    target_cols = []
    for col in cols:
        remainder = num % col
        if remainder ==0:
            target_cols.append(col)
        else:
            continue
    if len(target_cols) !=0:
        if first_match:
            return target_cols[0]
        else:
            return target_cols # A full list of matching columns is returned
    else:
        print("Not get desired number of columns layout for the map plot, consider different 'cols' input")
###############################################################################################################################################################################################################
def plot_color_gradients(cmap_category, cmap_list, gradient):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows-1)*0.1)*0.22
    fig, axs = plt.subplots(nrows=nrows, figsize=(6.4, figh))
    fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2, right=0.99)
    axs[0].set_title(cmap_category + ' colormaps', fontsize=14)
    for ax, cmap_name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=cmap_name)
        ax.text(-.01, .5, cmap_name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)
    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()
###############################################################################################################################################################################################################
class show_cmap_list:
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    def __init__(self):
        self.cmaps = [('Perceptually Uniform Sequential', [
                    'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
                 ('Sequential', [
                    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
                 ('Sequential (2)', [
                    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                    'hot', 'afmhot', 'gist_heat', 'copper']),
                 ('Diverging', [
                    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
                 ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
                 ('Qualitative', [
                    'Pastel1', 'Pastel2', 'Paired', 'Accent',
                    'Dark2', 'Set1', 'Set2', 'Set3',
                    'tab10', 'tab20', 'tab20b', 'tab20c']),
                 ('Miscellaneous', [
                    'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
                    'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
                    'gist_ncar'])]
            
    def __call__(self):
        for cmap_category, cmap_list in self.cmaps:
            nrows = len(cmap_list)
            figh = 0.35 + 0.15 + (nrows + (nrows-1)*0.1)*0.22
            fig, axs = plt.subplots(nrows=nrows, figsize=(6.4, figh))
            fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2, right=0.99)
    
            axs[0].set_title(cmap_category + ' colormaps', fontsize=14)
    
            for ax, cmap_name in zip(axs, cmap_list):
                ax.imshow(self.gradient, aspect='auto', cmap=cmap_name)
                ax.text(-.01, .5, cmap_name, va='center', ha='right', fontsize=10,
                        transform=ax.transAxes)
    
            # Turn off *all* ticks & spines, not just the ones with colormaps.
            for ax in axs:
                ax.set_axis_off()
        plt.show()
###############################################################################################################################################################################################################
def extract_nearest_neighbours(data_arr, data_ser, lon1, lat1, ens_dim=False, **kwargs):
    """
    Extract coordinate pairs to run from nearest neighbors in case NaN values are deteced in the current series
    
    Parameter    
    ----------
    data_arr: xr.core.dataarray.DataArray, the input data array to look up the nearest points where the no NaN series can be extracted.
    data_ser: pd.Series, the input series with NaN vlaues detected
    lon1: float, the longitude of the point.
    lat1: float, the latitude of the point.
    ens_dim: bool, if the ensemble member dimension exist
    kwargs: any additional positional arguments
    """
    # Obtain the longitude and latitude name used in the data array
    lon_name, lat_name= get_latlon_names(data_arr)
    # Extract the longitude and latitude vectors
    lon_vec = data_arr.coords[lon_name]
    lat_vec = data_arr.coords[lat_name]
    # Check the integer index of the supplied lon and lat, where NaN is encountered
    lon_index = int(np.where(lon_vec.data==lon1)[0])
    lat_index = int(np.where(lat_vec.data==lat1)[0])
    data_ser_test = data_ser.copy(deep=True)
    if not all((lon_index>=0,lat_index>=0)): # "Negative values are found in coordinate indexing"
        if len(kwargs)!=0:
            kwargs["Bug_dict"].update({"lon"+str(lon1)+"_lat"+str(lat1):'Negative coordinate indexing are found that should not happen'})
        print("Negative coordinate indexing are found that should not happen")
        return data_ser_test
    # None of situations below should happen;
    if any(( (lon_index == 0) and (lat_index==0), # Both lon and lat are the first elements
             (lon_index == (len(lon_vec)-1)) and (lat_index==(len(lat_vec)-1)), # Both lon and lat are the last elements
             (lon_index==0) and (lat_index==(len(lat_vec)-1)), # First lon combines with the last lat
             (lat_index==0) and (lon_index == (len(lon_vec)-1)) )): # First lat combines with the last lon
        print("The boundary coordinate pairs are encountered in the first place and can not continue searching points to fill NaN")
        if len(kwargs)!=0:
            kwargs["Bug_dict"].update({"lon"+str(lon1)+"_lat"+str(lat1):'the boundary coordinate pairs are encountered in the first place and can not continue searching points to fill NaN!'}) 
        return data_ser_test
    # Create an empty list to store the simulation series
    count=0
    while any(pd.isnull(data_ser_test)):
        # Return the updated lon1 and lat1
        lon1_update = lon_vec.data[lon_index]
        lat1_update = lat_vec.data[lat_index]
        # Extract the respective simulation series for the updated lon and lat
        if not ens_dim:
            data_ser_test = data_arr.loc[{lon_name:lon1_update, lat_name:lat1_update}]
        else:
            data_ser_test = data_arr.loc[{lon_name:lon1_update, lat_name:lat1_update, "number":kwargs["ens_member"]}]
        # Update the integer index to ensure searching for nearest index
        # Nearest search along the lat and lon vectors sequentially            
        if count%4==0:
            lat_index = lat_index-1
        elif count%4==1:
            lat_index = lat_index+1
        elif count%4==2:
            lon_index = lon_index-1
        elif count%4==3:
            lon_index = lon_index+1
        #  Count plus one for each iteration
        count += 1
        # The index can not exceed the length of the vector of either longitude or latitude or being negative
        #list(product((lon_index<=0,lat_index<=0,(lon_index +1) >= len(lon_vec),(lat_index +1) >= len(lat_vec)),repeat=2))
        if lon_index <= 0:
            lon_index = 0
        elif lat_index <= 0:
            lat_index = 0
        elif (lon_index +1) >= len(lon_vec):
            lon_index = len(lon_vec)-1
        elif (lat_index +1) >= len(lat_vec):
            lat_index = len(lat_vec)-1
        # None of situations below should happen; It means the limit of lon and lat vector has been reached
        if any((  (lon_index == 0) and (lat_index==0), # Both lon and lat are the first elements
                 (lon_index == (len(lon_vec)-1)) and (lat_index==(len(lat_vec)-1)), # Both lon and lat are the last elements
                 (lon_index==0) and (lat_index==(len(lat_vec)-1)), # First lon combines with the last lat
                 (lat_index==0) and (lon_index == (len(lon_vec)-1)) )): # First lat combines with the last lon
            print("The boundary coordinate pairs are encountered and can not continue searching points to fill NaN")
            if len(kwargs)!=0:
                kwargs["Bug_dict"].update({"lon"+str(lon1)+"_lat"+str(lat1):'the boundary coordinate pairs are encountered and can not continue searching points to fill NaN!'}) 
            return data_ser_test.to_series()
        # Here the lon_index and lat_index need to be checked if they are unchanged for a certain number of loops. Panda series is returned
    return data_ser_test.to_series() # When the while loop is break, meaning Non-NaN value series is found and returned in the form of Panda series
###############################################################################################################################################################################################################
def Derive_ML_FitData(T_min, T_max, phenology_pred, initial_date = pd.to_datetime("09-01",format="%m-%d"), lead_time = 7,
                        hot_day_threshold = 25, cold_day_threshold = 0, GDD_threshold= 0): # months = list(range(9,12+1)) + list(range(1,3+1)) ):
    """
    Derive a dataframe from the observed weather and phenology simulation. The anomaly values relative to 
    climatology mean (over all study years) is computed. 
    
    Parameters
    ----------
    T_min: panda core series, the full minimum temperature series covering the study period (intend to be subset)
    T_max: panda core series, the full maximum temperature series covering the study period (intend to be subset)
    phenology_pred: panda core series, the time series of phenology predictions for a given stage
    initial_date: datetime/timestamp, the initial calculation/forecast date
    lead_time: int, months after the initia_date to form a range of date
    hot_day_threshold: int, the maximum temperature threshold to compute the hot days. Defaul to 25 C° following the CDO definition of eca_csu
    cold_day_threshold: int, number of months from the initial_date. Defaul to 0 C° following the CDO definition of eca_cfd
    GDD_threshold: float, the base temperature used to calculate the phenology progressing rate. Default to budburst thermal forcing module parameter, variety-specific value. 
    """
    assert np.array_equal(T_min.index.year.unique(), T_max.index.year.unique(), equal_nan=False), "number of years are not equal between Tmin and Tmax series" # DO not compare NaN to ensure no NaN data is extracted
    # Access the full study years
    full_years = T_min.index.year.unique()
    #study_years = years[1:] # For budburst simulations, the study years shall start from the second year after the first one.
    ML_df_dict= {} # Create an empty dictionary to store values
    # Initialize the dict for each independent variables (or input features for classification)
    ML_df_dict["hot_days"] = {}
    ML_df_dict["cold_days"] = {}
    ML_df_dict["seasonal_min"] = {}
    #ML_df_dict["seasonal_mean"] = {}
    ML_df_dict["seasonal_max"] = {}
    ML_df_dict["seasonal_GDD"] = {}
    # Monthly minimum and maximum temperature predictors
    months = list(pd.period_range(initial_date,initial_date+DateOffset(months=lead_time-1),freq="M").month) # Compute the target months
    # Define the monthly stat for calculations
    monthly_stats = ["Tmin", "Tmax"]
    for monthly_stat in monthly_stats:            
        for month in months:
            month_abbr = calendar.month_abbr[month]
            ML_df_dict[month_abbr+ "_{}".format(monthly_stat)] = {}
    # Initialize the dict for target output variable
    ML_df_dict["phenology_SM"] = {}
    #initial_date_M_D_format =  initial_date.strftime("%m-%d")
    for twoyear_list, phenology_pred_year in zip(sort_twoyear_combo(list(full_years)), phenology_pred):
        # Compute the mean temperature series
        T_mean = pd.Series(np.nanmean([T_min,T_max], axis=0), index=T_min.index, name="tg")
        # Subset the consecutively 2-year climate so that it begins and ends over a specific range within this 2-year
        seasonal_climate_Tmin = subset_two_year_climate(T_min, twoyear_list, initial_date, lead_time)
        seasonal_climate_Tmax = subset_two_year_climate(T_max, twoyear_list, initial_date, lead_time)
        seasonal_climate_Tmean = subset_two_year_climate(T_mean, twoyear_list, initial_date, lead_time)
        #target_period = pd.period_range(initial_date,initial_date+DateOffset(months=lead_time-1),freq="M")
        # Compute the number of hot days (>=25 celcius degree)
        hot_days= seasonal_climate_Tmax[seasonal_climate_Tmax>=hot_day_threshold].count()
        if max(twoyear_list) not in ML_df_dict["hot_days"].keys():
            ML_df_dict["hot_days"][max(twoyear_list)] = hot_days # Attach to target dict
        # Compute the number of cold days (<0 celcius degree)
        cold_days= seasonal_climate_Tmin[seasonal_climate_Tmin<=cold_day_threshold].count()
        if max(twoyear_list) not in ML_df_dict["cold_days"].keys():
            ML_df_dict["cold_days"][max(twoyear_list)] = cold_days # Attach to target dict
        # Compute the minimum temperature
        min_temperature = np.nanmean(seasonal_climate_Tmin)
        if max(twoyear_list) not in ML_df_dict["seasonal_min"].keys():
            ML_df_dict["seasonal_min"][max(twoyear_list)] = min_temperature # Attach to target dict   
        # Compute the minimum temperature
        max_temperature = np.nanmean(seasonal_climate_Tmax)
        if max(twoyear_list) not in ML_df_dict["seasonal_max"].keys():
            ML_df_dict["seasonal_max"][max(twoyear_list)] = max_temperature # Attach to target dict   
        # Compute the average temperature
        # mean_temperature = np.nanmean(seasonal_climate_Tmean)
        # if max(twoyear_list) not in ML_df_dict["seasonal_mean"].keys():
        #     ML_df_dict["seasonal_mean"][max(twoyear_list)] = mean_temperature # Attach to target dict
        # Compute the GDD with base temperature corresponding to those of budburst thermal forcing module, i.e. variety-specific
        seasonal_climate_GDD = max(seasonal_climate_Tmean.apply(GDD_simple, args=(GDD_threshold,)).cumsum())
        if max(twoyear_list) not in ML_df_dict["seasonal_GDD"].keys():
            ML_df_dict["seasonal_GDD"][max(twoyear_list)] = seasonal_climate_GDD # Attach to target dict
        # # Compute the budburst DOY
        # dormancy_sm_ob, budburst_sm_ob = run_BRIN_model(T_min.loc[T_min.index.year.isin(twoyear_list)], T_max.loc[T_max.index.year.isin(twoyear_list)], 
        #                                                 Q10=1.52, CCU_dormancy = 183.23, 
        #                                                 T0_dormancy = 213, CGDH_budburst = 293.1, 
        #                                                 TMBc_budburst= 25, TOBc_budburst = 5.28, Richarson_model="daily")
        # Compute monthly average minimum and maximum temperatures as part of the predictors from September to March
        for month in months: 
            monthly_Tmin = np.nanmean(seasonal_climate_Tmin[seasonal_climate_Tmin.index.month == month])
            monthly_Tmax = np.nanmean(seasonal_climate_Tmax[seasonal_climate_Tmax.index.month == month])
            # Access the month abbreviation
            month_abbr = calendar.month_abbr[month]
            # Attach the monthly stat to target dict
            if max(twoyear_list) not in ML_df_dict[month_abbr+ "_Tmin"].keys():
                ML_df_dict[month_abbr+ "_Tmin"][max(twoyear_list)] = monthly_Tmin # Attach to target dict
            if max(twoyear_list) not in ML_df_dict[month_abbr+ "_Tmax"].keys():
                ML_df_dict[month_abbr+ "_Tmax"][max(twoyear_list)] = monthly_Tmax # Attach to target dict
        # Attach the simulated budburst DOY into target dict
        if max(twoyear_list) not in ML_df_dict["phenology_SM"].keys():
            if not pd.isna(phenology_pred_year):
                ML_df_dict["phenology_SM"][max(twoyear_list)] = int(phenology_pred_year.dayofyear) # Attach to target dict
            elif pd.isna(phenology_pred_year):
                ML_df_dict["phenology_SM"][max(twoyear_list)] = np.nan
            else:
                continue
    # Create the target dataframe with absolute values
    ML_df_abs = pd.DataFrame(ML_df_dict)
    # Fill na if any
    if any(ML_df_abs.isnull().any()):
        ML_df_abs.fillna(method="ffill", inplace=True)
    # Copy the df to a new one to calculate the anomalous phenology occurrence 
    ML_df_category = ML_df_abs.copy(deep=True)
    for col_name in ML_df_category.columns:
        #ML_df_anomaly[col_name] = ML_df_anomaly[col_name].apply(lambda x: x - np.nanmean(ML_df_anomaly[col_name])) # Apply the abnormaly calculation in each yearly value
        ML_df_category[col_name] = ML_df_category[col_name].apply(categorize_variable, args=(ML_df_category[col_name],) ) # Categorize the variables according to its percentile distribution over years
    
    # Round the element to the nearest integer for hot and cold days stat
    # ML_df_anomaly["hot_days"] = ML_df_anomaly["hot_days"].apply(lambda x: round(x))
    # ML_df_anomaly["cold_days"] = ML_df_anomaly["cold_days"].apply(lambda x: round(x))
    # # Transform the phenolgoy anomalous simulations as early, late or normal
    
    # ML_df_anomaly["hot_days"]   = ML_df_anomaly["hot_days"].apply(categorize_variable, args=ML_df_anomaly["hot_days"])
    
    # ML_df_anomaly["phenology_SM"]   = ML_df_anomaly["phenology_SM"].apply(phenology_categoty, args=(5,))
    return ML_df_category  
###############################################################################################################################################################################################################  
def subset_dataset(xarray_data, lon_target_vector, lat_target_vector, study_period, preceding_year = True):
    """
    Subset the xarray dataset so that only study region with study period is selected
    
    Parameter    
    ----------
    xarray_data: xr.core.dataarray.DataArray, the input original Xarray DataArray to subset
    lon_target_vector: array-like, the longitude vector 
    lat_target_vector: array-like, the latitude vector
    study_period: array-like, the study period 
    preceding_year: bool, if the preceding one year should be included in the analysis
    """
    # Subset the dataset to study region                                                  
    xarray_data_subset_region = xarray_data.where(((xarray_data.latitude >= min(lat_target_vector)-2) & (xarray_data.latitude <= max(lat_target_vector)+2)) &
                                                             ((xarray_data.longitude >= min(lon_target_vector)-2) & (xarray_data.longitude <= max(lon_target_vector)+2)),
                                                              drop=True)        
    # Then subset the dataset to study time period
    if preceding_year:
        xarray_data_subset_region_time = xarray_data_subset_region.where(xarray_data_subset_region.time.dt.year.isin([min(study_period)-1]+list(study_period)),  drop=True)  
    else:
        xarray_data_subset_region_time = xarray_data_subset_region.where(xarray_data_subset_region.time.dt.year.isin(list(study_period)),  drop=True)
         
    return xarray_data_subset_region_time
###############################################################################################################################################################################################################
def add_gdf_labels(axe_input, feature_input, col_name="DOC_ID", digits=2, 
                   textcrs="offset points", xytext_position = (-0.5,-0.5), 
                   fontsize=1, fontstyle= 'italic', fontweight="bold", fontcolor="blue", 
                   map_proj_CRS = ccrs.PlateCarree() ): # Define the projection type
    """
    Add the text labels onto the shape features, mainly based on geodataframe utility
    
    Parameters
    ----------
    axe_input: the matplotlib geo-axe instance, the axe where data is plotted 
    feature_input: geodataframe, the input feature that is read into the geodataframe
    col_name: str, the column name of geodataframe that holds information on the desired text to be labelled 
    digits: int, number of decimal digits to keep in the computed geographic coordinate   
    textcrs: str, the coordinate system that xytext is given in.
    xytext_position: tuple, the position (x, y) to place the text at.
    fontsize: int, the font size to be labelled
    fontstyle: str, the font style to apply 
    """
    assert isinstance(feature_input, gpd.geodataframe.GeoDataFrame), "the input feature input is not an instance of geodataframe, but the format of {} is found".format(type(feature_input))
    feature_input = feature_input.to_crs(map_proj_CRS)
    for feature_entry in feature_input[col_name]:
        feature_entry_shp = feature_input.loc[feature_input[col_name]==feature_entry,:]
        centroid_point = feature_entry_shp.centroid # Compute the centroid point of the analyzed polygon feature
        centroid_lon1 = round(float(centroid_point.x), digits) # Round to desired number of decimal digits
        centroid_lat1 = round(float(centroid_point.y), digits) # Round to desired number of decimal digits
        # Add the text label into the shape file
        axe_input.annotate(str(feature_entry), xy=(centroid_lon1, centroid_lat1), xytext=xytext_position, textcoords=textcrs, fontsize =fontsize, fontstyle =fontstyle, c = fontcolor, fontweight =fontweight)
###############################################################################################################################################################################################################
def add_gridlines(subplot_axe, grid_lons,  grid_lats, proj= ccrs.PlateCarree(), top_labels=True,bottom_labels= False, left_labels = True,
                  right_labels = False, xlines= False, ylines = False, fontsize = 2, font_col = "gray", font_style = "bold"):
    """
    Add the grid line coordinates for the map plot
    
    Parameter    
    ----------
    subplot_axe: mathplotlib axe object, the input axe where the grid lines are casted 
    grid_lons: One-D array-like, the target sequence of longitude to be plotted 
    grid_lats: One-D array-like, the target sequence of latitude to be plotted 
    proj: str/Projected CRS, the target projection CRS to use 
    top_labels,bottom_labels, left_labels, right_labels: bool, which side of frame to draw the gridline labels
    xlines, ylins: bool,  whether to draw the gridlines
    fontsize: int, font size in use for the gridline labels
    font_col: str, color to use for the font
    """
    gl = subplot_axe.gridlines(crs=proj, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--',  xpadding=1, ypadding=2.5)
    # Add the grid line lables
    gl.top_labels = top_labels
    gl.bottom_labels  = bottom_labels
    gl.left_labels = left_labels
    gl.right_labels  = right_labels
    # DO not show xlines grid lines
    gl.xlines  = xlines
    gl.ylines  = ylines
    # Add the x- and y-locators in the subplot map
    gl.xlocator = mticker.FixedLocator(grid_lons) # Pass the desired longitude vector to be plotted on the map
    gl.ylocator = mticker.FixedLocator(grid_lats) # Pass the desired latitude vector to be plotted on the map
    gl.xformatter = LONGITUDE_FORMATTER # Format the locators
    gl.yformatter = LATITUDE_FORMATTER # Formate the locators
    # Add the label styles
    gl.xlabel_style = {'size': fontsize, 'color': font_col, "weight": font_style}
    gl.ylabel_style = {'size': fontsize, 'color': font_col, "weight": font_style}
###############################################################################################################################################################################################################
def add_scale_bar(subplot_axe, lonvec, latvec, dimension = "si-length", units="km", length_fraction=0.2, location="lower right", sep=5,
                     pad=0.2, label_loc="bottom", width_fraction= 0.01, font_properties={"size":4}):
    """
    Add the scale bar into the subplot
    
    Parameter    
    ----------
    subplot_axe: mathplotlib axe object, the input axe where the grid lines are casted 
    lonvec: One-D array-like, the target sequence of longitude to be plotted 
    latvec: One-D array-like, the target sequence of latitude to be plotted 
    dimension: str, dimension of dx and units
    units: str, the distance unit to use
    length_fraction: float, desired length of the scale bar as a fraction of the subplot's width
    location: str, a location code, same as matplotlib's legend
    sep: float, separation in points between the scale bar and scale, and between the scale bar and label. 
    pad: float, padding inside the box, as a fraction of the font size
    label_loc: str, location of the label with respect to the scale bar
    width_fraction: float, width of the scale bar as a fraction of the subplot's height
    font_properties: dict, a dictionary with specified font size, style and color etc.
    """
    dx = great_circle(np.max(lonvec), np.min(latvec), np.max(lonvec)+1, np.min(latvec)) # compute the distance between two points at the latitude (Y) you wish to have the scale represented
    scale_bar = ScaleBar(dx, units=units, dimension= dimension, length_fraction=length_fraction, location=location,sep=sep,
                         pad=pad,label_loc=label_loc, width_fraction= width_fraction, font_properties=font_properties)
    subplot_axe.add_artist(scale_bar)
###############################################################################################################################################################################################################
def compute_df_RMSE(input_df, ob_col):
    """
    Compute the RMSE per row of df given the observed series
    Mandatory Parameters
    ---------- 
    input_df: df, input df to process with 
    ob_col: series, the observed series/column to compare with input_df
    """
    # Obtain the RMSE for each year over all ensemble members
    df_Sqr_diff = input_df.sub(ob_col, axis=0).applymap(lambda x:x**2).sum(axis=1, skipna=True) # Compute the sum of squared diff for all ensenble members over all years 
    # A series of RMSE over study years is returned
    df_RMSE =  np.sqrt(df_Sqr_diff.div(len(input_df.columns))) # Compute the RMSE
    return df_RMSE
###############################################################################################################################################################################################################
def write_CF_attrs(da, CRS_str = "epsg:4326", grid_mapping_name ="CRS"):
    """
    Write the CF convention to the target dataset
    
    Mandatory Parameters
    ---------- 
    da: xarray data array, the input xarray dataset object
    CRS_str: str, the CRS string
    grid_mapping_name: str, the grid mapping name
    """
    if isinstance(da,  xr.core.dataset.Dataset):
        # Get the underlying var name
        da_var_name = list(da.data_vars)[0]
        da_array = da[da_var_name]
    else:
        da_array = da.copy(deep=True) # Then it must be a xr.core.dataset.Dataset instance
    # Get the actual lon and lat name for the data array
    lon_name, lat_name = get_latlon_names(da_array)
    # Write the CF convention to the target dataset
    da_array.rio.write_crs(CRS_str, grid_mapping_name = grid_mapping_name, inplace=True).rio.set_spatial_dims(x_dim= lon_name, y_dim=lat_name, inplace =True).rio.write_coordinate_system(inplace=True)
    return da_array
###############################################################################################################################################################################################################
def time_depend_GDD(T_mean_input, fc_init_date, flo_time_ref = pd.to_datetime("07-31", format="%m-%d"),
                    ver_time_ref = pd.to_datetime("09-30", format="%m-%d"), GDD_threshold=0, cum_sum=False):
    """
    Compute GDD for a specified time period per year
    
    Mandatory Parameters
    ---------- 
    T_mean_fc: series, the daily mean temperature series
    fc_init_date: datetime, the initial forecast datetime of the forecast datasets
    flo_time_ref: datetime, the ending datetime that is fixed to represent a common flowering time in local condition
    ver_time_ref: datetime, the ending datetime that is fixed to represent a common veraison time in local condition
    """
    # Confirm a list or an array of months that subset yearly data
    flo_month_ref = pd.period_range(fc_init_date, flo_time_ref, freq="M").month
    ver_month_ref = pd.period_range(fc_init_date, ver_time_ref, freq="M").month
    # Subset the daily input temperature series with specified list of monhths  
    T_mean_flo_ser = T_mean_input.loc[T_mean_input.index.month.isin(flo_month_ref)]
    T_mean_ver_ser = T_mean_input.loc[T_mean_input.index.month.isin(ver_month_ref)]
    # Compute the daily Degree Day values for the subset periods
    T_mean_flo_dd = T_mean_flo_ser.apply(GDD_simple, args=(GDD_threshold,))
    T_mean_ver_dd = T_mean_ver_ser.apply(GDD_simple, args=(GDD_threshold,))
    if not cum_sum: # Not compute the cumulative sum but only compute the yearly sum values
        # Groupby and aggregate into yearly data
        T_mean_GDD_flo = T_mean_flo_dd.groupby(T_mean_flo_dd.index.year).agg(np.nansum)
        T_mean_GDD_ver = T_mean_ver_dd.groupby(T_mean_ver_dd.index.year).agg(np.nansum)
        # Asssert both flo and ver GDD series having the same number of years
        assert np.array_equal(T_mean_GDD_flo.index, T_mean_GDD_ver.index, equal_nan=True), "Unequal number of years found between the flowering and veraison series"
        study_years = T_mean_GDD_flo.index # Access the underlying study year series from the flowering series
        date_range = pd.date_range("{}-12-31".format(str(min(study_years))), "{}-12-31".format(str(max(study_years))), freq="Y") 
        # Re-formuate the date time series into the output series
        T_mean_GDD_flo.index = date_range
        T_mean_GDD_ver.index = date_range
        return T_mean_GDD_flo, T_mean_GDD_ver
    else:
        # Groupby and compute the cumulative sum of each year
        T_mean_GDD_flo_cumsum = T_mean_flo_dd.groupby(T_mean_flo_dd.index.year).agg(np.cumsum)
        T_mean_GDD_ver_cumsum = T_mean_ver_dd.groupby(T_mean_ver_dd.index.year).agg(np.cumsum)
        # # Compute the 5th, median and 95th percentile for the cumulative GDD of flowering stage. Note the last element is removed 
        # P5_flo = T_mean_GDD_flo_cumsum.groupby(T_mean_GDD_flo_cumsum.index.day_of_year).agg(np.percentile, q=5)[:-1]
        # P50_flo = T_mean_GDD_flo_cumsum.groupby(T_mean_GDD_flo_cumsum.index.day_of_year).agg(np.percentile, q=50)[:-1]
        # P95_flo = T_mean_GDD_flo_cumsum.groupby(T_mean_GDD_flo_cumsum.index.day_of_year).agg(np.percentile, q=95)[:-1]
        # # Compute the 5th, median and 95th percentile for the cumulative GDD of veraison stage
        # P5_ver = T_mean_GDD_ver_cumsum.groupby(T_mean_GDD_ver_cumsum.index.day_of_year).agg(np.percentile, q=5)[:-1]
        # P50_ver = T_mean_GDD_ver_cumsum.groupby(T_mean_GDD_ver_cumsum.index.day_of_year).agg(np.percentile, q=50)[:-1]
        # P95_ver = T_mean_GDD_ver_cumsum.groupby(T_mean_GDD_ver_cumsum.index.day_of_year).agg(np.percentile, q=95)[:-1]
        # Note the last element is removed to ensure consistency over years
        return T_mean_GDD_flo_cumsum, T_mean_GDD_ver_cumsum
###############################################################################################################################################################################################################
def sig_test_p(input_p, thresh_p = 0.05):
    """
    Convert the probability of a given statistic test into the significance test ressult 
    
    Mandatory Parameters
    ---------- 
    input_p: float, the probability of a given statistical test result that is used to evaluate the significance.
    """
    if input_p <= thresh_p: # Working with the probability of 95%
        sig_result = 1 # 1 or True for significance 
    else:
        sig_result = 0 # 0 or False for non-significance
        
    return sig_result
###############################################################################################################################################################################################################
def compute_fairRPS(ref, fc, fc_time_step, 
                    k_categories = 3, M=25, full_series=False):
    """
    Compute the fair ranked probability skill scores between reference series and an ensemble probability forecast/predictions
    This function can not be implemented !! Problems exist
    Mandatory Parameters
    ---------- 
    ref: array/series, the reference series to be compared with by the ensemble forecast
    fc: dataframe, the forecast dataframe that represents the ensemble probability forecast/predictions (each column corresponds to a ensemble member forecast)
    fc_time_step: array, a series of study years/time step
    k_categories: int, number of discrete categories in the ref series
    M: int, number of ensemble members in probability forecast
    full_series: bool, if the full series (over time step) of skill score needs to be returned along with the final score.
    """
    # Step 1. Check the format and shape of input are ok and ready to use. Then copy them into function variables
    assert isinstance(ref, (pd.core.series.Series,np.ndarray)), "the reference series does not follow a required format, {} is found".format(type(ref))
    assert isinstance(fc, pd.core.frame.DataFrame), "the focast datasets does not follow the dataframe format, {} is found".format(type(ref))
    # Copy the input variables into function-inner variables
    ref_func = ref.copy(deep=True) # Copy the input ref series into the function-specific ref_func variable
    fc_func = fc.copy(deep=True) # Copy the input fc df into the function-specific fc_func variable
    # Step 2. Confirm number of timesteps are equal between reference series and each of the series
    if len(ref_func) != fc_func.shape[0]: # If the length of reference series does not equal to the length of rows in forecast df, errors are raised
        raise ValueError("Number of entires are not equal across the time_step dimension")
    else: # When the test is passed, assign the same time_step into their index 
        ref_func.index = fc_time_step
        fc_func.index = fc_time_step
    # Step 3. Convert the continuous variables into the tercile group variables
    ref_tercile = ref_func.apply(tercile_var, args=(ref_func,) ) # Obtain the reference tercile series   
    for col_name, col_ser in fc_func.items(): # Iterate over each column of fc df to get the desired df values
        fc_func[col_name] = col_ser.apply(tercile_var, args=(col_ser,True), Q1=ref_func.quantile(q=0.33),Q2=ref_func.quantile(q=0.67) ) 
    fc_tercile = fc_func.copy(deep=True) # Obtain the tercile series for all forecast members
    # Step 4. Compute FairRPS for each of the forecast-event pair
    fairRPS_dict = {} # Create an empty list to store the score value per time step
    for time_step in fc_time_step:
        # Obtain the reference value for a given time step
        ref_time = ref_tercile.loc[time_step]
        # Obtain all ensemble member values for a given time step
        fc_ens_time = fc_tercile.loc[time_step]
        # Create an empty list to store the FairRPS score over categories K per time step
        fairRPS_k = []
        for k_cat in range(k_categories): # Iterate over each category to compute the score per time step 
            k_cat_true = k_cat+1 # Since it is 0 based, it needs to add 1 to be 1-based
            if ref_time == k_cat_true: # If the observation occurred, give 1 or give 0 
                O_k = 1
            else:
                O_k = 0 
            # E_K is number of ensemble members that correctly predict the event category k_cat_true
            E_k = len(fc_ens_time[fc_ens_time==k_cat_true]) # Note this value can be 0
            # Compute the FPRS score per time step
            fairRPSt = ((E_k/M) - O_k)**2  - ((E_k * (M - E_k))/ ((M**2) * (M-1)))
            # Append the score into the target list
            fairRPS_k.append(fairRPSt)
        # Attach the summed values into the target dict
        fairRPS_dict[time_step] = np.nansum(fairRPS_k) # Summ all category values as the value per time step
    # Step 5. Obtain a timeseries of fairRPS
    fairRPS_ser = pd.Series(fairRPS_dict, name="fairRPS_ser")
    # Step 6. Compute the final fairRPS score as the average over the series
    fairRPS_final = np.nanmean(fairRPS_ser)
    # Step 7. Return the results depending on the option set
    if full_series: 
        return (fairRPS_ser, fairRPS_final) # Return, beyong the final score, the series values of fairRPS
    else:
        return fairRPS_final
###############################################################################################################################################################################################################
def GSS(input_data_df, target_category_event, fc_thresh, ref_index_name = "ob", score_type = "ETS"):
    """
    Compute the Gilbert Skill Score (GSS) or called Equitable Threat Score (ETS) to measure how well did the forecast "yes" events 
    correspond to the observed "yes" events (accounting for hits due to chance).
    
    Reference1: https://cawcr.gov.au/projects/verification/ 
    Reference2: https://www.nature.com/articles/s41598-018-19586-6#Sec4
    
    Mandatory Parameters
    ---------- 
    input_data_df: df, the input dataframe with index being the time stamps, first column being reference records of categorical events, the rest columns being the forecast records of categorical events for all ensemble members
    target_category_event: int, the integer representation of a given categorical event. For example, 1/2/3 correspond to early/normal/late categorical event
    fc_thresh: float, a percentage that represents the threshold beyond which the forecast event is considered to occur
    ref_index_name: str, the column name of input_data_df that denotes the reference categorical events (first column)
    score_type: str, the type of GSS score that is computed. Two options are available, TS or ETS. The two scores are essentially the same, despite TS is not adjusted for random chance.
    """

    # Check the input data that meet the requirement
    assert isinstance(input_data_df, pd.core.frame.DataFrame), "current implementation only supports the input type of panda data frame, but the {} type is found".format(type(input_data_df))
    # Access the total study time stamps from the input dataframes. The GSS is computed along the studied time stamps, in other words, forecast events are performance along the time stamps
    study_time_stamps = input_data_df.index
    # Define the essential terms for the GSS, i.e. starting with 0 for each essential term that compose the GSS
    hit = 0
    miss = 0
    correct_negative = 0
    false_alarm = 0
    for study_time_stamp in study_time_stamps:
        # Access the yearly series content
        row_ser = input_data_df.loc[study_time_stamp,:]
        # Obtain the reference category event
        ref_category = row_ser.loc[row_ser.index.to_series().str.contains(ref_index_name, regex=False)]
        # Deteremine if the reference event is considered occurring according to the studied catgorical event
        if int(ref_category) == target_category_event:
            ref_event = "Yes"
        else:
            ref_event = "No"
        # Obtain the forecast series of category events over all ensemble members
        fc_events = row_ser.loc[~row_ser.index.isin([ref_category.index])]
        # Deterimne if the forecast event is considered to occur according to the results of all forecast ensemble members, as well as the threshold defined for the occurrence
        if (sum(fc_events == target_category_event)/len(fc_events)) > fc_thresh: # Fraction of ensemble members that exceed the defined threshold, which assumed to occur
            fc_event = "Yes"
        else:
            fc_event = "No"
        # Update the hit, miss and false alarm
        if fc_event == "Yes": # If the forecasy event is about to occur
            if ref_event == "Yes": # Event forecast to occur, and did occur. Hit plus 1 
                hit+=1
            elif ref_event == "No": # Event forecast to occur, but did not occur. false_alarm plus 1 
                false_alarm+=1
        elif fc_event == "No": # If the forecasy event is about not to occur
            if ref_event == "Yes": # Event forecast not to occur, but did occur. miss plus 1
                miss+=1
            elif ref_event == "No": # Event forecast not to occur, and it did not occur. correct_negative plus 1
                correct_negative+=1
    if score_type == "ETS":
        # Compute the expected fraction of hits for a random forecast
        hit_random = ( (hit + miss) * (hit + false_alarm) )/ (hit+ miss+ correct_negative+false_alarm)
        # Finally, based on the term obtained, compute the GSS score for a given categorical event over all forecast years for this grid point
        if  hit + miss + false_alarm - hit_random ==0:
            GSS_point = np.nan
        else:
            GSS_point = (hit - hit_random)/ (hit + miss + false_alarm - hit_random)
    elif score_type == "TS":
        GSS_point = hit / (hit + miss + false_alarm)
    return GSS_point
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def kdf_plot(subplot_axe, x, kde_5th, kde_median, kde_95th, color, 
             linewidth=0.8, linestyle='-', fill_linewidth=0.5, fill_color=None, alpha=0.5, fill_regions=True, prob_area=False):
    """
    Make the kernel density function (kdf) plot given the input datasets while fill the areas using the kdf of 5th percentile and the kdf of 95th percentile
    
    Mandatory Parameters
    ---------- 
    subplot_axe: matplotlib axe instance, the axe instance to make the subplot
    x: one-D array-like, the input sample dataset that represents the x-values of fitted kernel density function
    kde_5th: kds instance, a fitted kdf that represents the kdf for the 5th percentile of data
    kde_median: kds instance, a fitted kdf that represents the kdf for the 50th percentile (median) of data
    Mkde_95th kds instance, a fitted kdf that represents the kdf for the 95th percentile of data
    color: str, color used to make the line plot and line filling
    linewidth: float, the line width of (median kdf) line plot
    linestyle: str, the line width of (median kdf) line plot
    fill_linewidth: float, the width of filling lines
    fill_color: str, the filling color
    alpha:float, the alpha value of filling color
    fill_regions: bool, if it is to fill the region between two input curves
    prob_area:bool, if it is to fill the probability area
    """
    # Make the median line plot of the kdf that represent the kdf of the 50th percentile (median) of data
    subplot_axe.plot(x, kde_median.pdf(x), color=color, linewidth=linewidth, linestyle=linestyle)
    if fill_regions:
        #fill_x_first = kde_5th.pdf(x) >= kde_95th.pdf(x)
        # Fill the regions with specified colors where y1(kde_5th.pdf(x)) >=y2 (kde_95th.pdf(x))
        subplot_axe.fill_between(x, kde_5th.pdf(x), kde_95th.pdf(x), # Provided the x position, ymax, ymin positions to fill 
                  where= kde_5th.pdf(x) >= kde_95th.pdf(x), interpolate=True, # Only fill the x regions where kde_5th.pdf(x) >= kde_95th.pdf(x)
                  facecolor=fill_color, # The fill color
                  color= fill_color,   # The outline color
                  edgecolors = fill_color, # The line edge color
                  linewidth = fill_linewidth,
                  alpha=alpha)
        # Fill the regions with specified colors where y1 (kde_5th.pdf(x)) <y2 (kde_95th.pdf(x))
        subplot_axe.fill_between(x, kde_95th.pdf(x), kde_5th.pdf(x), # Provided the x position, ymax, ymin positions to fill 
                  where= kde_95th.pdf(x) >= kde_5th.pdf(x), interpolate=True, # Only fill the x regions where kde_5th.pdf(x) >= kde_95th.pdf(x)
                  facecolor=fill_color, # The fill color
                  color= fill_color,   # The outline color
                  edgecolors = fill_color, # The line edge color
                  linewidth = fill_linewidth,
                  alpha=alpha)
    elif prob_area:
        subplot_axe.fill_between(x, kde_median.pdf(x), 0, # Provided the x position, ymax, ymin positions to fill 
                  facecolor=fill_color, # The fill color
                  color= fill_color,   # The outline color
                  edgecolors = fill_color, # The line edge color
                  linewidth = fill_linewidth,
                  alpha=alpha)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def zonal_statistics(data_array, geometry_file, col_name, stat="mean", **kwargs):
    """
    Compute the zonal statistics based on input geometry vector file and data_array xarray format
    
    Parameter    
    ----------
    data_array: xarray dataarray, the input data array that holds the point/pixel values. The data must be 2-D: lon/lat or x/y similarly only.
    geometry_file: geo-dataframe, a vector geometry that defines the target study regions/AOI
    col_name: str, the col_name in the geo-dataframe to access 
    stat: str, the target statistics to compute
    kwargs: any additionaly key word arguments
    """
    # Pre-define a list of target stat names used for spatial zonal analysis
    mean_str=["mean","average","Mean","Average"]
    median_str=["median","Median"]
    std_str=["std","STD","standard deviation", "Standard Deviation"]
    cv_str=["cv","CV","coefficient of variation","Coefficient of Variation"]
    range_str=["range","Range"]
    major_str = ["major","Major", "Main", "main", "most frequent"]
    assert isinstance(data_array, (xr.core.dataset.Dataset, xr.core.dataarray.DataArray)), "the input data array is not an instance class of xarray data array, but the format of {} is found".format(type(data_array))
    assert isinstance(geometry_file, gpd.geodataframe.GeoDataFrame), "the feature input is not an instance class of geodataframe, but the format of {} is found".format(type(geometry_file))
    # 1. Retrieven the lon and lat vectors from the input data array
    lon_name, lat_name = get_latlon_names(data_array) # Access the data array longitude and latitude names
    lon_vector = data_array.coords[lon_name].data # Access the longitude vector
    lat_vector = data_array.coords[lat_name].data # Access the latitude vector
    # 2. Obtain the coordinate list that is assembled from the supplied input data array
    coordinates = list(product(lon_vector, lat_vector)) 
    # 3. Obtain a list of coordinates inferred from the data array
    geometry_list = geometry_file[col_name]
    # 4. Create a target array that is copied from the input data array
    target_array = data_array.copy(deep=True)
    # 4.1 Create an empty array that is filled with NaN values
    init_data = np.empty(target_array.data.shape)
    init_data[:] = np.nan
    # 4.2 Assign all values with NaN initially
    target_array.data = init_data 
    # 4.3 Check if the keyword "decimal_place" is supplied
    if "decimal_place" in kwargs.keys():
        decimal_place_= kwargs["decimal_place"]    
    # 5. Iterate over each AOI to extract number of target grid points
    for geometry_item in geometry_list:
        # 5.1 Obtain the geometry shape data
        geometry_shp = geometry_file.loc[geometry_file[col_name]==geometry_item,:]
        # 5.2 Determine number of grid points inside the shape file and return its lon and lat vectors
        target_lon = []
        target_lat = []
        target_points = []
        # 5.3 Iterate over each target grid point
        for coordinate in coordinates:
            grid_point = Point(coordinate) # Form the geographic coordinate point
            if any(geometry_shp.geometry.contains(grid_point, align=False)): #align=False)):
                target_lon.append(round(grid_point.x, decimal_place_)) # Append the target longitude 
                target_lat.append(round(grid_point.y, decimal_place_)) # Append the target latitude
                target_points.append(grid_point)
        # 5.4 Select target grid points for each DOC and then compute the desired stat over the area 
        zonal_array = data_array.sel({lon_name:target_lon,lat_name:target_lat}, method="nearest") 
        zonal_array_data = zonal_array.data # Access the spatial data for all grid points
        # xr.apply_ufunc(np.nanmean, data_array,                       
        #                input_core_dims=[[lat_name,lon_name]],  
        #                output_core_dims=[[]],   exclude_dims=set((lat_name,lon_name)))
        # 5.5 Compute the desired statistics
        if stat in mean_str: # Compute the mean
            zonal_stat =np.nanmean(zonal_array_data)
        elif stat in median_str: # Compute the median
            zonal_stat =np.nanmedian(zonal_array_data)
        elif stat in std_str: # Compute the standard deviation
            zonal_stat =np.nanstd(zonal_array_data)
        elif stat in cv_str: # Compute the coefficient of variation
            zonal_stat = variation(zonal_array_data,nan_policy="omit",ddof=1) # Compute the CV
        elif stat in range_str: # Compute the 90% uncertainty range
            zonal_stat = iqr(zonal_array_data,rng=(5,95),nan_policy="omit")
        elif stat in major_str: # Compute the major value of an array
            zonal_stat, count = mode(zonal_array_data, axis=None, nan_policy="omit")
        # If there is more than one most frequent values, only the smallest is returned.
        # if len(zonal_stat) > 1:
        # zonal_stat = np.nanmean(zonal_stat)
        # 5.6 Assign the zonal statistical value for all grid points that fall within a certain shape geometry
        for target_point in target_points:
            target_array.loc[{lon_name:round(target_point.x, decimal_place_),lat_name:round(target_point.y, decimal_place_)}] = float(zonal_stat)
    # 6. Return the array with new values assigned
    return target_array
###############################################################################################################################################################################################################
########################################################################## Function Blocks ####################################################################################################################
#if __name__ == "__main__":
#    