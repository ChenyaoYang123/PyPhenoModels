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
###############################################################################################################################################################################################################
# Define the disk drive letter according to the username
if getpass.getuser() == 'Clim4Vitis':
    script_drive = "H:\\"
    #shape_path = r"H:\Grapevine_model_GridBasedSimulations_study4\shapefile"
elif getpass.getuser() == 'admin':
    script_drive = "G:\\"
elif (getpass.getuser() == 'CHENYAO YANG') or (getpass.getuser() == 'cheny'):
    script_drive = "D:\\"
target_dir = r"Mega\Workspace\Study for grapevine\Study6_Multi_phenology_modelling_seasonal_forecast\script_collections" # Specific for a given study
###############################################################################################################################################################################################################
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
###############################################################################################################################################################################################################
add_script(script_drive, target_dir)
from Multi_phenology_model_classes import * # Add the multi-phenology model class script
###############################################################################################################################################################################################################
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
###############################################################################################################################################################################################################
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
###############################################################################################################################################################################################################
def load_and_concat_da(data_path, concat_dim ="time"):
    """
    Load all .nc file into xarray dataset objects and concatenate them along the concat_dim
    
    Parameter    
    ----------
    data_path: iterable, an iterable of path or path-like object that points to the target .nc files 
    concat_dim: str, the dimension to concatenate
    """
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

    # Obtain the dimension vector that should correspond to the same vectionr in the xarray object
    dim_vec = output_data.index # The output data requires that the index being the dimension looked up by the xarray
    # The saving dimension must not in the indexing dimension of xarray data array
    save_point = {key:value for key, value in point_dims.items() if key not in save_dim} 
    # Iterate over the dimensional vector and output data
    for dim_ele, data_ele in zip(dim_vec, output_data): # Here the output data and save dimension should be consistent in length
        save_point_dim = save_point.copy()
        save_point_dim.update({save_dim:dim_ele}) # Update the dict with each saving dimmensional element 
        data_arr.loc[save_point_dim] = data_ele # Save a scaler value one at a time for the xarray object
    # Not return anything because the data array object is loaded into the dictionary
###############################################################################################################################################################################################################
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
###############################################################################################################################################################################################################
def check_NaN(ser, fill_method1="quadratic",fill_method2="bfill",fill_method3="ffill", fill_extra=True):
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
def simulation_maps(file_name, data_arrays, shape_path, savepath, 
                    colormap, cbar_label, plot_dim= "space", origin=ccrs.CRS("EPSG:4326"), proj=ccrs.PlateCarree(), # ccrs.Geodetic()
                    subplot_row=10, subplot_col=3, fig_size=(6,21), specify_bound= False, extend="both", add_scalebar=False,
                    fig_format=".png",**kwargs):
    
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
        list_array = [array for array in data_arrays.sel({"time":data_arrays.time})]  # lon= data_arrays.coords[lon_name],lat=data_arrays.coords[lat_name])]        
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
        plot_axe.set_title(kwargs["plot_title"], fontdict={"fontsize":7,"fontweight":'bold'},
                                   loc="center",x=0.5,y=0.95,pad=0.05)
        # Set the y-axis label
        plot_axe.set_ylabel(kwargs["y_axis_label"], fontdict={"fontsize":7,"fontweight":'bold'}, labelpad = 0.05, loc = "center")
        # Save the plot to a local disk
        mkdir(savepath)
        fig.savefig(join(savepath,file_name+fig_format))
        plt.close(fig)
    elif plot_dim in ["corre", "CORRE", "correlation", "CORRELATION", "Correlation", "r"]:
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
        # Create a subplot axe in the figure class instance
        subplot_axe=fig.add_subplot(ax_grid,projection=proj)
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
            norm=norm_bound, cmap=colormap
            )
        # data_plot = data_array.plot(x=lon_name, y=lat_name,ax=subplot_axe, 
        #     norm=norm_bound, cmap=colormap,extend = extend,
        #     add_colorbar=False, add_labels=False
        #     ) 
        # Set the extent for each subplot map
        extent=[np.min(lonvec)-0.5,np.max(lonvec)+0.5, np.min(latvec)-0.5,np.max(latvec)+0.5] # Minor adjustment with 0.5 degree in each direction
        subplot_axe.set_extent(extent, crs=origin)
        # Add the geometry of shape file for each subplot map
        subplot_axe.add_geometries(GDF_shape.geometry, proj,
        facecolor='none', edgecolor='black')
        # Set the the yearly data as the subplot title
        if plot_dim in ["Space", "space", lon_name, lat_name]:
            subplot_name = str(data_array.time.data)
        elif plot_dim in ["corre", "CORRE", "correlation", "CORRELATION", "Correlation", "r"]:
            assert "forecast_month_var" in kwargs.keys(), "missing 'forecast_month_var' in the funcation call"
            subplot_name = kwargs["forecast_month_var"][data_array.name]
        # Set the subplot title
        subplot_axe.set_title(subplot_name, fontdict={"fontsize":9,"fontweight":'bold'},
                                   loc="right", x=0.95, y=0.95, pad=0.05)
        # Set the scale bar # Scalebar_package from mathplotlib https://pypi.org/project/matplotlib-scalebar/
        if add_scalebar is True:
            #scale_bar = AnchoredSizeBar(subplot_axe.transData, 10000, '10 km', 'upper right',frameon=False,size_vertical = 100) # ccrs.PlateCarree() unit is meter
            dx = great_circle(np.max(lonvec), np.min(latvec), np.max(lonvec)+1, np.min(latvec)) # compute the distance between two points at the latitude (Y) you wish to have the scale represented
            scale_bar = ScaleBar(dx, units="km", dimension= "si-length", length_fraction=0.25, location="upper right",sep=5,
                                 pad=0.2,label_loc="bottom", width_fraction= 0.01)
            subplot_axe.add_artist(scale_bar)
        #scale_bar = ScaleBar(13.875, units="km", dimension= "si-length", length_fraction=0.25,location="upper right",
        #                     pad=0.2,label_loc="bottom", width_fraction= 0.05
                             
                             #) #width_fraction= 0.01, label= '50 km',
                             #location= 'upper right',pad=0.2,) # The spatial resolution for each grid point is at 0.125° x 0.125° (roughly 13.875 Km x 13.875 Km)
        # Append the subplot axe to an empty list
        #axe_list.append(subplot_axe)
    # Set one colorbar for the whole subplots
    fig.subplots_adjust(right=0.85,wspace=0.05,hspace=0.05) # Make room for making a colorbar
    # Add the colorbar axe
    cbar_ax = fig.add_axes([0.9, 0.45, 0.03, 0.2])
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
    cb.ax.get_yaxis().labelpad = 30 
    # Save the plot to a local disk
    mkdir(savepath)
    fig.savefig(join(savepath,file_name+fig_format), bbox_inches="tight",pad_inches=0.05, dpi=600)
    plt.close(fig)  
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
        # None of situations below should happen;
        if any((  (lon_index == 0) and (lat_index==0), # Both lon and lat are the first elements
                 (lon_index == (len(lon_vec)-1)) and (lat_index==(len(lat_vec)-1)), # Both lon and lat are the last elements
                 (lon_index==0) and (lat_index==(len(lat_vec)-1)), # First lon combines with the last lat
                 (lat_index==0) and (lon_index == (len(lon_vec)-1)) )): # First lat combines with the last lon
            print("The boundary coordinate pairs are encountered and can not continue searching points to fill NaN")
            if len(kwargs)!=0:
                kwargs["Bug_dict"].update({"lon"+str(lon1)+"_lat"+str(lat1):'the boundary coordinate pairs are encountered and can not continue searching points to fill NaN!'}) 
            return data_ser_test
        # Here the lon_index and lat_index need to be checked if they are unchanged for a certain number of loops
    return data_ser_test # When the while loop is break, the data_ser_test is already free of any NaN values.
###############################################################################################################################################################################################################
# # Get the respective DOY of the dormancy break
# dormancy_break_DOY = dormancy_break_date.dayofyear
# # Determine if the year at which the dormancy date is predicted is in the preceding year or in the next year 
# if dormancy_break_date.year == max(two_year_tuple):
#     dormancy_break_DOY_final = (dormancy_break_DOY + 366) if calendar.isleap(dormancy_break_date.year-1) else (dormancy_break_DOY + 365)
# else:
#     dormancy_break_DOY_final = dormancy_break_DOY
# def save_data(data_array):
#     """Save single data file, uses metadata"""
    
#     #metadata = data_array.attrs
#     path = _path(metadata, new=True)
#     dataset = data_array.to_dataset(name=metadata['variable'])
#     dataset.to_netcdf(path, mode='w', format="NETCDF4", engine="netcdf4")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. Specify all user-dependent path 
# 1.1 Path for local meteorological files
root_path=r"H:\Grapevine_model_SeasonalForecast_Budburst_study6" # Main path
E_OBS_path = r"H:\E_OBS_V24" # Define the E-OBS dataset path
forecast_path = r"H:\Grapevine_model_SeasonalForecast_Budburst_study6\Forecast_datasets_ECMWF" # Define the forecast dataset path
output_path = join(root_path,"output") # Define the output path
meta_data_path = join(root_path,"metadata") # Metadata path
var_identifiers_EOBS = ['tx', 'tg', 'tn'] # Define a list of variable short names for E-OBS dataset 
target_dataset = [] # Define an empty list to collect target observational gridded datasets. Here it only referes to observational weather data
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
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 2. Collect a list of points that comprise the study region
determine_lat_lon_vector = "by_shape" # Collect data points from shape file or from forecast dataset itself
fc_times = [2, 3, 4]  # Specify the forecast time either by month or by day. But this should be case-specific
forecast_time_data = sort_FCdatasets_by_FCdate(forecast_path, fc_times) # Obtain the target forecast dict
metero_var_dict_forecast= {"Tmin":"mn2t24", "Tmax":"mx2t24", "Prec":"tp"} # Define the mteorological variable dictionary

if determine_lat_lon_vector=="by_shape":
    #GDF_shape = GDF_shape[(GDF_shape["name"]!="Vinho Verde") & (GDF_shape["name"]!="Douro")] # Sampling the geodataframe so that only Lisbon wine region is studied
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
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 3. Define essential inputs and output locations
# 3.1 Define the study years and desired column order for the weather data
begin_year = 1993 # The starting year needs to be one year before the actual starting year  
end_year = 2017 
study_period = np.arange(begin_year,end_year+1,1) # Define the study period
date_range_output= pd.date_range("{}-12-31".format(begin_year), "{}-12-31".format(end_year), freq="Y") # Date index written into output file
metero_var = ["tn","tg","tx"] # Define the meteorological variable that is going to be the input for the phenology model
metero_var_dict_OB = {"Tmin":"tn", "Tmean":"tg", "Tmax":"tx"} # Define the respective meteorological dict
target_column_order = ['day','month','Year', *metero_var, 'lat','lon'] # Pre-specified an order of columns for meteorological data
forecast_time = [2, 3, 4]  # Define the forecast month; This should be case-specific
ensemble_members = 25 # Define the number of ensemble used in the seasonal forecast
resolution = 0.1 # Define the gridding resolution
decimal_place = abs(decimal.Decimal(str(resolution)).as_tuple().exponent) # Extract number of decimal places in the input float from the defined resolution
#time_vector = pd.date_range("{}-01-01".format(begin_year), "{}-12-31".format(end_year), freq="D")
#time_vector = pd.date_range("{}-12-31".format(str(begin_year)), "{}-12-31".format(str(end_year)), freq="Y")
# 3.2 Generate vector values for each dimensional coordinate
time_vector = study_period
# Note here the shapely object will generate random floating points when extracting the longitude and latitude
lon_target_vector = np.unique([round(target_point.x, decimal_place) for target_point in target_points]) # A unique list of longitude
lat_target_vector = np.unique([round(target_point.y, decimal_place) for target_point in target_points]) # A unique list of latitude
coords_xarray = [ ("lat", lat_target_vector), ("lon", lon_target_vector)] # Create the coordinate dimensions ("time",time_vector)
# Randomly generate a 3-D dataset to pre-populate the dataset
# random_data = np.random.rand(len(lat_target_vector), len(lon_target_vector))
# random_data[:] = np.nan
# 3.3 Generate the output template for saving the results
# 3.3.1 Define the output variable names and dimensions of output coordinates
outputvars_list = ["flowering_pred","veraison_pred"]
output_template_score = xr.DataArray(coords=coords_xarray) # Create a dimensional template xarray object that is going to be used as the output structure
output_template_sm_ob = xr.DataArray(coords=coords_xarray + [("time", date_range_output)]) # Create a dimensional template xarray object object that is going to be used as the output structure
output_template_sm_fc = xr.DataArray(coords=coords_xarray + [("time", date_range_output), ("number", range(ensemble_members))]) # Create a dimensional template xarray object object that is going to be used as the output structure
# 3.3.2 Create a dictionary of output xarray object to save forecast scores
forecast_score_dict = {} # Dictionary to store forecast performance scores
for forecast_month in forecast_time:
    forecast_score_dict[str(forecast_month)+"_flo"] = output_template_score.copy(deep=True)
    forecast_score_dict[str(forecast_month)+"_ver"] = output_template_score.copy(deep=True)
# 3.3.3 Create a dictionary of output xarray object to save simulations with observed weather
forecast_ob_dict ={} # Dictionary to store simulated data with observed weather
forecast_ob_dict["ob_flo"] = output_template_sm_ob.copy(deep=True)
forecast_ob_dict["ob_ver"] = output_template_sm_ob.copy(deep=True)
# 3.3.4 Create a dictionary of output xarray object to save simulations with forecast datasets
forecast_sm_dict ={} # Dictionary to store simulated data with forecast data
for forecast_month in forecast_time:
    # forecast_sm_dict[str(forecast_month)+"ob_flo"] = output_template_sm_ob.copy(deep=True)
    # forecast_sm_dict[str(forecast_month)+"ob_ver"] = output_template_sm_ob.copy(deep=True)
    forecast_sm_dict[str(forecast_month)+"sm_flo"] = output_template_sm_fc.copy(deep=True)
    forecast_sm_dict[str(forecast_month)+"sm_ver"] = output_template_sm_fc.copy(deep=True)
# 3.4 Pre-load the required datasets into dictionary to avoid reading files each time per iteration
# 3.4.1 Pre-load the observed weather datasets, i.e. E-OBS
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
# 3.4.2 Pre-load the weather forecast datasets
forecast_data_dict = {} 
# Define the meteorological variable names
Tmin_varname = metero_var_dict_forecast["Tmin"]
Tmax_varname = metero_var_dict_forecast["Tmax"]
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
    # Attach the pre-load data into the target dict
    forecast_data_dict[forecast_month][Tmin_varname] = Tmin_da
    forecast_data_dict[forecast_month][Tmax_varname] = Tmax_da
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 4. Run the simulations throughout target regions and compute the forecast score for each point
Bug_dict = {} # Create a bug dictionary to catch the bugs during simulations
# Iterate over each target grid point
for index, target_point in enumerate(target_points):
    timer = Timer()
    timer.start()
    # Get the latitude and longitude for the point
    lon1 = round(target_point.x, decimal_place)
    lat1 = round(target_point.y, decimal_place)
    # Create point dimensional dictionaries that are used for saving simulation data
    point = {output_template_score.dims[0]:lat1, output_template_score.dims[1]:lon1} # Basic point dim that only contains lat1 and lon1
    point_time = point.copy() # Create a new point dim via copying original point dim
    point_time["time"] = date_range_output # Update the basic point dimension with extra time dimensions    
    print("Start running for point No.{} out of total {}".format(str(index+1),str(len(target_points))))  
    # Extract the daily timeseries of observed weather datasets for the specified grid point
    site_data_df  = collect_weather_data(xarray_data_dict, lon1, lat1, study_period, dataset_loaded= True, 
                                         var_identifiers = metero_var, target_column_order = target_column_order, extract_method="nearest")
    # Extract the minimum and maximum temperature series that is mainly used as reference data to fill the gap for forecast dataset
    T_min = site_data_df[metero_var_dict_OB["Tmin"]] # Ensure the input data is a series
    T_min.name ="tasmin" # Reset the Tmin series name into "tasmin"
    T_max = site_data_df[metero_var_dict_OB["Tmax"]] # Ensure the input data is a series
    T_max.name ="tasmax" # Reset the Tmax series name into "tasmax"
    # Extract the Tmean series
    T_mean = site_data_df[metero_var_dict_OB["Tmean"]] # Ensure the input data is a series
    T_mean.name ="tas" # Reset the Tmean series name into "tas"
    T_mean_ob = T_mean.loc[T_mean.index.year.isin(study_period)] # Subset the temperature data to cover the study period only
    try: # # Run the model to get the simulated flowering and veraison date based on the observed weather data
        flo_ob, ver_ob = run_models(T_mean_ob) # Perform simulations with observed weather 
        # point_coords = {key:([value] if key not in "time" else value) for key,value in point_time.items()} # The same key used from point time but with different values. Use if else in dict comprehension
        # a = xr.DataArray(np.expand_dims(flo_ob, axis=(0,1)), coords= point_coords, dims=list(forecast_ob_dict["ob_flo"].dims))
    except:
        Bug_dict.update({"lon"+str(lon1)+"_lat"+str(lat1):'Issues in simulated values with observed weather!'}) # catch the erroneous simulation values
    # Re-set the datetime index to be consistent with those in the xarray object 
    flo_ob.index = date_range_output
    ver_ob.index = date_range_output
    # Check NaN values for simulation series. If any, fill NaN for simulation series from the nearest neighbours
    if any(pd.isnull(flo_ob)):
        # Fill NaN values of simulations from those of nearest neighbours
        flo_ob = extract_nearest_neighbours(forecast_ob_dict["ob_flo"], flo_ob, lon1, lat1, Bug_dict=Bug_dict)
        # In case this still failed, append it to the Bug_dict and continue to the next iteration
        if any(pd.isnull(flo_ob)):
            Bug_dict.update({"lon"+str(lon1)+"_lat"+str(lat1):'failed to fill NaN for the flowering stage at this point'}) # catch the points that NaN values filling failed
            forecast_ob_dict["ob_flo"].loc[point_time] = -999
            continue
    if any(pd.isnull(ver_ob)):
        # Fill NaN values of simulations from those of nearest neighbours
        ver_ob = extract_nearest_neighbours(forecast_ob_dict["ob_ver"], ver_ob, lon1, lat1, Bug_dict=Bug_dict)
        # In case this still failed, append it to the Bug_dict and continue to the next iteration
        if any(pd.isnull(ver_ob)):
            Bug_dict.update({"lon"+str(lon1)+"_lat"+str(lat1):'failed to fill NaN for the veraison stage at this point'}) # catch the points that NaN values filling failed
            forecast_ob_dict["ob_ver"].loc[point_time] = -999
            continue
    # Save the timeseries of simulated data into the xarray data array objects for point with simulation series free of NaN
    save_to_dataarray(forecast_ob_dict["ob_flo"], "time", flo_ob, point_time)
    save_to_dataarray(forecast_ob_dict["ob_ver"], "time", ver_ob, point_time)
    # Iterate over the seasonal forecast datasets for each forecast month and each ensemble member (a nested for loop)
    for forecast_month in forecast_data_dict.keys():
        xr_concat_fc_data = forecast_data_dict[forecast_month] # Access the pre-loaded xarray dataset object at a given forecast date
        # Create empty list to store the Pearson correlation coefficient computed for each ensemble member at a given forecast month/date
        corre_fc_ensemble_flo = []
        corre_fc_ensemble_ver = []
        # Iterate over the ensemble memeber under a given forecast month/date
        for ens_member in range(ensemble_members):
            point_time["number"] = ens_member # Update the point dimension with extra ensemble member dimension
            # Extract the forecast datasets (with gap filled) at a specific forecast date/time/month for a given point
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
            try:
                ######################Section for running the budburst model######################
                # dormancy_out, budburst_out = run_BRIN_model(T_min, T_max, CCU_dormancy = 144.19, T0_dormancy = 213, CGDH_budburst = 823.9, 
                #                    TMBc_budburst= 25, TOBc_budburst = 0.42, Richarson_model="daily") # The parameter set is applied for TN calibrated from Luisa. L. et al. 2020 
                ######################Section for running the budburst model######################
                flo_fc, ver_fc = run_models(T_mean_fc) # Run the model to get the simulated flowering and veraison date based on the forecast weather data
            # Check if there are any NaN values in the series
            except:
                Bug_dict.update({"lon"+str(lon1)+"_lat"+str(lat1):'Issues in simulated values with forecast datasets!'}) # catch the erroneous simulation values
                #continue # Skip errorneous ensemble simulation
            # Re-set the datetime index to be consistent with those in the xarray object
            flo_fc.index = date_range_output
            ver_fc.index = date_range_output
            # Check and fill NaN for simulation series if any
            if any(pd.isnull(flo_fc)):
                flo_fc = extract_nearest_neighbours(forecast_sm_dict[str(forecast_month)+"sm_flo"], flo_fc, lon1, lat1, ens_dim=True, ens_member=ens_member, Bug_dict=Bug_dict)
                # Fill NaN values of simulations from those of nearest neighbours. But in case this failed, append it to the Bug_dict and continue to the next iteration
                if any(pd.isnull(flo_fc)):
                    Bug_dict.update({"lon"+str(lon1)+"_lat"+str(lat1):'failed to fill NaN for the flowering stage at this point for the ensemble member {}'.format(ens_member)}) # catch the points that NaN values filling failed
                    forecast_sm_dict[str(forecast_month)+"sm_flo"].loc[point_time] = -999 
                    continue
            if any(pd.isnull(ver_fc)):
                ver_fc = extract_nearest_neighbours(forecast_sm_dict[str(forecast_month)+"sm_ver"], ver_fc, lon1, lat1, ens_dim=True, ens_member=ens_member, Bug_dict=Bug_dict)
                # Fill NaN values of simulations from those of nearest neighbours. But in case this failed, append it to the Bug_dict and continue to the next iteration
                if any(pd.isnull(ver_fc)):
                    Bug_dict.update({"lon"+str(lon1)+"_lat"+str(lat1):'failed to fill NaN for the veraison stage at this point for the ensemble member {}'.format(ens_member)}) # catch the points that NaN values filling failed
                    forecast_sm_dict[str(forecast_month)+"sm_ver"].loc[point_time] = -999 
                    continue
            # Attach the simulation simulation series into target output files
            save_to_dataarray(forecast_sm_dict[str(forecast_month)+"sm_flo"], "time", flo_fc, point_time)
            save_to_dataarray(forecast_sm_dict[str(forecast_month)+"sm_ver"], "time", ver_fc, point_time)
            # Compute the correlation coefficient
            flo_corr = np.corrcoef(flo_ob, flo_fc, rowvar=False)[0][1]
            ver_corr = np.corrcoef(ver_ob, ver_fc, rowvar=False)[0][1]
            # Append the correlation coefficient into the result list
            corre_fc_ensemble_flo.append(flo_corr)
            corre_fc_ensemble_ver.append(ver_corr)
        # Compute the ensemble mean of correlation coefficient
        ens_corre_flo = np.nanmean(corre_fc_ensemble_flo)
        ens_corre_ver = np.nanmean(corre_fc_ensemble_ver)
        # Attach the computed ensemble mean of correlation coefficient into the target xarray data array object
        forecast_score_dict[str(forecast_month)+"_flo"].loc[point] = ens_corre_flo # Since here it is already a scaler-based saving, it is not necessary to use save_to_dataarray()
        forecast_score_dict[str(forecast_month)+"_ver"].loc[point] = ens_corre_ver # Since here it is already a scaler-based saving, it is not necessary to use save_to_dataarray()
    print("Finish processing for point No.{} out of total {}".format(str(index+1),str(len(target_points))))  
    timer.end()
# Write the output into the target 3-D dataset
output_path = join(output_path,"simulation_score")
output_path_sm = join(output_path,"simulation_fc")
output_path_ob = join(output_path,"simulation_ob")
# Make directories to save files
mkdir(output_path)
mkdir(output_path_sm)
mkdir(output_path_ob)
# Save the score arrays into .nc files at local disk
for key ,value in forecast_score_dict.items():
    output_da = value.to_dataset(name = key)
    # Save to disk as .nc file
    output_da.to_netcdf(join(output_path,"{}.nc".format(key)), mode='w', format="NETCDF4", engine="netcdf4")
# Save the simulation arrays with forecast datasets into .nc files at local disk
for key ,value in forecast_ob_dict.items():
    output_da = value.to_dataset(name = key)
    # Save to disk as .nc file
    output_da.to_netcdf(join(output_path_sm,"{}.nc".format(key)), mode='w', format="NETCDF4", engine="netcdf4")
# Save the simulation arrays with observed weather datasets into .nc files at local disk
for key ,value in forecast_sm_dict.items():
    output_da = value.to_dataset(name = key)
    # Save to disk as .nc file
    output_da.to_netcdf(join(output_path_sm,"{}.nc".format(key)), mode='w', format="NETCDF4", engine="netcdf4")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 5. Visualize the results
# 5.1 Visualize the results for the simulated year-to-year flowering and veraison stages using observed weather 
# Check all supported cmaps in mathplotlib
mathplot_lib_cmaps = show_cmap_list()
mathplot_lib_cmaps()
# Define the requried cmaps
cmap_dict = {"flowering_pred":discrete_cmap(6, base_cmap="YlGn"), 
            "veraison_pred" :discrete_cmap(6, base_cmap="YlGnBu")} # plt.cm.get_cmap("autumn_r")}
# Empirically define the bounds for the flowering and veraison stage
cmap_bounds =  {"flowering_pred":np.linspace(140, 220, cmap_dict["flowering_pred"].N-1), 
            "veraison_pred" :np.linspace(200, 280, cmap_dict["veraison_pred"].N-1)}
# 5.1.1 Define the essential input variables
map_proj = ccrs.PlateCarree() # Define the projection type
outputvars_list = ["flowering_pred","veraison_pred"] # Define a list of output variables
output_dict = {"flowering_pred": forecast_ob_dict["ob_flo"], 
                "veraison_pred": forecast_ob_dict["ob_ver"]} # Define the output dict
# 5.1.2 Iterate over each output variable to make the plot, i.e. flowering and veraison stage
for plot_var in outputvars_list:
    output__array = output_dict[plot_var]
    # Filter the output array
    output__array = output__array.where(~output__array.time.isin([output__array.time.data[0]]), drop=True) # Remove the first year data since it can empty
    # Access the cmap to use in the plot
    cmap_use = cmap_dict[plot_var]
    # Remember that the first and the last bin of supplied cmap will be used for ploting the colorbar extension. 
    # The bounds will determine number of color bins with extend set to "both" will add 2 extra bins, therefore the bound should anticipate number of color bins from cmap
    simulation_maps(plot_var, output__array, GDF_shape, output_path, 
                    cmap_use, "{} DOY".format(re.findall(r"\w+_pred",plot_var)[0].strip("pred_")), plot_dim= "space",
                    subplot_row=6, subplot_col=5, fig_size=(11,15), extend="both", add_scalebar=False, specify_bound=True,
                    fig_format=".png", bounds = cmap_bounds[plot_var])
# 5.2 Visualize the results for the performance score, i.e. correlation coefficient
# 5.2.1 Define the path to target .nc files that store the data to plot
forecast_flo_corre = glob.glob(join(output_path,"*flo.nc"))
forecast_ver_corre = glob.glob(join(output_path,"*ver.nc"))
# 5.2.2 Define the cmap to use in correlation coefficient plot
cmap_corre = discrete_cmap(10, base_cmap="summer_r")
# 5.2.3 Define the forecast month variable dictionary
forecast_month_var = {"2_flo": "February_fc_flowering",
                 "3_flo": "March_fc_flowering",
                 "4_flo": "April_fc_flowering",
                 "2_ver": "February_fc_veraison",
                 "3_ver": "March_fc_veraison",
                 "4_ver": "April_fc_veraison"
                 }
# 5.2.4 Collect and sort the plot datasets
forecast_list = [xr.open_dataset(data_path, mask_and_scale=True, engine = "netcdf4") for data_path in forecast_flo_corre+forecast_ver_corre] # Collect all output datasets into a single list
forecast_list.sort(key=lambda x : int(re.compile(r"\d+_").findall(list(x.data_vars.keys())[0])[0].strip("_"))) # Sort the dataset by forecast month. Sort by defauly is a inplace operation
forecast_list_sorted = [dataset[list(dataset.data_vars.keys())[0]] for dataset in forecast_list] # Gather sorted datasets
# 5.2.5 Plot the forecast performance scores
simulation_maps("correlation_fc", forecast_list_sorted, GDF_shape, output_path,
                cmap_corre, "Pearson correlation coefficient", plot_dim= "correlation",
                subplot_row=3, subplot_col=2, fig_size=(6,12), extend="neither", add_scalebar=False, specify_bound=True,
                fig_format=".png", bounds = np.linspace(0, 1, cmap_corre.N+1), forecast_month_var=forecast_month_var)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Plot using Xarray´s inner plotting functionalities
# plots = xarray_dataarray.plot.pcolormesh(x = "lon", y= "lat",
#     transform=map_proj,  # the data's projection
#     col="time", # Define the subplot dimension. Subplot over time 
#     col_wrap=3,  # set number of columns, used alongside with "col"
#     aspect = len(xarray_dataarray.coords["lon"]) / len(xarray_dataarray.coords["lat"]),  # for a sensible figsize
#     subplot_kws={"projection": map_proj} , cmap= "YlGnBu")  # the plot's projection
# grid = gridspec.GridSpec(nrows = subplot_row, ncols = subplot_col, hspace = 0.05, wspace = 0.05)
# fig = plt.figure(figsize=fig_size) # Create a figure instance class
# # We have to set the map's options on all axes
# for ax in enumerate(plots.axes.flat):
#     ax.coastlines()
#     ax.set_extent([minx-0.5, maxx+0.5, miny-0.5, maxy+0.5]) # The extent is derived from previous boundary setting crs=map_proj
#     ax.add_geometries(GDF_shape.geometry, map_proj,
#         facecolor='none', edgecolor='black')
# ncfile = join(root_path, "ECMWF-SEAS5_datasets", "T_max_01.grib")
# nc_var_identifier = "Maximum_temperature"
# lon1, lat1, time_period = target_point.x, target_point.y, study_period
# data = extract_site_data_series(ncfile,nc_var_identifier,lon1,lat1,time_period,method="nearest", _format="GRIB")
    