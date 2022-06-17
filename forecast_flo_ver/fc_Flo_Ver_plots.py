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
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 2 Define essential input path to make various kinds of plots
# 2.1 Define the main path including the output saving path
root_path= join(script_drive, "Grapevine_model_SeasonalForecast_FloVer") # Main root path for the data analysis
data_input_path = join(root_path, "output", "out_flo_ver") # Specify the main data input (input for the plot analysis) path for the analysis
output_path = join(root_path,"output") # Define the output plot path
# 2.2 Define the path to shape file
shape_path = join(script_drive, r"Mega\Workspace\SpatialModellingSRC\GIS_root\Root\shape_wine_regions") # Define the main path to target shapefile 
study_shapes = glob.glob(join(shape_path,"*.shp")) # Obtain a list of shape files used for defining the study region
study_region = "DOC_PT_final" # Define the name of file used for study region. 
study_outline = "PT_outline" # Define the name of file used for the background outline
study_region_shape = [shape for shape in study_shapes if study_region in shape][0] # Squeeze the list
study_outline_shape = [shape for shape in study_shapes if study_outline in shape][0] # Squeeze the list
proj = "epsg:4326" # ccrs.PlateCarree() # Define the target projection CRS
GDF_shape= gpd.read_file(study_region_shape).to_crs(proj)  # to_crs(proj) # Load the study region shape file into a geographic dataframe
GDF_shape_outline= gpd.read_file(study_outline_shape).to_crs(proj) # Load the outline shp as the background
DOC_wine_regions = GDF_shape["DOC_ID"] # Get a list of DOC wine regions
# 2.3 Set the fixed extent for the target map extent to be plotted
minx, miny, maxx, maxy = GDF_shape_outline.total_bounds # The extent coordinate is derived from outline shape file
extent=[minx-0.75, maxx+0.75, miny-0.75, maxy+0.75] # Minor adjustment of extent
# 2.4 Get the grid longitude and latitude vectors
grid_lons = np.arange(round(minx), round(maxx)+1, 1)
grid_lats = np.arange(round(miny), round(maxy)+1, 1)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Input code session+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Plot code session++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 3. Make the seasonal cumulative GDD plot
# 3.1 Define the GDD plot data path
GDD_save = "GDD_output"
output_path_GDD_ob = join(output_path, "out_flo_ver", GDD_save, "ob") # Define the saving path to plot the GDD data of reference
output_path_GDD_fc = join(output_path, "out_flo_ver", GDD_save, "fc")  # Define the saving path to plot the GDD data of forecast
mkdir(output_path_GDD_ob)
mkdir(output_path_GDD_fc)
# 3.2 Read the GDD output .nc files between the forecast and reference dataset
ob_GDD_files = glob.glob(join(output_path_GDD_ob,"*.nc"))
sm_GDD_files = glob.glob(join(output_path_GDD_fc,"*.nc"))
# 3.3 Define the number of decimal place to keep while loading the coordiante floating values
decimal_place = 1
# 3.4 Define the study period
begin_year = 1993 # The starting year
end_year = 2017 # The last simulation year
study_period = np.arange(begin_year,end_year+1,1) # Define the study period
leap_years = [year for year in study_period if year%4==0] # Get a list of leap years from the study period
# 3.5 Access the current settings of period duration
end_flo_time =  time_depend_GDD.__defaults__[0] # By extracting the default information from the current function
end_ver_time =  time_depend_GDD.__defaults__[1] # By extracting the default information from the current function
# 3.6 Define a list of plot types
plot_types = ["GDD_abs", "GDD_ecdf", "GDD_kde"] # All plot types will be generated
# 3.7 Iterate over paired .nc files to make the GDD plot
for plot_type in plot_types:
    for ob_gdd, fc_gdd in zip(ob_GDD_files, sm_GDD_files): # Iterate over each paird gdd file  
        # Load the datasets as the xarray objects
        ob_gdd_xr = xr.open_dataset(ob_gdd, mask_and_scale=True, engine = "netcdf4", decode_times =True)
        fc_gdd_xr = xr.open_dataset(fc_gdd, mask_and_scale=True, engine = "netcdf4", decode_times =True)
        # Check if the variable names are identical between the two xarray objects
        if not all(var1==var2 for var1, var2 in zip(ob_gdd_xr.data_vars, fc_gdd_xr.data_vars)):
            raise ValueError("The observational and forecast datasets have different variable names")
        else:
        # Adopt the variable name from the first xarray object
            var_name = [var1 for var1, var2 in zip(ob_gdd_xr.data_vars, fc_gdd_xr.data_vars) if ("gdd" in var1) & ("gdd" in var2)][0] 
        if "5" in var_name: # Skip the month 5 initialization forecast # Case-specific operation
            continue
        # Obtain the forecast dataset initialization month
        fc_init_month_int = int("".join(re.findall(r"\d?",var_name)))
        fc_init_month = calendar.month_abbr[fc_init_month_int]
        # Obtain the variable name for the current iteration loop
        output_varname = "".join(re.findall(r"_\w+",var_name)).strip("_")
        if "flo" in output_varname: # The forecast is always initialized on the first day of the month
            seasonal_date_range = pd.date_range(datetime(leap_years[0], fc_init_month_int, 1), datetime(leap_years[0], end_flo_time.month, end_flo_time.day), freq="D").strftime('%m-%d')
        elif "ver" in output_varname: # The forecast is always initialized on the first day of the month
            seasonal_date_range = pd.date_range(datetime(leap_years[0], fc_init_month_int, 1), datetime(leap_years[0], end_ver_time.month, end_ver_time.day), freq="D").strftime('%m-%d')
        # Access the underlying data array of the xarray dataset objects
        ob_gdd_xr_arr = ob_gdd_xr[var_name]
        fc_gdd_xr_arr = fc_gdd_xr[var_name]
        # Get the lon and lat names from the xarray object
        lon_name_ob, lat_name_ob = get_latlon_names(ob_gdd_xr_arr)
        lon_name_fc, lat_name_fc = get_latlon_names(fc_gdd_xr_arr)
        if lon_name_ob==lon_name_fc:
            lon_name = lon_name_ob
        if lat_name_ob==lat_name_fc:
            lat_name = lat_name_ob
        # Compute the lon and lat vectors 
        lon_vector = ob_gdd_xr_arr[lon_name].data if np.array_equal(ob_gdd_xr_arr[lon_name],fc_gdd_xr_arr[lon_name],equal_nan=True) else np.nan
        lat_vector = ob_gdd_xr_arr[lat_name].data if np.array_equal(ob_gdd_xr_arr[lat_name],fc_gdd_xr_arr[lat_name],equal_nan=True) else np.nan
        # Form the list of coordinates to be studied
        coordinates = list(product(lon_vector, lat_vector)) 
        # Define number of cols and rows used in the plot
        subplot_row, subplot_col = int((len(DOC_wine_regions)+3)/5),  5
        # Set the grid system and figure size, depending on the plot type
        if plot_type == "GDD_abs": 
            hspace, wspace = 0.05, 0.25
        elif plot_type == "GDD_kde": 
            hspace, wspace = 0.25, 0.3
        elif plot_type == "GDD_ecdf":
            hspace, wspace = 0.25, 0.05
        grid = gridspec.GridSpec(nrows = subplot_row, ncols = subplot_col, hspace = hspace, wspace = wspace) # Specify the grid system used in the plot
        if plot_type == "GDD_abs":
            fig = plt.figure(figsize=(subplot_col*2.4, subplot_row*2.4)) # Create a figure instance class
        else:
            fig = plt.figure(figsize=(subplot_col*2, subplot_row*2.4)) # Create a figure instance class
        # Append to this empty list a number of grid specifications, depending on the number of columns and rows specified
        axis_list=[]
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
        subplots = [] # Append to an empty list all created subplot axes
        # Iterate over each geometry entry (wine region) in the shape file to make the plot
        for index, (DOC_wine_region,axe_grid) in enumerate(zip(DOC_wine_regions, axis_list)):
            # Create the subplot axe with a given axe_grid specification
            subplot_axe=fig.add_subplot(axe_grid)
            # Select a given DOC wine region
            DOC_shape_region = GDF_shape.loc[GDF_shape["DOC_ID"]==DOC_wine_region,:]
            # Determine number of grid points inside each geometry (wine region) and return its lon and lat vectors
            target_lon = []
            target_lat = []
            target_points = []
            for coordinate in coordinates: # Check every single points that possibly exist inside the .nc file
                grid_point = Point(coordinate) # Form the shaply grid point
                if any(DOC_shape_region.geometry.contains(grid_point, align=False)): # Check if a given shape contains a testing grid point
                    target_lon.append(round(grid_point.x, decimal_place)) # Append the target longitude 
                    target_lat.append(round(grid_point.y, decimal_place)) # Append the target latitude
                    target_points.append(grid_point)
            # Print number of grid points that fall within the study shape geometry (wine region)
            print("The DOC ID {} has total {} points".format(str(index+1), str(len(target_points))))
            # Select target grid points for each shape geometry and then compute the spatial average over times      
            GDD_ob_DOC_mean = ob_gdd_xr_arr.sel({lon_name:target_lon,lat_name:target_lat}, method="nearest").mean(dim=[lon_name,lat_name], skipna=True, keep_attrs= True)
            GDD_fc_DOC_mean = fc_gdd_xr_arr.sel({lon_name:target_lon,lat_name:target_lat}, method="nearest").mean(dim=[lon_name,lat_name], skipna=True, keep_attrs= True)
            # Obtain the seasonal date information for the observed and forecast datasets
            GDD_ob_DOC_date = pd.to_datetime(GDD_ob_DOC_mean.time.data).strftime('%m-%d')
            GDD_fc_DOC_date = pd.to_datetime(GDD_fc_DOC_mean.time.data).strftime('%m-%d')
            # Filter datasets to ensure GDD data is within the range
            GDD_ob_DOC_mean = GDD_ob_DOC_mean.where(GDD_ob_DOC_mean.time.loc[GDD_ob_DOC_date.isin(seasonal_date_range)], drop=True)
            GDD_fc_DOC_mean = GDD_fc_DOC_mean.where(GDD_fc_DOC_mean.time.loc[GDD_fc_DOC_date.isin(seasonal_date_range)], drop=True)
            # Groupby day of year for both series
            GDD_ob_DOC_mean_groupby = GDD_ob_DOC_mean.groupby(GDD_ob_DOC_mean.time.dt.dayofyear)
            GDD_fc_DOC_mean_groupby = GDD_fc_DOC_mean.groupby(GDD_fc_DOC_mean.time.dt.dayofyear)
            # Compute the 5th percentile, median and 95th percentile values over study years for observed GDD values
            GDD_ob_DOC_5th = GDD_ob_DOC_mean_groupby.quantile(0.05, skipna=True, keep_attrs= True)[:-1] # Note it is essential to remove the last value, since the days are not even with leap years
            GDD_ob_DOC_median = GDD_ob_DOC_mean_groupby.quantile(0.5, skipna=True, keep_attrs= True)[:-1]
            GDD_ob_DOC_95th = GDD_ob_DOC_mean_groupby.quantile(0.95, skipna=True, keep_attrs= True)[:-1]
            # Compute the 5th percentile, median and 95th percentile values over study years for forecast GDD values
            GDD_fc_DOC_5th = GDD_fc_DOC_mean_groupby.quantile(0.05, skipna=True, keep_attrs= True)[:-1] # Note it is essential to remove the last value, since the days are not even with leap years
            GDD_fc_DOC_median = GDD_fc_DOC_mean_groupby.quantile(0.5, skipna=True, keep_attrs= True)[:-1]
            GDD_fc_DOC_95th = GDD_fc_DOC_mean_groupby.quantile(0.95, skipna=True, keep_attrs= True)[:-1]
            # For the plot type of absolute values of GDD:
            if plot_type == "GDD_abs":
                # Obtain the x-axis datetime information for the plot
                GDD_ob_DOC_mean_begin_year_date = GDD_ob_DOC_mean.where(GDD_ob_DOC_mean.time.dt.year==begin_year, drop=True)# Selet the first study year, which by chance is not a leaping year
                # Collect all dates information into a list
                date_list = [calendar.month_abbr[int(month)] + "_" + str(day) for month, day in zip(GDD_ob_DOC_mean_begin_year_date.time.dt.month.data, GDD_ob_DOC_mean_begin_year_date.time.dt.day.data)]
                # Filter out some dates in the supplied date_List
                #subplot_axe.set_xticklabels(date_list_x_axis)
                # Make the line and colorband plots for the reference GDD data 
                subplot_axe.plot(date_list, GDD_ob_DOC_median.data, color='blue', linewidth=1, linestyle='-')
                # Fill color band between the 95th percentile and 5th percentile
                subplot_axe.fill_between(date_list, GDD_ob_DOC_95th.data, GDD_ob_DOC_5th.data, # Provided the x position, ymax, ymin positions to fill 
                      facecolor="blue", # The fill color
                      color= "blue",   # The outline color
                      edgecolors = "blue", # The line edge color
                      linewidth =0.5,
                      alpha=0.2) 
                # Make the line and colorband plots for the forecast GDD data 
                subplot_axe.plot(date_list, GDD_fc_DOC_median.data, color='red', linewidth=1, linestyle='-')
                # Fill color band between the 95th percentile and 5th percentile
                subplot_axe.fill_between(date_list, GDD_fc_DOC_95th.data, GDD_fc_DOC_5th.data, # Provided the x position, ymax, ymin positions to fill 
                      facecolor="red", # The fill color
                      color= "red",   # The outline color
                      edgecolors = "red", # The line edge color
                      linewidth =0.5,
                      alpha=0.4)
                # Set the x-axis ticks
                x_ticks = [date for i, date in enumerate(date_list) if i%15==0] # Spacing with every 15 days
                y_max =  get_nearest_hundreds(max(np.hstack([GDD_ob_DOC_95th.data, GDD_fc_DOC_95th.data])),100) # Get the nearest hundred value for the y-axis
                y_ticks = np.arange(0, y_max+400, 400) # Set the y-axis ticks for every 400 GDD space (fixed)
                subplot_axe.set_xticks(x_ticks) # Set the x-axis ticks
                subplot_axe.set_xticklabels([str(tick) for tick in x_ticks], rotation=90, fontsize =5) # Set the x-axis tick labels
                subplot_axe.set_yticks(y_ticks) # Set the y-axis ticks
                subplot_axe.set_yticklabels([str(tick) for tick in y_ticks],  fontsize =5) # Set the y-axis tick labels
                ##  Perform conventional statistical analysis between (regional mean) reference and forecast datasets over all years
                # First metric: R2, the coefficient of determination
                corr, corr_p = pearsonr(GDD_ob_DOC_mean.data, GDD_fc_DOC_mean.data)
                R2 = round(corr**2, 2) # Express the value as 2 decimal digit floating number
                # Second metric: MAE, the mean absolute errors
                MAE = mean_absolute_error(GDD_ob_DOC_mean.data, GDD_fc_DOC_mean.data)
                MAE = round(MAE) # Express the value as integer
                # Third metric: RMSE, the root mean squared errors
                RMSE = np.sqrt(mean_squared_error(GDD_ob_DOC_mean.data, GDD_fc_DOC_mean.data))
                RMSE = round(RMSE) # Express the value as integer
                # Fourth metric: nRMSE, the normalized root mean squared errors
                nRMSE = RMSE / (max(GDD_ob_DOC_mean.data)-min(GDD_ob_DOC_mean.data)) # The nRMSE is to weigh the RMSE over the range (max- min) of data
                nRMSE = round(nRMSE*100, 1) # Express the value as 1 decimal digit floating number
                # Label the above computed statistical values
                subplot_axe.text(0.82, 0.2, "R2={}".format(str(R2)), fontsize=5, style="italic", color='black', horizontalalignment='center', transform = subplot_axe.transAxes)
                subplot_axe.text(0.82, 0.15,"MAE={}".format(str(MAE)), fontsize=5, style="italic", color='black', horizontalalignment='center', transform = subplot_axe.transAxes)
                subplot_axe.text(0.8, 0.1, "RMSE={}".format(str(RMSE)), fontsize=5, style="italic", color='black', horizontalalignment='center', transform = subplot_axe.transAxes)
                subplot_axe.text(0.8, 0.05,"nRMSE={}%".format(str(nRMSE)), fontsize=5, style="italic", color='black', horizontalalignment='center', transform = subplot_axe.transAxes)
                # Perform the two-sample Kolmogorov-Smirnov test for the goodness of fit between two sample distributions 
                ks_statistics, p_value = ks_2samp(GDD_ob_DOC_mean, GDD_fc_DOC_mean, alternative='two-sided', mode= "auto")
                # Statistic significant analysis
                if p_value<= 0.05: # Significance at 5% level
                    # Set the background color as light grey in case of a statistic significance result
                    subplot_axe.set_facecolor("lightgrey") 
             # For the plot type of kernel density estimations of GDD:
            elif plot_type == "GDD_kde":
                # Kernel density estimations (kde) of observated datasets for 5th, 50th (median) and 95th percentiles
                kde_ob_5th = gaussian_kde(GDD_ob_DOC_5th)
                kde_ob_median = gaussian_kde(GDD_ob_DOC_median)
                kde_ob_95th = gaussian_kde(GDD_ob_DOC_95th)
                # Kernel density estimations (kde) of forecast datasets for 5th, 50th (median) and 95th percentiles
                kde_fc_5th = gaussian_kde(GDD_fc_DOC_5th)
                kde_fc_median = gaussian_kde(GDD_fc_DOC_median)
                kde_fc_95th = gaussian_kde(GDD_fc_DOC_95th)
                # Make the kdf plots
                kdf_plot(subplot_axe, kde_ob_median.dataset[0], kde_ob_5th, kde_ob_median, kde_ob_95th, color="blue", fill_color= "blue", fill_linewidth=0.5, alpha=0.15, linewidth=1, fill_regions=True, prob_area=False)
                kdf_plot(subplot_axe, kde_fc_median.dataset[0], kde_fc_5th, kde_fc_median, kde_fc_95th, color="red",  fill_color="red", fill_linewidth=0.5, alpha=0.3, linewidth=1, fill_regions=True, prob_area=False)
                # Set the x-axis ticks for each subplot
                ob_x_max =  get_nearest_hundreds(max(np.hstack([kde_ob_95th.dataset[0],kde_fc_95th.dataset[0]])), 100) # Get the nearest hundredth value for the maximum value of x-axis
                # Set the x-axis ticks for every 400 GDD (fixed)
                x_ticks = np.arange(0, ob_x_max+400, 400) 
                subplot_axe.set_xticks(x_ticks) # Set the x-axis ticks
                subplot_axe.set_xticklabels([str(tick) for tick in x_ticks], rotation=90, fontsize =5) # Set the x-axis tick labels
                # Perform the two-sample Kolmogorov-Smirnov test for goodness of fit between two sample distributions 
                ks_statistics, p_value = ks_2samp(GDD_ob_DOC_mean, GDD_fc_DOC_mean, alternative='two-sided', mode= "auto")
                # Add the statistical test labels
                if p_value<= 0.05: # Significance at 5% level
                    # Set the background color as light grey in case of a statistic significance result
                    subplot_axe.set_facecolor("lightgrey") 
            # For the plot type of empirical cumulative distribution function of GDD:
            elif plot_type == "GDD_ecdf":
                # Altenative option
                ########################## 1-D smoothing spline fit to a given set of data points##########################
                #bin_weights= np.ones_like(GDD_fc_DOC_median) / len(GDD_fc_DOC_median)
                # hist, bin_edges  = np.histogram(GDD_fc_DOC_median,bins = round(len(GDD_fc_DOC_median)/10), density=True)                       
                # x = bin_edges[:-1] + (bin_edges[1] - bin_edges[0])/2   # convert bin edges to centers
                # f = UnivariateSpline(x, hist) # s=round(len(GDD_fc_DOC_median)/10))
                # subplot_axe.plot(x, f(x))
                # #subplot_axe.show()
                # # bin_weights= np.ones_like(GDD_ob_DOC_median) / len(GDD_ob_DOC_median)
                # hist, bin_edges  = np.histogram(GDD_ob_DOC_median,bins = round(len(GDD_ob_DOC_median)/10), density=True)                       
                # x = bin_edges[:-1] + (bin_edges[1] - bin_edges[0])/2   # convert bin edges to centers
                # f = UnivariateSpline(x, hist) # s=round(len(GDD_ob_DOC_median)/10))
                # subplot_axe.plot(x, f(x))
                ########################## 1-D smoothing spline fit to a given set of data points##########################
                # Fit an empirical cumulative distribution function (ECDF) for the reference datasets
                ecdf_ob_5th = ECDF(GDD_ob_DOC_5th)
                ecdf_ob_median = ECDF(GDD_ob_DOC_median)
                ecdf_ob_95th = ECDF(GDD_ob_DOC_95th)
                # Fit an empirical cumulative distribution function (ECDF) for the forecast datasets
                ecdf_fc_5th = ECDF(GDD_fc_DOC_5th)
                ecdf_fc_median = ECDF(GDD_fc_DOC_median)
                ecdf_fc_95th = ECDF(GDD_fc_DOC_95th)
                # Plot the ECDF for the reference dataset
                subplot_axe.plot(ecdf_ob_5th.x, ecdf_ob_5th.y, color='blue', linewidth=0.8, linestyle=':')
                subplot_axe.plot(ecdf_ob_median.x, ecdf_ob_median.y, color='blue', linewidth=1.2, linestyle='-')
                subplot_axe.plot(ecdf_ob_95th.x, ecdf_ob_95th.y, color='blue', linewidth=0.8, linestyle=':')
                # subplot_axe.fill_between(ecdf_ob_median.x, ecdf_ob_5th.y, ecdf_ob_95th.y, # Provided the x position, ymax, ymin positions to fill 
                #       facecolor="hotpink", # The fill color
                #       color= "hotpink",   # The outline color
                #       edgecolors = "hotpink", # The line edge color
                #       linewidth =0.5)
                #       alpha=0.4)   
                # Plot the ECDF for the forecast dataset
                subplot_axe.plot(ecdf_fc_5th.x, ecdf_fc_5th.y, color='red', linewidth=0.8, linestyle=':')
                subplot_axe.plot(ecdf_fc_median.x, ecdf_fc_median.y, color='red', linewidth=1.2, linestyle='-')
                subplot_axe.plot(ecdf_fc_95th.x, ecdf_fc_95th.y, color='red', linewidth=0.8, linestyle=':')
                # subplot_axe.fill_between(ecdf_fc_median.x, ecdf_fc_95th.y, ecdf_fc_5th.y, # Provided the x position, ymax, ymin positions to fill 
                #      facecolor="deepskyblue", # The fill color
                #      color= "deepskyblue",   # The outline color
                #      edgecolors = "deepskyblue", # The line edge color
                #      linewidth =0.5)
                # Set the x-axis ticks
                x_max =  get_nearest_hundreds(max(np.hstack([ecdf_ob_95th.x, ecdf_fc_95th.x])), 100) # Get the nearest hundred value for the x-axis
                x_ticks = np.arange(0, x_max+400, 400) # Set the x-axis ticks for every 400 GDD                                    
                subplot_axe.set_xticks(x_ticks) # Set the x-axis ticks
                subplot_axe.set_xticklabels([str(tick) for tick in x_ticks], rotation=90, fontsize =5) # Set the x-axis tick labels
                y_ticks = np.arange(0, 1+0.1, 0.1) # Set the y-axis ticks with fixed 0.1 spacing, as ECDF always ranges from 0 to 1                
                subplot_axe.set_yticks(y_ticks) # Set the y-axis ticks
                subplot_axe.set_yticklabels([str(round(tick,1)) for tick in y_ticks], fontsize =5) # Set the y-axis tick labels
                # Perform the two-sample Kolmogorov-Smirnov test for goodness of fit between two sample distributions 
                ks_statistics, p_value = ks_2samp(GDD_ob_DOC_mean, GDD_fc_DOC_mean, alternative='two-sided', mode= "auto")
                # Add the statistical test labels
                if p_value<= 0.05: # Significance at 5% level
                    # Set the background color as light grey in case of a statistic significance result
                    subplot_axe.set_facecolor("lightgrey") 
            ## Common axe layout option
            # Set the boundary values for the axis
            subplot_axe.set_xbound(0, max(subplot_axe.get_xbound())) # Set the x-axis bound
            subplot_axe.set_ybound(0, max(subplot_axe.get_ybound())) # Set the y-axis bound
            # Set the x-axis and y-axis tick parameters
            subplot_axe.tick_params(axis='x',length=1, labelsize=5, pad=3)
            subplot_axe.tick_params(axis='y',length=1, labelsize=5, pad=3)
            # Set the subplot title
            subplot_axe.set_title("Wine_region" + str(index+1), fontdict={"fontsize":6}, loc="center",  y = 0.85)
            # Append each modified subplot axe into the target empty list
            subplots.append(subplot_axe)
        if plot_type == "GDD_abs": # In case of "GDD_abs" plot type:
            # Get shared x-axis among certain axes
            for i, subplot_axe_obj in enumerate(subplots):
                if i < (len(subplots) - 5): # Only show the last 5 axe x-tick labels
                    plt.setp(subplot_axe_obj.get_xticklabels(), visible=False) # Disable the selected axe x-ticks
                else:
                    plt.setp(subplot_axe_obj.get_xticklabels(), visible=True) # Enable the selected axe x-ticks
        elif plot_type == "GDD_ecdf": # In case of "GDD_ecdf" plot type:
           # Get shared y-axis among certain axes
           for i, subplot_axe_obj in enumerate(subplots):
               if (i % subplot_col) == 0: # A tricky way to share the y-axis for each row
                   plt.setp(subplot_axe_obj.get_yticklabels(), visible=True) # Enable the selected axe y-ticks
               else:
                   plt.setp(subplot_axe_obj.get_yticklabels(), visible=False) # Disable the selected axe y-ticks
        #axis_list[0].get_shared_x_axes().join(axis_list[0], *axis_list[1:])
        # Get the forecast initialization month str
        fc_init_month = calendar.month_abbr[int("".join(re.findall(r"\d?",var_name)))] 
        # Save the plot into local disk as a file with specified format per plot type, variety-stage, initialization month
        fig.savefig(join(output_path, "out_flo_ver", GDD_save,"{0}_{1}_{2}.pdf".format(str(plot_type), str(output_varname), str(fc_init_month))), dpi=600, bbox_inches="tight")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 4 Plot for the year-to-year flowering and veraison DOY for two different varieties
# 4.1 Check all supported cmaps in mathplotlib
mathplot_lib_cmaps = show_cmap_list()
mathplot_lib_cmaps()
# 4.2 Define the map projection CRS
map_proj = ccrs.PlateCarree()
# 4.3 Define the phenology stages to be plotted
outputvars_list = ["flowering_pred","veraison_pred"]
# 4.4 Define the cmap dict to be plotted
cmap_dict = {"flowering_pred":discrete_cmap(6, base_cmap="YlGn"), 
             "veraison_pred" :discrete_cmap(6, base_cmap="YlGnBu")} # plt.cm.get_cmap("autumn_r")}
# 4.5 Empirically define the bounding values for the flowering and veraison stage
cmap_bounds =  {"flowering_pred":np.linspace(140, 220, cmap_dict["flowering_pred"].N-1), 
                "veraison_pred" :np.linspace(200, 280, cmap_dict["veraison_pred"].N-1)}
# 4.6 Define the name abb for the studied varieties
study_vars = ["TF","TN"]
# 4.7 Iterate over each variety to make the flowering and veraison phenology DOY plot
for study_var in study_vars:
    # Define the reference phenology DOY path
    var_path_OB = join(data_input_path,  study_var, "simulation_ob")
    # Obtain a list of .nc files for the referenced flowering and veraison phenology simulations (DOY)
    OB_nc_flo = [ncfile for ncfile in glob.glob(join(var_path_OB,"*.nc")) if "flo." in ncfile][0]
    OB_nc_ver = [ncfile for ncfile in glob.glob(join(var_path_OB,"*.nc")) if "ver." in ncfile][0]
    # Read the data into the xarray objects
    OB_nc_flo_xr = xr.open_dataset(OB_nc_flo, mask_and_scale=True, engine = "netcdf4", decode_times =True)
    OB_nc_ver_xr = xr.open_dataset(OB_nc_ver, mask_and_scale=True, engine = "netcdf4", decode_times =True)
    # Gather the xarray dataset objects into a target output dictionary
    output_dict = {"flowering_pred": OB_nc_flo_xr["ob_flo"], 
                    "veraison_pred": OB_nc_ver_xr["ob_ver"]}
    # Set the output map saving path
    save_path = join(output_path, "out_flo_ver",study_var)
    # Iterate over each target phenology stage to make the spatial map plot
    for plot_var in outputvars_list:
        # Load the reference simulations into xarray object
        output__array = output_dict[plot_var]
        # Write CF standard attributes (including the CRS information) into target data array
        output__array = write_CF_attrs(output__array)
        # Clip the data array based on the supplied geometry shape
        clipped_array = output__array.rio.clip(GDF_shape.geometry, GDF_shape.crs, all_touched=False, drop=True, invert=False, from_disk = False)
        # Remove the first year
        #output__array = output__array.where(~output__array.time.isin([output__array.time.data[0]]), drop=True) # Remove the first year data since it can empty
        # Access the cmap to use in the plot
        cmap_use = cmap_dict[plot_var]
        # Get number of subplots depending on the study period years
        subplots = len(clipped_array.time)
        # Get number of desired columns and rows used in the spatial map plot
        cols = int(arrange_layout(subplots))
        rows = int(subplots/cols)
        # Remember that the first and the last bin of supplied cmap will be used for ploting the colorbar extension for extend = both. 
        # The bounds will determine number of color bins with extend set to "both" will add 2 extra bins, therefore the bound should anticipate number of color bins from cmap
        simulation_maps(plot_var, clipped_array, GDF_shape, save_path, 
                        cmap_use, "{} DOY".format(re.findall(r"\w+_pred",plot_var)[0].strip("_pred")), plot_dim= "space",
                        subplot_row = cols, subplot_col = rows, fig_size= (cols*1.5, rows*2.4), extend="both", add_scalebar=True, specify_bound=True, dpi=600,
                        fig_format=".png", bounds = cmap_bounds[plot_var], grid_lon= list(grid_lons), grid_lat=list(grid_lats), subplot_title_font_size=5, outline = GDF_shape_outline,
                        label_features = True, label_features_font_size=1)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 5 Plot for the forecast skill scores (r, MAE and RMSE)
## !!! Note the plot is per variety specific
# 5.1 Define some essential input path and a list of computed stat metrics
study_var = "TN" # Define the name abb for the studied variety.
output_path = join(root_path,"output") # Define the output path
save_path = join(output_path, "out_flo_ver", study_var)# Access the output path of a given variety
score_stats = ["r", "MAE", "RMSE"] # Define a list of score metrics to be plotted 
# 5.2 Define the boundary values for each stat plot
bound_r = np.linspace(0, 1, 6) # Define the bounding range for the correlation coefficient
bound_MAE =   np.linspace(0, 20, 6) # Define the bounding range for the MAE
bound_RMSE = np.linspace(0, 20, 6) # Define the bounding range for the RMSE
# 5.3 Define the bounding dict accordingly
bound_dict ={"r":bound_r,
             "MAE":bound_MAE,
             "RMSE":bound_RMSE
             }
# 5.4 Create a cmap for plotting the spatial distribution of r, MAE and RMSE
cmap_errors = modify_cmap(discrete_cmap(8, base_cmap="PiYG_r"),select_bins=[0,3,7],remove_bins=True)
cmap_dict = {"r": discrete_cmap(5, base_cmap="summer_r"),
             "MAE":cmap_errors,
            "RMSE":cmap_errors
             } #
# 5.5 Define the forecast month variable dictionary
forecast_month_var = {"2_flo": "February_fc_flowering",
                          "3_flo": "March_fc_flowering",
                          "4_flo": "April_fc_flowering",
                          "5_flo": "May_fc_flowering",
                          "2_ver": "February_fc_veraison",
                          "3_ver": "March_fc_veraison",
                          "4_ver": "April_fc_veraison",
                          "5_ver": "May_fc_veraison"
                         }
# 5.6 Iterate over each score metric to make the respective spatial map plot
for score_stat in score_stats:
    # Define the path to target .nc files that store the data to plot
    forecast_flo_score = glob.glob(join(save_path, "simulation_score",score_stat,"*flo.nc"))
    forecast_ver_score = glob.glob(join(save_path, "simulation_score",score_stat,"*ver.nc"))
    # Define the cmap to use in the spatial map plot
    cmap_plot = cmap_dict[score_stat]
    # Collect and sort the plot datasets
    forecast_list = [xr.open_dataset(data_path, mask_and_scale=True, engine = "netcdf4") for data_path in forecast_flo_score+forecast_ver_score] # Collect all output datasets into a single list
    forecast_list_sorted = []
    # Collect the underlying data array objects into a list
    for fc_data in forecast_list:
        # Collect the variable name
        data_var =  [var_name for var_name in list(fc_data.data_vars.keys()) if "CRS" not in var_name][0]
        data_array = fc_data[data_var] # Access the underlying data array
        forecast_list_sorted.append(data_array) # Append the data array into target empty list
    # Sort the dataset by forecast month. Sort by defauly is an inplace operation
    forecast_list_sorted.sort(key=lambda x: int(re.compile(r"\d+").findall(x.name)[0]) ) 
    # forecast_list_sorted = [dataset[list(dataset.data_vars.keys())[0]] for dataset in forecast_list] # Gather sorted datasets
    # Clip each data array and append it to a target empty list
    forecast_list_sorted_plot = []
    for fc_da in forecast_list_sorted:
        fc_da_cf_attr = write_CF_attrs(fc_da)
        # Clip the data array based on the supplied geometry shape
        clipped_fc_da = fc_da_cf_attr.rio.clip(GDF_shape.geometry, GDF_shape.crs, all_touched=False, drop=True, invert=False, from_disk = False)
        # Plotting the zonal statistics for a given region
        clipped_fc_da_zonal_stat = zonal_statistics(clipped_fc_da, GDF_shape, col_name="DOC_ID", decimal_place=1)
        # Append the clipped data array into target list
        forecast_list_sorted_plot.append(clipped_fc_da_zonal_stat)
    # Get the bounds of each score    
    data_bounds = bound_dict[score_stat]
    # Define the cbar label 
    if score_stat != "r":
        cbar_label = score_stat+" (days)"
    else:
        cbar_label = score_stat
    # Make the respective map plot
    simulation_maps("{}_fc".format(score_stat), forecast_list_sorted_plot, GDF_shape, save_path,
                    cmap_plot, cbar_label, plot_dim= "scores",
                    subplot_row=4, subplot_col=2, fig_size=(5,12), extend="neither", add_scalebar=True, specify_bound=True,
                    fig_format=".png", bounds = data_bounds, forecast_month_var=forecast_month_var,
                    grid_lon= list(grid_lons), grid_lat=list(grid_lats), subplot_title_font_size=6, outline = GDF_shape_outline)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 6. Make the ensemble forecast probability spatial map plot (possibly compute the BS or other similar scores)
# 6.1 Specify the forecast and reference data input path of the two different varieties
TF_path = join(data_input_path, "TF")
TN_path = join(data_input_path, "TN")
TF_path_SM = join(TF_path,  "simulation_fc")
TF_path_OB = join(TF_path,  "simulation_ob")
TN_path_SM = join(TN_path,  "simulation_fc")
TN_path_OB = join(TN_path,  "simulation_ob")
# 6.2 Obtain a list of .nc files of the two varieties for both reference and forecast datasets 
TF_SM_nc = glob.glob(join(TF_path_SM,"*.nc"))
TF_OB_nc = glob.glob(join(TF_path_OB,"*.nc"))
TN_SM_nc = glob.glob(join(TN_path_SM,"*.nc"))
TN_OB_nc = glob.glob(join(TN_path_OB,"*.nc"))
# 6.3 Define two essential dictionaries used for data output analysis
fc_names_dict = {"2sm": "Feb_1st",
                 "3sm": "Mar_1st",
                 "4sm": "Apr_1st"}
var_data = {"TF":{"OB":TF_OB_nc, "SM":TF_SM_nc},
            "TN":{"OB":TN_OB_nc, "SM":TN_SM_nc},
            }
#xr.apply_ufunc(tercile_var_xr, var_OB_nc_xr_arr) #input_core_dims =[["time"]])# input_core_dims =[["time"]])
decimal_place = 1
# 6.4 Iterate over the datasetes to explore the inter-annual variability features (iterations should be per variety basis)
BS_score_dict = {}        
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
        # Obtain the studied variable name
        var_name = "flo" if "_flo" in OB_var_name else "ver" 
        # Access the underlying xarray dataarray object
        var_OB_arr = var_OB_nc_xr[OB_var_name]
        # Get the lon and lat names from the xarray object
        lon_name, lat_name = get_latlon_names(var_OB_arr)
        # Get the tercile grouped array
        var_OB_arr_terciles = categorize_array(var_OB_arr)
        # Create multiple empty lists
        early_pheno_years=[]
        normal_pheno_years=[]
        late_pheno_years=[]
        for year in var_OB_arr_terciles.time.data: # Iterate over study years
            var_OB_arr_tercile_yearly_data = var_OB_arr_terciles.sel({"time":year}).data.flatten()
            #var_OB_arr_tercile_yearly_data = var_OB_arr_tercile_yearly_data[~np.isnan(var_OB_arr_tercile_yearly_data)]
            # Compute the most frequent value in the yearly array data
            major_tercile = pd.Series(var_OB_arr_tercile_yearly_data).mode(dropna=True)[0]
            if major_tercile==1: # Assign 1 for early phenology class
                early_pheno_years.append(int(pd.to_datetime(year).year))
            elif major_tercile==2: # Assign 2 for normal phenology class
                normal_pheno_years.append(int(pd.to_datetime(year).year))
            elif major_tercile==3: # Assign 3 for late phenology class
                late_pheno_years.append(int(pd.to_datetime(year).year))
        # Obtain the lon and lat vectors from the supplied OB data array    
        lon_target_vector = np.unique([round(lon_val, decimal_place) for lon_val in var_OB_arr.coords[lon_name].data ]) # A unique list of longitude
        lat_target_vector = np.unique([round(lat_val, decimal_place) for lat_val in var_OB_arr.coords[lat_name].data ]) # A unique list of latitude
        coords_xarray = [ (lat_name, lat_target_vector), (lon_name, lon_target_vector)] # Create the coordinate dimensions ("time",time_vector)
        # Create the current dict key for the dict BS_score_dict
        if variety_name + "_" + var_name not in BS_score_dict.keys():
            BS_score_dict[variety_name + "_" + var_name] = {}
        for var_SM_nc in var_SM_nc_list: # With observations read in place, forecast datasets initialized in different dates will be applied
            if "5sm" in var_SM_nc: # Skip the forecast dataset that is initialized on May 1st
                continue
            # Load the forecast dataset into the xarray obejct
            var_SM_nc_xr = xr.open_dataset(var_SM_nc, mask_and_scale=True, engine = "netcdf4", decode_times =True)
            if any(var_name in data_var_name for data_var_name in list(var_SM_nc_xr.data_vars)): # Only iterate through the forecast simulation files that share the same variable name as those of observations
                SM_var_name = [data_var_name for data_var_name in list(var_SM_nc_xr.data_vars) if var_name in data_var_name][0] # Unpack the list
                # Access the underlying xarray dataarray object for forecast dataset
                var_SM_arr = var_SM_nc_xr[SM_var_name]
                var_SM_arr_terciles = var_SM_arr.copy(deep=True)
                # Iterate over each ensemble member
                for ens_member in var_SM_arr.coords["number"].data:
                    # Obtain the data for each ensemble member
                    var_SM_arr_ens = var_SM_arr.sel({"number":ens_member}, method="nearest")
                    var_SM_arr_ens_terciles  = categorize_array(var_SM_arr_ens, reference_tercile =True, ref_arr = var_OB_arr) # Categorize the data into tercile groups
                    # Replace each slice of array with tercile grouped values 
                    var_SM_arr_terciles.loc[{"number":ens_member}] =  var_SM_arr_ens_terciles
                # Create a template xarray data array objects for BS1, BS2 and BS3 with dimensions of "lat", "lon" and "time"
                # The BS template is used for saving the score, i.e. forecast ensemble probability or something similar
                BS_template_tercile1 = xr.DataArray(coords=coords_xarray + [("time", np.array(early_pheno_years))]) 
                BS_template_tercile2 = xr.DataArray(coords=coords_xarray + [("time", np.array(normal_pheno_years))])
                BS_template_tercile3 = xr.DataArray(coords=coords_xarray + [("time", np.array(late_pheno_years))])
                # Accees the forecast initialization month for the current variety and stage combination
                fc_init_month = calendar.month_abbr[int("".join(re.findall(r"\d?",SM_var_name)))]
                # Iterate over both referenced and forecast datasets year by year
                for yearly_date_ob, yearly_date_fc in zip(var_OB_arr_terciles.time.data, var_SM_arr_terciles.time.data):
                    if yearly_date_ob != yearly_date_fc:
                        raise ValueError("The date time information is inconsistent between the two compared data array")
                    else:
                        # Obtain the information on the current iteration year
                        year_iter = pd.to_datetime(yearly_date_ob).year
                        #print(year_iter)
                    # Obtain the correct saving xarray data array object according to the current iteration year
                    # if year_iter in early_pheno_years:
                    #     BS_output_tercile = BS_template_tercile1.copy
                    # elif year_iter in normal_pheno_years:
                    #     BS_output_tercile = BS_template_tercile2.copy(deep=True)
                    # elif year_iter in late_pheno_years:
                    #     BS_output_tercile = BS_template_tercile3.copy(deep=True)
                    # Access the yearly data array from both observations and forecast
                    yearly_ob_data =  var_OB_arr_terciles.sel({"time":yearly_date_ob})
                    yearly_fc_data =  var_SM_arr_terciles.sel({"time":yearly_date_fc})
                    # Iterate over each grid point to compute the score
                    # Compute the lon and lat vectors 
                    lon_vector = yearly_ob_data[lon_name].data if np.array_equal(yearly_ob_data[lon_name], yearly_fc_data[lon_name],equal_nan=True) else np.nan
                    lat_vector = yearly_ob_data[lat_name].data if np.array_equal(yearly_ob_data[lat_name], yearly_fc_data[lat_name],equal_nan=True) else np.nan
                    # Form the list of coordinates to be studied
                    coordinates = list(product(lon_vector, lat_vector)) 
                    # Iterete over all coordinate grid points to extract the point timeseries values
                    for coordinate in coordinates:
                        # Unpack the coordinate tuple to obtain the longitude and latitude of a given grid point
                        lon1 = round(coordinate[0], 1) # Round to 1 decimal digit
                        lat1 = round(coordinate[1], 1) # Round to 1 decimal digit
                        # Extract the observational value at a single point, i.e. floating value at a given point
                        ob_data_val = yearly_ob_data.sel({lon_name:lon1, lat_name:lat1}, method="nearest").data # A value that represent a phenology class
                        # Extract the ensemble forecast member data at the same point as that of observation
                        fc_data_ser = yearly_fc_data.sel({lon_name:lon1, lat_name:lat1}, method="nearest").data # A series of phenology class forecast by each ensemble member
                        # Replace any values that equal to -999 into nan
                        if ob_data_val == -999:
                            ob_data_val = np.nan
                        elif any(fc_data_ser==-999):
                            fc_data_ser[fc_data_ser==-999] = np.nan # Equal any values at -999 to np.nan
                        # Check if any NaN values in the data, in case yes, it needs to be skipped
                        if np.isnan(ob_data_val):
                            continue # Skip the point with observed NaN values
                        elif any(np.isnan(fc_data_ser)):
                            continue # Skip the current point where NaN values are deteced in the series values
                        ## Proceed to computation section in case NaN values checks are passed ##
                        # Compute the selected forecast skill score for the point
                        fc_ens_correct = len(fc_data_ser[fc_data_ser == int(ob_data_val)]) # Number of forecast members that correctly predict the observation type
                        ens_size = len(yearly_fc_data.coords["number"]) # The ensemble size 
                        # Compute the fair BS score for the single point (https://www-miklip.dkrz.de/about/problems/)
                        # fair_BS = (fc_ens_correct/ens_size -1)**2 -  (fc_ens_correct*(ens_size- fc_ens_correct))/(ens_size**2 * (ens_size-1)) # The observation should always be 1
                        ens_prob = fc_ens_correct/ens_size # A simple measure of number of ensemble members that correctly predict the event 
                        # Attached the computed results into xarray data array object
                        if year_iter in early_pheno_years:
                            BS_template_tercile1.loc[{lon_name: lon1, lat_name: lat1, "time": year_iter}] = float(ens_prob)
                        elif year_iter in normal_pheno_years:
                            BS_template_tercile2.loc[{lon_name: lon1, lat_name: lat1, "time": year_iter}] = float(ens_prob)
                        elif year_iter in late_pheno_years:
                            BS_template_tercile3.loc[{lon_name: lon1, lat_name: lat1, "time": year_iter}] = float(ens_prob)
                # Attach the computed BS results into the target dict
                BS_score_dict[variety_name + "_" + var_name][str(fc_init_month) + "_" + "BS1"] = BS_template_tercile1.copy(deep=True)
                BS_score_dict[variety_name + "_" + var_name][str(fc_init_month) + "_" + "BS2"] = BS_template_tercile2.copy(deep=True)
                BS_score_dict[variety_name + "_" + var_name][str(fc_init_month) + "_" + "BS3"] = BS_template_tercile3.copy(deep=True)
# 6.5 Analyze the results and make the spatial map plots
# 6.5.1 Define essential input for the spatial map plot
BS_save_path = join(root_path,"output", "out_flo_ver", "plots", "BS_scores") # Define the output path
mkdir(BS_save_path) # Make the directory if not exist
cmap = discrete_cmap(5, base_cmap="summer_r") # Set the cmap to apply
cmap_bounds = np.linspace(0, 1, cmap.N+1) # Set the cmap bound to apply
map_proj = ccrs.PlateCarree() # Define the projection type
subplot_names = ["median", "90% range"] # Create a list of subplot names used for aggregated ploting
# 6.5.2 Get the maximum number of subplots from all associated data array
subplots = -np.inf
for var_stage in BS_score_dict.keys():
    for fc_BS in BS_score_dict[var_stage].keys():
        data_arr = BS_score_dict[var_stage][fc_BS]
        subplot_number = len(data_arr.time.data)
        if subplots<=subplot_number:
            subplots = subplot_number
# 6.5.3 Iterate over each variety and stage to make the respective plot
for var_stage in BS_score_dict.keys():
    var_stage_save_path = join(BS_save_path, var_stage)
    mkdir(var_stage_save_path)
    for fc_BS in BS_score_dict[var_stage].keys():            
        BS_score_arr = BS_score_dict[var_stage][fc_BS]
        #BS_score_sum = BS_score_arr.sum() # Sum values along all dimension, ("lon", "lat", "time")
        print(var_stage + " for " + fc_BS + "with a score of {}".format(str(float(BS_score_arr.median()) )))
        # Write CF standard attributes (including the CRS information) into target data array
        output__array = write_CF_attrs(BS_score_arr)
        # Clip the 3-D array
        clipped_array = output__array.rio.clip(GDF_shape.geometry, GDF_shape.crs, all_touched=False, drop=True, invert=False, from_disk = False)
        ## Make the temporal median plot for a given phenology group at a given forecast initialization date at a given variety and phenology stage
        median_arr = clipped_array.median(dim="time", skipna=True, keep_attrs= True)
        range_arr = clipped_array.quantile(0.95, dim="time", skipna=True, keep_attrs= True) - clipped_array.quantile(0.05, dim="time", skipna=True, keep_attrs= True)
        # Compute the zonal statistics for the target array
        median_arr_zonal_stat = zonal_statistics(median_arr, GDF_shape, col_name="DOC_ID", decimal_place=1)
        range_arr_zonal_stat =  zonal_statistics(range_arr, GDF_shape, col_name="DOC_ID", decimal_place=1)
        # Gather all plots into a list
        agg_arr = []
        agg_arr.append(median_arr_zonal_stat)
        agg_arr.append(range_arr_zonal_stat)
        # Get number of desired columns used in the spatial map plot
        subplot_col = len(clipped_array.time)
        for clipped_array_time in clipped_array.time.data:
            # Select the array for a given date time
            time_array = clipped_array.sel({"time":clipped_array_time})
            # Compute the zonal statistics for the selected time array
            clipped_array_yearly = zonal_statistics(time_array, GDF_shape, col_name="DOC_ID", decimal_place=1)
            # Replace the yearly array values of original array by the computed yearly results
            clipped_array.loc[{"time":clipped_array_time}] = clipped_array_yearly.data
        # Remember that the first and the last bin of supplied cmap will be used for ploting the colorbar extension. 
        # The bounds will determine number of color bins with extend set to "both" will add 2 extra bins, therefore the bound should anticipate number of color bins from cmap
        simulation_maps(fc_BS, clipped_array, GDF_shape, var_stage_save_path, 
                        cmap, "Ens_prob", plot_dim= "space",
                        subplot_row = 1, subplot_col = subplots, fig_size= (subplots, 2), extend="neither", 
                        add_scalebar=True, specify_bound=True, dpi=900,
                        fig_format=".png", bounds = cmap_bounds, grid_lon= list(grid_lons), grid_lat=list(grid_lats), 
                        subplot_title_font_size=3, outline = GDF_shape_outline, label_features = True)
        # Extract the forecast initialization month to be used in the file name for saving
        fc_init_month_fname = "".join(re.findall(r"\w+_", fc_BS)).strip("_")
        # Extract the tercile group number to be used in the file name for saving
        tercile_group = "".join(re.findall(r"\d+", fc_BS))
        # Form the file name used for file saving
        file_name = fc_init_month_fname + "_" + tercile_group
        # Make the plot
        simulation_maps(file_name, agg_arr, GDF_shape, var_stage_save_path, 
                        cmap, "Ens_prob", plot_dim= "aggregate",
                        subplot_row = 1, subplot_col = 2, fig_size= (2, 2), extend="neither", 
                        add_scalebar=True, specify_bound=True, fig_format=".png", dpi=900, bounds = cmap_bounds, 
                        grid_lon= list(grid_lons), grid_lat=list(grid_lats), subplot_title_font_size=3, outline = GDF_shape_outline, 
                        label_features = True, label_features_font_size=1.25, temporal_agg_var=subplot_names) 
                        #col_name="DOC_ID", list_geometry = [1, 10, 15, 16, 18, 20, 21, 22, 24, 26, 27, 36, 41,42])
# 6.6 ECDF plot for the spatial variability 
# Set a list of str that represent the forecast initialization months
fc_init_list = ["Feb", "Mar", "Apr"] 
# Set the line color for early/normal/late phenology tercile class
plot_line_col = {"early_phenology":"red", 
                 "normal_phenology":"black", 
                 "late_phenology":"blue"}
# Set the area plot subplot title
area_plot_subplot_title = {"early_phenology":"early_tercile", 
                 "normal_phenology":"normal_tercile", 
                 "late_phenology":"late_tercile"}
# Set the subplot configuration
subplot_row = 3 # Row-wise plot
subplot_col = len(fc_init_list) # Number of columns that depend on number of forecast initialization months per variety and stage
xy_ticks= np.arange(0, 1+0.1, 0.1) # Set the ticks that applied to both x- and y-axis, as x (ens_probabilitly from 0 to 1) and y (ecdf with probability from 0 to 1) share the same bounding values
# Set the plot type 
plot_types = ["line_plot", "area_plot"] # Set the plot type for the 5th percentile, median and 95th percentile over all years involved in a given category
# Set the plot saving path
BS_save_path = join(root_path,"output", "out_flo_ver", "plots", "BS_scores") # Define the output path
mkdir(BS_save_path) # Make the directory if not exist

for plot_type in plot_types: 
    for var_stage in BS_score_dict.keys():
        # Data to be plotted is contained in this dict
        var_stage_dict = BS_score_dict[var_stage] # Obtisn the underlying variety-stage ens score for all forecast initialization months
        # Set the figure and axe array configuration
        fig, axe_arr = plt.subplots(nrows = subplot_row, ncols = subplot_col, 
                                    figsize=(subplot_col*1.5, subplot_row), sharey="row",  sharex="col", dpi=600,
                                    gridspec_kw = dict(hspace = 0.05, wspace = 0.05)) # Create a figure instance class
        # Iterate over each forecast initiliazation and axe pair under a given variety-phenology stage pair
        for fc_init, axe_col_idx in zip(fc_init_list, range(subplot_col)): # axe_arr.flat
            # Obtain the str that represent a given forecast initialization month
            fc_init_keys = [key for key in list(var_stage_dict.keys()) if fc_init in key]
            # Under a given forecast initialization month, access all its phenology tercile classes, i.e. 1:early phenology; 2:normal phenology; 3: late phenology
            data_tercile1 =  var_stage_dict[[key for key in fc_init_keys if "1" in key][0]]
            data_tercile2 =  var_stage_dict[[key for key in fc_init_keys if "2" in key][0]]
            data_tercile3 =  var_stage_dict[[key for key in fc_init_keys if "3" in key][0]]
            # Create an empty dict to store plot data
            data_tercile_P_dict = {}
            # Create a nested dict to store data for the 5th, 50th and 95th percentile
            data_tercile_P_dict["5P"] = {}
            data_tercile_P_dict["median"] = {}
            data_tercile_P_dict["95P"] = {}
            # Under a given forecast initialization month, iterate over all its phenology data classes to make the ECDF line plots
            for tercile_str, data_tercile in {"early_phenology":data_tercile1, 
                                              "normal_phenology":data_tercile2, 
                                              "late_phenology":data_tercile3}.items():
                # Compute the median, 5th percentile and 95th percentile over all involved years (temporal variability) under each data tercile class
                data_tercile_5P = data_tercile.quantile(0.05, dim="time", skipna=True, keep_attrs= True)
                data_tercile_median = data_tercile.quantile(0.5, dim="time", skipna=True, keep_attrs= True)
                data_tercile_95P = data_tercile.quantile(0.95, dim="time", skipna=True, keep_attrs= True)
                # Flatten the data array and remove the NaN values
                data_tercile_5P_arr = data_tercile_5P.data[~np.isnan(data_tercile_5P.data)]
                data_tercile_median_arr = data_tercile_median.data[~np.isnan(data_tercile_median.data)]
                data_tercile_95P_arr = data_tercile_95P.data[~np.isnan(data_tercile_95P.data)]
                # Remove any infinite values
                if any(np.isinf(data_tercile_5P_arr)):
                    data_tercile_5P_arr = data_tercile_5P_arr[~np.isinf(data_tercile_5P_arr)]
                if any(np.isinf(data_tercile_95P_arr)):
                    data_tercile_95P_arr = data_tercile_95P_arr[~np.isinf(data_tercile_95P_arr)]
                # Attach the results into target dict
                if tercile_str not in data_tercile_P_dict["5P"].keys():
                    data_tercile_P_dict["5P"][tercile_str] = data_tercile_5P_arr.copy()
                if tercile_str not in data_tercile_P_dict["median"].keys():
                    data_tercile_P_dict["median"][tercile_str] = data_tercile_median_arr.copy()
                if tercile_str not in data_tercile_P_dict["95P"].keys():
                    data_tercile_P_dict["95P"][tercile_str] = data_tercile_95P_arr.copy()
            # For the line plot
            if plot_type=="area_plot":
                # Iterate over each percentile, axe pair
                for index, (tercile_pheno_str, axe) in enumerate(zip(plot_line_col.keys(), axe_arr[:, axe_col_idx].flat)):
                    ## Set the common axe parameters 
                    # Set the x- and y-axis tick labels
                    axe.set_xticks(xy_ticks) # Set the x-axis ticks
                    axe.set_xticklabels([str(round(tick,1)) for tick in xy_ticks], fontsize =3.5) # Set the x-axis tick labels
                    axe.set_yticks(xy_ticks) # Set the y-axis ticks
                    axe.set_yticklabels([str(round(tick,1)) for tick in xy_ticks], fontsize =3.5) # Set the y-axis tick labels
                    # Set the x-axis and y-axis tick parameters
                    axe.tick_params(axis='x',length=1, labelsize=3.5, pad=2)
                    axe.tick_params(axis='y',length=1, labelsize=3.5, pad=2)
                    # Set the boundary values for the axis, always from 0 to 1 under the mathplotlib axe coordinate
                    axe.set_xbound(0, max(axe.get_xbound())) # Set the x-axis bound
                    axe.set_ybound(0, max(axe.get_ybound())) # Set the y-axis bound
                    if index == 0: # Only set the title for the first (upper top) axe
                         # Set the subplot title as the forecast initialization month (1st day of the month)
                        axe.set_title(fc_init + "_1st", fontdict={"fontsize":5}, loc="center",  y = 0.95)
                    # Access the data array computed for the 5th, 95th percentile over all years involved in a given phenology tercile class
                    arr_5P_data = data_tercile_P_dict["5P"][tercile_pheno_str]
                    arr_median_data = data_tercile_P_dict["median"][tercile_pheno_str]
                    arr_95P_data = data_tercile_P_dict["95P"][tercile_pheno_str]
                    # Compute the ECDF curve for each tercile percentile
                    arr_5P_data_ecdf = ECDF(arr_5P_data)
                    arr_median_data_ecdf = ECDF(arr_median_data)
                    arr_95P_data_ecdf = ECDF(arr_95P_data)
                    # Obtain the plot color 
                    col = plot_line_col[tercile_pheno_str]
                    # Make the area plot
                    #x_ticks = np.linspace(0, 1, len(arr_5P_data_ecdf.y))
                    # Fill the upper line
                    axe.fill_between(arr_5P_data_ecdf.x, arr_5P_data_ecdf.y, 0,  # Provided the x position, ymax, ymin positions to fill 
                          facecolor= col, # The fill color
                          color= col,   # The outline color
                          edgecolors = None, # The line edge color
                          linewidth =0.5,
                          alpha=0.4)
                    # Fill the bottom line as while to override previous fill
                    axe.fill_between(arr_95P_data_ecdf.x, arr_95P_data_ecdf.y, 0,  # Provided the x position, ymax, ymin positions to fill 
                          facecolor= "white", # The fill color
                          color= "white",   # The outline color
                          edgecolors = None, # The line edge color
                          linewidth =0.5,
                          alpha=1)
                    # axe.plot(arr_5P_data_ecdf.x, arr_5P_data_ecdf.y, color=col, linewidth=0.5, linestyle='-') #dashes=[1,4]) #linestyle='--') # Expect to be beneath the median line     
                    # axe.plot(arr_95P_data_ecdf.x, arr_95P_data_ecdf.y, color=col, linewidth=0.5, linestyle='-') #dashes=[1,4]) #linestyle='--') # Expect to be beneath the median line     
                    # Add the text symbol for the percentile type
                    axe.text(0.8, 0.1, area_plot_subplot_title[tercile_pheno_str], fontsize=3.5, fontweight="bold", 
                             color='k', fontstyle='italic',horizontalalignment='center',transform=axe.transAxes)
            elif plot_type=="line_plot":
                # Iterate over each percentile, axe pair
                for index, (percentile_str, axe) in enumerate(zip(data_tercile_P_dict.keys(), axe_arr[:, axe_col_idx].flat)): 
                    ## Set the common axe parameters 
                    # Set the x- and y-axis tick labels
                    axe.set_xticks(xy_ticks) # Set the x-axis ticks
                    axe.set_xticklabels([str(round(tick,1)) for tick in xy_ticks], fontsize =3.5) # Set the x-axis tick labels
                    axe.set_yticks(xy_ticks) # Set the y-axis ticks
                    axe.set_yticklabels([str(round(tick,1)) for tick in xy_ticks], fontsize =3.5) # Set the y-axis tick labels
                    # Set the x-axis and y-axis tick parameters
                    axe.tick_params(axis='x',length=1, labelsize=3.5, pad=2)
                    axe.tick_params(axis='y',length=1, labelsize=3.5, pad=2)
                    # Set the boundary values for the axis, always from 0 to 1 under the mathplotlib axe coordinate
                    axe.set_xbound(0, max(axe.get_xbound())) # Set the x-axis bound
                    axe.set_ybound(0, max(axe.get_ybound())) # Set the y-axis bound
                    if index == 0: # Only set the title for the first (upper top) axe
                         # Set the subplot title as the forecast initialization month (1st day of the month)
                        axe.set_title(fc_init + "_1st", fontdict={"fontsize":5}, loc="center",  y = 0.95)
                        ## Compare the distribution of data for early/normal/late tercile class
                    # Obtain the respective 3 arrays
                    early_arr = data_tercile_P_dict[percentile_str]["early_phenology"]
                    normal_arr = data_tercile_P_dict[percentile_str]["normal_phenology"]
                    late_arr = data_tercile_P_dict[percentile_str]["late_phenology"]
                    # Perform the two-sample Kolmogorov-Smirnov test for goodness of fit between each paired two sample distributions 
                    ks_statistics, p1 = ks_2samp(normal_arr, early_arr, alternative='two-sided', mode= "auto") # p1 represents normal vs early
                    ks_statistics, p2 = ks_2samp(normal_arr, late_arr, alternative='two-sided', mode= "auto") # p2 represents normal vs late
                    ks_statistics, p3 = ks_2samp(early_arr, late_arr, alternative='two-sided', mode= "auto") # p3 represents early vs late
                    if all([p1 <0.05, p2<0.05, p3<0.05]): # In case each paired distribution show statistical significance, label the text
                        # Label the significance analysis results 
                        axe.text(0.85, 0.1, "p<0.05", fontsize=3.5, fontweight="bold", 
                              color='k', fontstyle='italic',horizontalalignment='center',transform=axe.transAxes)
                    # axe.text(0.85, 0.15, str(format(p2,".5f")), fontsize=3.5, fontweight="bold", 
                    #           color='k', fontstyle='italic',horizontalalignment='center',transform=axe.transAxes)
                    # axe.text(0.85, 0.2, str(format(p3,".5f")), fontsize=3.5, fontweight="bold", 
                    #           color='k', fontstyle='italic',horizontalalignment='center',transform=axe.transAxes)
                    # Define the text x_position depending on the type of plot
                    if percentile_str == "median":
                        x_position = 0.12
                    else:
                        x_position = 0.08
                    # Add the text symbol for the percentile type
                    axe.text(x_position, 0.9, percentile_str, fontsize=3.5, fontweight="bold", 
                             color='k', fontstyle='italic',horizontalalignment='center',transform=axe.transAxes)
                    # Make the line plot for each phenology class
                    for tercile_str_pheno, data_arr_P in data_tercile_P_dict[percentile_str].items():
                        # Obtain the respective line color under a given phenology class
                        line_col = plot_line_col[tercile_str_pheno]
                        # Set the alpha values
                        if "normal" in tercile_str_pheno:
                            alpha_val = 0.8
                        else:
                            alpha_val = 0.5                 
                        # Compute the ECDF curve for each tercile class
                        ecdf_val = ECDF(data_arr_P)
                        # ecdf_5th = ECDF(data_tercile_5P_arr)
                        # ecdf_median = ECDF(data_tercile_median_arr)
                        # ecdf_95th = ECDF(data_tercile_95P_arr)
                        # Make the line plot
                        #axe.plot(ecdf_5th.x, ecdf_5th.y, color=line_col, linewidth=0.3, linestyle=':') # dashes=[1,4]) #linestyle='--') # Expect to be above the median line 
                        axe.plot(ecdf_val.x, ecdf_val.y, color=line_col, linewidth=0.5, linestyle='-', alpha=alpha_val) # Median line
                        #axe.plot(ecdf_95th.x, ecdf_95th.y, color=line_col, linewidth=0.3, linestyle=':') #dashes=[1,4]) #linestyle='--') # Expect to be beneath the median line     
        # Save the plot into local disk as a file with specified format per plot type, variety-stage, initialization month
        fig.savefig(join(BS_save_path,"{}_{}_ensProb_ECDF.png".format(plot_type, str(var_stage))), 
                            dpi=600, bbox_inches="tight")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 7. Plot on the sigmoid function curve calibrated for 2 different varieties (TF and TN) and 2 different stages (flowering and veraison)
from Multi_phenology_model_classes import sigmoid_model
# 7.1 Define the set of parameters for TN used in this study
a_flo_TN, b_flo_TN = -0.15058, 23.72417
a_ver_TN, b_ver_TN = -39.99993, 15.4698
# 7.2 Define the set of parameters for TF used in this study
a_flo_TF, b_flo_TF =  -0.32078, 15.68677
a_ver_TF, b_ver_TF =  -39.99987, 15.48739
# 7.3 Construct a dict of sigmoid models that correspond to two varieties for the two different stages
sigmoid_models_dict = {"Sigmoid_flo_TN":sigmoid_model(a_flo_TN,b_flo_TN),
                       "Sigmoid_ver_TN":sigmoid_model(a_ver_TN,b_ver_TN),
                       "Sigmoid_flo_TF":sigmoid_model(a_flo_TF,b_flo_TF),
                       "Sigmoid_ver_TF":sigmoid_model(a_ver_TF,b_ver_TF)
                       }
# 7.4 Arrange the subplot layout
fig, axes = plt.subplots(int(len(sigmoid_models_dict)/2), int(len(sigmoid_models_dict)/2))
x = np.arange(0, 25+1, 1) # Fabricate a sequence of x values supplied to the sigmoid model
x_ticks = np.arange(0, 25+1, 1) # Set the fixed x ticks
y_ticks = np.arange(0, 1+0.1, 0.1) # Set the fixed y ticks 
# 7.5 Iterate over each axe subplot and sigmoid model dict
for axe, (sigmoid_name, sigmoid_cls) in zip(axes.flatten(), sigmoid_models_dict.items()):
    # Collect a list of y values that correspond to the sigmoid predicted values
    y = [sigmoid_cls.predict(x_val) for x_val in x] 
    # Make the line plot with a fixed x list values
    axe.plot(x, y, '--b') 
    # Set the subplot title
    axe.set_title(sigmoid_name, fontdict={"fontsize":7,
                                          "color":"black",
                                          "fontweight":"bold"
                                          }, y=0.85, pad =3)
    ## Set other text values in the plot
    # Set the a parameter text
    axe.text(0.9, 0.8, "a="+str(round(sigmoid_cls.a, 2)), fontsize=6, fontweight="bold", color='k', fontstyle='italic',horizontalalignment='center',transform=axe.transAxes)
    # Set the b parameter text
    axe.text(0.9, 0.7, "b="+str(round(sigmoid_cls.b, 2)), fontsize=6, fontweight="bold",color='k', fontstyle='italic',horizontalalignment='center',transform=axe.transAxes)
    # Set x-ticks and x-tick labels
    axe.set_xticks([int(value) for i,value in enumerate(x_ticks) if i%2==0])
    axe.set_xticklabels([str(int(value)) for i,value in enumerate(x_ticks) if i%2==0], size=6)
    # Set the y-ticks and y-tick labels
    axe.set_yticks([round(value,1) for i, value in enumerate(y_ticks)  if i%2==0 ])
    axe.set_yticklabels([str(round(value,1)) for i, value in enumerate(y_ticks) if i%2==0  ], size=6)
    # Set the y-axis lines 
    # axe.axvline(x=0, ymin=0, ymax=1, color='black') 
save_path = join(output_path, "out_flo_ver","calibration") # Re-define the output path
mkdir(save_path)
# Save the plot to local disk
fig.savefig(join(save_path,"sigmoid_curve.png"), dpi=600)
plt.close()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Plot code session++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
########################################################################## Coding Blocks #################################################################################################################################