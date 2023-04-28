import pandas as pd
import numpy as np
import math
########################################### Define different phenology model classes ###########################################################
class sigmoid_model: 
    '''
    A sigmoid model class
    It is to be called inside the implement_phenology_model().
    
    Parameter
    ----------
    x: float, expected mean temperature on a given day.
    a: float, sharpness of the response curve. Values away from zero induce a sharper response curve.
    b: float, the mid-response temperature.
    '''
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def predict(self, x):
        para_a = self.a
        para_b = self.b
        try:
            dd_day = 1 / (1 + math.exp(para_a * (x-para_b)))
            return dd_day
        except OverflowError as e:
            print(str(e))
            return 0 # Failed predictions will lead to accumulated GDD null
###############################################################################################################################################
class GDD_model:
    '''
    The GDD model class. 
    It is to be called inside the implement_phenology_model().
    
    Parameter
    ----------
    x: float, expected mean temperature on a given day.
    Tdmin: float, base temperature below which development rate is null.
    '''
    def __init__(self, Tdmin):
        self.tdmin =  Tdmin
    
    def predict(self, x): 
        T_dmin = self.tdmin
        try:
            if x <= T_dmin:
                dd_day = 0
            else:
                dd_day = x - T_dmin
            return dd_day
        except:
            print("Simulation errors occur while running the GDD_model model")
            return 0  # Failed predictions will lead to accumulated GDD null
###############################################################################################################################################
class GDD_model_Richardson:
    '''
    The GDD-Richardson model class. 
    It is to be called inside the implement_phenology_model().
    
    Parameter
    ----------
    x : float, expected mean temperature on a given day.
    Tdmin: float, base temperature below which development rate is null.
    Tdmax: float, maximum temperature above which development rate reaches plateau.
    '''
    def __init__(self, Tdmin, Tdmax):
        self.tdmin =  Tdmin
        self.tdmax =  Tdmax
    
    def predict(self, x): 
        T_dmin = self.tdmin
        T_dmax = self.tdmax
        try:
            if x <= T_dmin:
                dd_day = 0
            else:
                dd_day= max(min(x-T_dmin, T_dmax-T_dmin),0)
            return dd_day
        except:
            print("Simulation errors occur while running the GDD_model_Richardson")
            return 0 # Failed predictions will lead to accumulated GDD null
###############################################################################################################################################
class wang_model:
    '''
    A beta model function class. 
    It is to be called inside the implement_phenology_model().
    
    Parameter
    ----------
    x : float, expected mean temperature on a given day.
    Tdmin: float, base temperature below which development rate is null.
    Tdopt: float, optimum temperature at which development rate is optimum.
    Tdmax: float, maximum temperature above which development rate is null.
    '''
    def __init__(self, Tdmin, Tdopt, Tdmax):
        self.tdmin =  Tdmin
        self.tdopt =  Tdopt
        self.tdmax =  Tdmax
        
    def predict(self, x):  
        T_dmin = self.tdmin
        T_dopt = self.tdopt
        T_dmax = self.tdmax
        try:
            # Compute the alpha that is going to be used as the exponent parameter in wang function
            alpha = math.log(2) / math.log((T_dmax-T_dmin)/(T_dopt-T_dmin))
            if (T_dmin <= x) & (x <= T_dmax):
                fwang_numerator = (2 * math.pow(x-T_dmin, alpha) *  math.pow(T_dopt-T_dmin, alpha)) - math.pow(x-T_dmin, 2 * alpha)
                fwang_denominator = math.pow(T_dopt-T_dmin, 2 * alpha)
                dd_day = float(fwang_numerator/fwang_denominator)
            else: # In case x < T_dmin or x > T_dmax
                dd_day = 0 
            return dd_day
        except:
            print("Simulation errors occur while running the wang_model")
            return 0 # Failed predictions will lead to accumulated GDD null
###############################################################################################################################################
class CERES_Rice_Head_Mat:
    '''
    A class partly represents the rice phenology module of CSM_CERES_Rice model (a core part of DSSAT modelling platform). 
    Only use the phenology sub-model for the period between the heading and maturity stage of the rice plant.
    Therefore, the photoperiod effect is not considered (only relevant before anthesis).
    
    Parameters
    ----------
    Tbase: float, the base development temperature for the response function.
    Tdopt: float, the optimum development temperature for the response function.
    x: pandas.core.dataframe row, input data for the model contains two columns of daily minimum and maximum temperatures,
       where x is the individual row of the df. This will apply to the df.apply() function.
    '''
    # Initialize the class
    def __init__(self, Tbase, Tdopt, Tmin_name = "Tmin_daily", Tmax_name = "Tmax_daily"):
        # Assign the Tmin and Tmax names into the instance attribute
        self.tmin_name = Tmin_name
        self.tmax_name = Tmax_name
        # Assign the two essential parameters
        self.t_base = Tbase 
        self.t_dopt = Tdopt
    # Define the implementation module for CERES_Rice_Head_Mat
    def predict(self, x):
        # Access the base and optimum temperature parameters for the model
        T_base = self.t_base
        T_dopt = self.t_dopt
        # Check and ensure the input Tmin and Tmax series column name match the specified Tmin_name and Tmax_name
        if self.tmin_name not in list(x.index): # Index attribute always contains the assocaited names
            x.rename({x.index[0]:self.tmin_name}, inplace=True) # Rename the input row column name for Tmin (always the first column)
        if self.tmax_name not in list(x.index): # Index attribute always contains the assocaited names
            x.rename({x.index[1]:self.tmax_name}, inplace=True) # Rename the input row column name for Tmax (always the second column)
        # Access the minimum and maximum temperature for that day using the tmin_name and tmax_name
        T_min_day = float(x[self.tmin_name]) 
        T_max_day = float(x[self.tmax_name])
        if T_max_day<T_min_day:
            raise ValueError("The daily minimum temperature is higher than the daily maximum temperature for model run")
        # Implementation 
        try:
            if (T_min_day>T_base) and (T_max_day<T_dopt):
                DD_day = ((T_min_day+ T_max_day)/2) - T_base
            else: # Else calculated from hourly temperature using the sinusoidal interpolation betwen daily maximum and minimum temperature
                # Create an empty list to store the results
                houly_dd = []
                # Iterate over each of the 24 hours to compute the hourly degree day value
                for h in range(24):
                    # Obtain the actual hour (as h starts from 0)
                    hour = h+1
                    # Obtain the hourly temperature using the sinusoidal interpolation between daily minimum and maxium temperature (different interpolation method relative to APSIM model)
                    T_hour = (T_min_day+ T_max_day)/2 + ((T_max_day - T_min_day)/2) * math.sin((math.pi * hour)/12) # When the hourly temperature is between T_base and T_dopt
                    # Modify the hourly temperature if it falls within a certain range of temperature 
                    if T_hour < T_base:
                        T_hour = T_base
                    elif T_hour>T_dopt:
                        T_hour = T_dopt - (T_hour-T_dopt)
                    # Computes the hourly degree day based on the hourly temperature obtained
                    T_hour_DD = (T_hour-T_base)
                    # Append the computed hourly degree day into the empty list
                    houly_dd.append(T_hour_DD)
                # Aggregate hourly degree day into the value for the whole day
                DD_day = np.nansum(houly_dd)/24
            return DD_day
        except:
            print("Simulation errors occur while running the CERES_Rice_Head_Mat")
            return 0 # Failed predictions will lead to accumulated GDD null
###############################################################################################################################################
class APSIM_ORYZA_Rice_Head_Mat:
    '''
    A class partly represents the rice phenology module of APSIM-Oryza model (the phenology sub model of ORYZA2000).
    Only use the phenology sub-model for the period between the heading and maturity stage of the rice plant.
    Therefore, the photoperiod effect is not considered (only relevant before anthesis).
    
    Parameters
    ----------
    Tbase: float, the base development temperature for the response function.
    Tdopt: float, the optimum development temperature for the response function.
    Tmax: float, the maximum (limiting) development temperature for the response function.
    x: pandas.core.dataframe row, input data for the model contains two columns of daily minimum and maximum temperatures,
       where x is the individual row of the df. This will apply to the df.apply() function.
    '''
    # Initialize the class
    def __init__(self, Tbase, Tdopt, Tmax, Tmin_name = "Tmin_daily", Tmax_name = "Tmax_daily"):
        # Assign the Tmin and Tmax names into the instance attribute
        self.tmin_name = Tmin_name
        self.tmax_name = Tmax_name
        # Assign the three essential parameters
        self.t_base = Tbase 
        self.t_dopt = Tdopt
        self.t_limit = Tmax
    # Define the implementation module for APSIM_ORYZA_Rice_Head_Mat
    def predict(self, x):
        # Access the base, optimum and limiting temperature parameters for the model
        T_base = self.t_base
        T_dopt = self.t_dopt
        T_limit = self.t_limit 
        # Check and ensure the input Tmin and Tmax series column name match the specified Tmin_name and Tmax_name
        if self.tmin_name not in list(x.index): # Index attribute always contains the assocaited names
            x.rename({x.index[0]:self.tmin_name}, inplace=True) # Rename the input row column name for Tmin (always the first column)
        if self.tmax_name not in list(x.index): # Index attribute always contains the assocaited names
            x.rename({x.index[1]:self.tmax_name}, inplace=True) # Rename the input row column name for Tmax (always the second column)
        # Access the minimum and maximum temperature for that day using the tmin_name and tmax_name
        T_min_day = float(x[self.tmin_name]) 
        T_max_day = float(x[self.tmax_name])
        if T_max_day<T_min_day:
            raise ValueError("The daily minimum temperature is higher than the daily maximum temperature for model run")
        # Implementation
        try:
            # Create an empty list to store the results
            houly_dd = []
            # Iterate over each of the 24 hours to compute the hourly degree day value using interpolated hourly temperature
            for h in range(24):
                # Obtain the actual hour (as h starts from 0)
                hour = h+1
                # Obtain the hourly temperature using the sinusoidal interpolation between daily minimum and maxium temperature (different interpolation method relative to CERES model)
                T_hour = (T_min_day+ T_max_day)/2 + ((T_max_day - T_min_day)/2) * math.cos(0.2618 * (hour-14)) # When the hourly temperature is between T_base and T_dopt
                # Calculate the hourly degree day according to the hourly temperature 
                if (T_base<=T_hour) and (T_hour <= T_dopt):
                    T_hour_DD = T_hour - T_base
                elif (T_dopt<T_hour) and (T_hour < T_limit):
                    T_hour_DD = (T_dopt-T_base)-(T_hour-T_dopt)*(T_dopt-T_base)/(T_limit-T_dopt)
                else:
                    T_hour_DD = 0
                # Append the computed hourly degree day into the target empty list
                houly_dd.append(T_hour_DD)
            # Aggregate hourly degree day into the value for the whole day
            DD_day = np.nansum(houly_dd)/24
            return DD_day
        except:
            print("Simulation errors occur while running the APSIM_ORYZA_Rice_Head_Mat")
            return 0 # Failed predictions will lead to accumulated GDD null
###############################################################################################################################################