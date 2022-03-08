import pandas as pd
import numpy as np
import math
############################################################################################################################################################################
###### Section 1. Define the BRIN model class that aims to compute the budburst date from the beginning of dormancy onset ######
############################################################################################################################################################################
class BRIN_model:
    '''
    Define a class representing the BRIN model to compute the budburst date, which is composed of modules to compute the date for both the dormancy break and burburst onset.
    Reference source: DOI 10.1007/s00484-009-0217-4
    
    Required input data and parameters to initialize the class
    ----------
    Tmin_input : a panda series, a time series of daily minimum temperature over study years with the index being the datetime on a daily scale. A minimum of 2-year is required. 
    Tmax_input : a panda series, a time series of daily maximum temperature over study years with the index being the datetime on a daily scale. A minimum of 2-year is required. 
    
    Endo-dormancy parameters
    ----------
    CCU: float, a cumulative chilling unit to break the dormancy with calculations starting from the starting_DOY.
    starting_DOY: int, the pre-specified day of year to start computation of Bidade model.
    
    Eco-dormancy parameters
    ----------
    CGDH: float, cumulative growing degree hours to reach the budbust onset from the dormancy break DOY.
    TMBc: float, upper temperatue threshold for the linear response function.
    TOBc: float, base temperature for the post-dormancy period.
    Richarson_model: str, choice of Richarson function, either with Richardson daily function ("daily") or hourly function ("hourly").
    '''
    # Define the BRIN model class instance attributes
    def __init__(self, Tmin_input, Tmax_input, CCU, starting_DOY,
                        CGDH, TMBc, TOBc, Richarson_model):
        ## Attributes list:
        # tmin: series, the daily Tmin pd.Series out of Tmin_input.
        # tmax: series, the daily Tmax pd.Series out of Tmax_input.
        # temp_df : df, the input df with two columns of daily minimum and maximum temperatures with 2 years since the dormancy calculation starts in the preceding year.
        # study_years: list of study years derived from the input.
        # two_year_combo: list of list integers, sublist elements are the combination of consecutively two years obtained from the study years.   
        self.tmin = Tmin_input
        self.tmax = Tmax_input
        # Check if two input series are having the same number of years
        assert np.array_equal(self.tmin.index.year.unique(), self.tmax.index.year.unique()),  "Tmin and Tmax series have different length of entries"# Test if two arrays are equal
        # Create additional attributes based on the input
        self.temp_df = pd.concat([self.tmin,self.tmax], axis = 1, join="inner") # Create the two-columns dataframe from tmin and tmax
        # Get the unique number of study years from the meteorology time series provided
        self.study_years = self.temp_df.index.year.unique()
        # Generate the consecutively 2-year combo from the yearly timeseries. This is necessary since the dormancy calculation starts from the previous year
        self.two_year_combo = [[year,year+1] for year in self.study_years if year != self.study_years[-1]] # list of list years that represent consecutively two study years
        # Define the parent class attributes that can access the child class attributes (for dormancy and post-dormancy module)
        self.endo_dormancy = self.dormancy_module(CCU, starting_DOY, self.temp_df, self.two_year_combo)
        self.eco_dormancy = self.postdormancy_module(CGDH, TMBc, TOBc, Richarson_model, self.temp_df, self.two_year_combo)
        
    ### Define the inner class that corresponds to a dormancy module that simulate the endo-dormancy period 
    ### using Q10 function (with daily minimum and maximum temperatures)
    class dormancy_module:
        '''
        Dormancy class module to calculate the dormancy break DOY from the starting_DOY
        
        Return
        ----------
        dormancy_output: series, a series of predicted dates for dormancy DOY
        '''   
        # Initialize the class with two compulsory parameters
        def __init__(self, CCU, starting_DOY, temp_df, two_year_combo):
            self.CCU = CCU
            self.starting_DOY = starting_DOY
            self.temp_df = temp_df
            self.two_year_combo = two_year_combo
        def predict(self):
            # Define the class instance method for prediction
            starting_DOY = self.starting_DOY
            CCU = self.CCU
            # Define an inner function within the predict() method, i.e. this inner function is only callable within predict() class instance method
            def Q10_func(x, Q10 = 2.17, tasmin="tasmin", tasmax="tasmax"): 
                '''
                The Bidade function in STICS is based on the Q10 concept, which aims to compute the cumulative chilling units from dormancy onset to dormancy break
                
                Parameter
                ----------
                x : df rows or columns, this will apply to a df.apply() function, so that x is each row or each column of df. The df is with two columns of daily minimum and maximum temperatures.
                Q10: float, base of the exponentional function in Bidabe model. Defaults to 2.17 for grapevine using a large sample dataset for calibration
                tasmin: str, the column head used to extract the column infor
                tasmax: str, the column head used to extract the column infor
                '''
                f = math.pow(Q10, -(float(x[tasmin])/10)) + math.pow(Q10, -(float(x[tasmax])/10))
                return f
            # Iterate over each two year combo to provide the prediction on the DOY of dormancty onset each season
            yearly_dormancy_date_ser = pd.Series(dtype= float)
            for two_year_list in self.two_year_combo:
                # Obtain a 2-year climate timeseries for the predictions. Note here the temp_df is the two-column temperature dataframe with index belonging to the datetime
                two_year_climate = self.temp_df.loc[self.temp_df.index.year.isin(two_year_list)]
                # Filter the 2-year climate so that the first year data starts from the pre-specified starting_DOY
                two_year_climate = two_year_climate.loc[~np.logical_and(two_year_climate.index.dayofyear < starting_DOY, 
                                                                    two_year_climate.index.year == min(two_year_list))]
                # Apply the Bidabe´s Q10 function over each row of the 2-year climate dataframe
                Q10_series = two_year_climate.apply(Q10_func, axis=1, raw=False, result_type= "reduce") # Raw needs to be set to False for row by row or column by column computation
                # Compute the cumulative values over a timeseries
                cdd_Q10 = Q10_series.cumsum() 
                # Find out the date where the threshold is firstly exceeded 
                if not cdd_Q10[cdd_Q10 >= CCU].empty:
                    DOY_index  = cdd_Q10[~(cdd_Q10 >= CCU)].argmax(skipna=True) + 1 # Return the date when the CCU has been satisfied
                    dormancy_break_date = cdd_Q10.index[DOY_index] # Obtain the target date in datetime object
                else:
                    dormancy_break_date = np.nan # In case the required thermal demand is not reached, assign the NaN value 
                # Create a series with a single entry representing the yearly dormancy date
                yearly_dormancy_date= pd.Series(dormancy_break_date, index=[dormancy_break_date.year])
                # Concatenate the resultant series into the empty series provided before 
                yearly_dormancy_date_ser = pd.concat([yearly_dormancy_date_ser,yearly_dormancy_date], axis=0, join="outer", ignore_index = False)
            # Copy and store the result series inside the instance attribute of .dormancy_output
            return yearly_dormancy_date_ser.copy(deep=True) # Return the yearly series of predicted dormancy date in datetime nc64 format
        
    ### Define the inner class that corresponds to a post-dormancy module that simulate the eco-dormancy period 
    ### using Richarson function either with daily or hourly function (with daily minimum and maximum temperatures)
    class postdormancy_module:
        '''
        Post-dormancy class module (either with hourly or daily Richarson) to calculate the budburst DOY from the dormancy-break DOY
               
        Return
        ----------
        Budburst_output: series, a series of predicted budburst DOY.
        '''
        # Initialize the class with compulsory parameters
        def __init__(self, CGDH, TMBc, TOBc, Richarson_model, temp_df, two_year_combo):
            self.CGDH = CGDH
            self.TMBc = TMBc
            self.TOBc = TOBc
            self.Richarson_model = Richarson_model
            self.temp_df = temp_df
            self.two_year_combo = two_year_combo
        # Define the class instance method for prediction
        def predict(self, _dormancy_output):
            CGDH = self.CGDH 
            TMBc = self.TMBc
            TOBc = self.TOBc
            Richarson_model = self.Richarson_model
            dormancy_output = _dormancy_output.copy(deep=True)
            # Define an inner function of predict() that correspond to the Richardson hourly function
            def Richarson_GDH_func(x, TMBc = TMBc, TOBc = TOBc, tasmin="tasmin", tasmax="tasmax", **kwargs): 
                '''
                The Richarson_GDH function in STICS is based on the hourly thermal time function, which aims to compute the cumulative thermal units from the dormancy break to budburst onset.
                This corresponds to the Richarson hourly function.
                
                Parameter
                ----------
                x : df rows or columns, this will apply to a df.apply() function, so that x is each row or each column of df. The df is with two columns of daily minimum and maximum temperatures.
                tasmin: str, the column head used to extract the column data.
                tasmax: str, the column head used to extract the column data.
                kwargs: additionaly keyword arguments passed to the method. By default, the df itself should be passed. 
                '''
                if len(kwargs)  != 0: # Check if the key word arguments are empty or not
                    input_df = kwargs["df_input"]# Obtain the dictionary value. By default, we know the supplied value must be the analyzed dataframe itself
                # Create an empty list to store the results
                houly_CGDH = []
                # The current date can not be equivalent to the last date of the input dataframe since this function needs to work on the two consecutive days
                if x.name != input_df.index[-1]: 
                    # Iterate over each of the 24 hours to compute the chilling effects for each hour
                    for h in range(24):
                        # Obtain the actual hour 
                        hour = h+1
                        # Compute the hourly temperature by interpolations between the daily minimum (current day and the next day) and maximum temperature (only the current day) 
                        if hour<=12:
                            hourly_temp  = x[tasmin] + hour * ((x[tasmax] - x[tasmin])/12)
                        elif hour>12:
                            # Note the datetime object is stored in the .name attribute when iterating over the rows
                            next_day_date = x.name + pd.DateOffset(days=1) 
                            hourly_temp  = x[tasmax] - (hour-12) * ((x[tasmax] - input_df.loc[next_day_date,tasmin])/12)   # Dateoffset with day 1: pd.DateOffset(days=1), pd.offsets.Day(1), pd.Timedelta(1, unit='d')
                        # Compute the growing degree hour effect from the hourly temperature 
                        if hourly_temp < TOBc:
                            hourly_gdh = 0
                        elif np.logical_and(TOBc < hourly_temp, hourly_temp<=TMBc):
                            hourly_gdh = hourly_temp - TOBc
                        elif hourly_temp > TMBc:
                            hourly_gdh = TMBc - TOBc
                        # Append the houly growing degree hourly effect into the target empty list
                        houly_CGDH.append(hourly_gdh)
                else:
                    pass
                # Return the cumulative GDH for a given day
                return np.nansum(houly_CGDH)
            # Define an inner function of predict() that correspond to the Richardson daily function
            def Richarson_daily_func(x, TMBc = TMBc, TOBc = TOBc):
                '''
                Richarson daily GDD function, which aims to compute the cumulative thermal units from the dormancy break to budburst onset (eco-dormancy period).
                
                Parameter
                ----------
                x : expected mean temperature on a given day, which is mainly applied in .apply() function.
                '''
                if x <= TOBc:
                    dd_day = 0
                else:
                    dd_day= max(min(x-TOBc, TMBc-TOBc),0)
                return dd_day
            
            # Firstly confirm if the two input array-like objects between two_year_combo and dormancy_data_ser are identical 
            assert len(self.two_year_combo) == len(dormancy_output), 'two input array length are not equal'
            # Iterate over each two year combo to provide the predictions on the date of budburst onset each season
            yearly_budburst_date_ser = pd.Series(dtype = float)
            for two_year_list, dormancy_date in zip(self.two_year_combo, dormancy_output): # The dormancy calculation must be performed first before calling the post-dormancy phase
                # Obtain a 2-year climate for the predictions
                two_year_climate = self.temp_df.loc[self.temp_df.index.year.isin(two_year_list)]
                # Filter the 2-year climate so that the first year date starts from the dormancy break date
                two_year_climate = two_year_climate.loc[~np.logical_and(two_year_climate.index < dormancy_date, 
                                                                    two_year_climate.index.year == min(two_year_list))]
                # Apply the Richarson_GDH function over each row of the 2-year climate dataframe
                if Richarson_model=="hourly":
                    GDH_series = two_year_climate.apply(Richarson_GDH_func, axis=1, raw=False, result_type= "reduce", df_input = two_year_climate) # Raw needs to be set to False for row by row or column by column computation
                elif Richarson_model=="daily":
                    two_year_climate_Tmean = two_year_climate.apply(np.nanmean, axis=1, raw=False, result_type= "reduce") # Compute daily mean temperature series 
                    GDH_series = two_year_climate_Tmean.apply(Richarson_daily_func, convert_dtype=True) # The apply function will lead to rest of series index
                # Compute the cumulative values over a timeseries
                cdd_GDH = GDH_series.cumsum()
                # Find out the date where the threshold is firstly exceeded 
                if not cdd_GDH[cdd_GDH >= CGDH].empty:
                    if cdd_GDH[~(cdd_GDH >= CGDH)].empty: # An empty sequence. This is the case where only one day value is higher than the threshold 
                        DOY_index = cdd_GDH.index.get_loc(cdd_GDH[cdd_GDH>=CGDH].idxmin()) # It is assumed the minimum index date (exceeding the threshold) corresponds to the right index 
                    else:
                        DOY_index  = cdd_GDH[~(cdd_GDH >= CGDH)].argmax(skipna=True) + 1 
                    budburst_date = cdd_GDH.index[DOY_index] # Obtain the target date in the datetime object
                    budburst_year = budburst_date.year
                else:
                    budburst_date = np.nan # In case the required thermal demand is not reached, assign the NaN value 
                    budburst_year = np.nan
                    print("Warning! NaN value of budburst date is found due to insufficient cumulative forcing unit in year {}".format(max(list(cdd_GDH.index.year.unique()))))
                # Create a series with a single entry representing the yearly budbreak date
                yearly_budburst_date= pd.Series(budburst_date, index=[budburst_year])
                # Concatenate the resultant series into the empty series provided before 
                yearly_budburst_date_ser = pd.concat([yearly_budburst_date_ser,yearly_budburst_date], axis=0, join="outer",ignore_index = False)
            # Copy and store the result series inside the instance attribute of .budburst_output for the yearly budburst date predictions
            return yearly_budburst_date_ser.copy(deep=True)
############################################################################################################################################################################       
class bug_holder:
    # Define the bug holder to catch any bugs during the simulation process
    bug_list = []
    def __init__(self,data):
        self.bug_list.append(data)
############################################################################################################################################################################
def run_BRIN_model(tasmin, tasmax, CCU_dormancy = None, T0_dormancy = 213, CGDH_budburst = None, 
                   TMBc_budburst= None, TOBc_budburst = None, Richarson_model= None, bug_catch= False, **kwargs):
    '''
    Run the BRIN model class to get the outputs for both the dormancy break and budburst.
    Important note that the minimum and maximum temperature series must have N +1 years, where N is the number of actual study years 
    
    Parameter
    ----------
    tasmin : a panda series, a time series of daily minimum temperature over study years with index being the datetimne on a daily scale. A minimum of 2 year is required. 
    tasmax : a panda series, a time series of daily maximum temperature over study years with index being the datetimne on a daily scale. A minimum of 2 year is required. 
    bug_catch: bool, if bug will be caught and write to an output variable. Note if this is True, addtional kwargs must be supplied
    CCU_dormancy: float, a BRIN model parameter that represents the cumulative chilling unit to break the dormancy with calculations starting from the starting_DOY.
    T0_dormancy: float, starting DOY to compute the endo-dormancy period. 
    CGDH_budburst: float, a BRIN model parameter that represents the cumulative growing degree hours to reach the budbust onset from the dormancy break DOY.
    TMBc_budburst: float, a BRIN model parameter that represents the upper temperatue threshold for the linear response function.
    TOBc_budburst: float, a BRIN model parameter that represents the base temperature for the post-dormancy period.
    Richarson_model: str, choice of Richarson function, either with Richardson daily function ("daily") or hourly function ("hourly").
    kwargs : any additional keyword arguments, normally expect the input argument of the grid point coordinate
    '''
    # Initialize the Brin model parent class
    BRIN_model_class = BRIN_model(tasmin, tasmax, CCU_dormancy, T0_dormancy,
                        CGDH_budburst, TMBc_budburst, TOBc_budburst, Richarson_model) # Initialize the BRIN model instance class, this will only process the input data
    # Initialize the endo-dormancy and eco-dormancy models (Inner classes)
    endo_dormancy_model = BRIN_model_class.endo_dormancy
    eco_dormancy_model = BRIN_model_class.eco_dormancy
    # Make the predictions 
    dormancy_output = endo_dormancy_model.predict()
    budburst_output = eco_dormancy_model.predict(dormancy_output) 
    # Check if the dormancy date equals to or be late than the budbreak date, which should always not happen. If happenns, assign NaN values
    for index, (dormancy_date, budburst_date) in enumerate(zip(dormancy_output, budburst_output)):
        if np.logical_or(pd.isnull(dormancy_date), pd.isnull(budburst_date)):
            continue # Skip NaN value in dormany or budburst DOY
        elif dormancy_date >= budburst_date: # If none are NaN values, they must conform to the datetime format
            dormancy_output.iloc[index] = np.nan
            budburst_output.iloc[index] = np.nan
            print("Warning!! The simulation errors are found as the simulated dormancy date is greater or equivalent to the budburst date")
            if bug_catch & (len(kwargs) != 0):
                bug_instances = bug_holder(list(kwargs.values())[0]) # Here we can append the coordinate of point into the bug_holder class
        else:
            continue
    if bug_catch & (len(kwargs) != 0):
        return (dormancy_output, budburst_output, bug_instances)
    else:
        return (dormancy_output, budburst_output)
############################################################################################################################################################################
###### Section 2. Define different phenology model classes that are about to run from the budburst date ######
############################################################################################################################################################################
class sigmoid_model: 
    '''
    A sigmoid model class.
    It is mainly designed to be called inside the phenology_SM_from_budburst(), which calculates 
    the daily effective degree day based on the PMP platform sigmoid model function with two cardinal parameters.
    
    Parameter
    ----------
    x : expected mean temperature on a given day, which is mainly applied in .apply() function.
    a: float, sharpness of the response curve. Values away from zero induce a sharper response curve.
    b: float, the mid-response temperature.
    '''
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def predict(self, x):
        para_a = self.a
        para_b = self.b
        dd_day = 1 / (1 + math.exp(para_a * (x-para_b)))
        return dd_day
###############################################################################
class triangular_model:
    '''
    A triangular model class. 
    It is mainly designed to be called inside the phenology_SM_from_budburst(), which calculates 
    the daily effective degree day based on the PMP platform triangular model function with three cardinal parameters.
    
    Parameter
    ----------
    x : expected mean temperature on a given day, which is mainly applied in .apply() function.
    Tdmin: float, base temperature below which development rate is null.
    Tdopt: float, optimum temperature at which development rate is optimum.
    Tdmax: float, maximum temperature beyond which development rate is null.
    '''
    def __init__(self, Tdmin, Tdopt, Tdmax):
        self.tdmin =  Tdmin
        self.tdopt =  Tdopt
        self.tdmax =  Tdmax
        
    def predict(self, x):
        T_dmin = self.tdmin
        T_dopt = self.tdopt
        T_dmax = self.tdmax
        
        if x <= T_dmin:
            dd_day = 0
        elif (T_dmin < x) & (x <= T_dopt):
            dd_day= (x - T_dmin)/(T_dopt - T_dmin)
        elif (T_dopt < x) & (x < T_dmax):
            dd_day= (x - T_dmax)/(T_dopt - T_dmax)
        else:
            dd_day = 0
        return dd_day
###############################################################################
class triangular_STICS_model:
    '''
    A triangular model class. 
    It is mainly designed to be called inside the phenology_SM_from_budburst(), which calculates 
    the daily effective degree day based on the STICS phenology module function with three cardinal parameters.
    
    Parameter
    ----------
    x : expected mean temperature on a given day, which is mainly applied in .apply() function.
    Tdmin: float, base temperature below which development rate is null. Default to 10 for grapevine.
    Tdmax: float, maximum temperature at which development rate is optimum. Default to 37 for grapevine.
    Tdstop: float, upper temperature limit beyond which development rate is null. Default to 100 for grapevine.
    '''
    def __init__(self, Tdmin, Tdmax, Tdstop):
        self.tdmin =  Tdmin
        self.tdmax =  Tdmax
        self.tdstop =  Tdstop
    
    def predict(self, x):  
        T_dmin = self.tdmin
        T_dmax = self.tdmax
        T_tdstop = self.tdstop

        if x <= T_dmin:
            dd_day = 0
        elif (T_dmin < x) & (x < T_dmax):
            dd_day = x - T_dmin
        elif (T_dmax <= x) & (x< T_tdstop):
            dd_day = (x - T_tdstop) * ((T_dmax-T_dmin)/(T_dmax-T_tdstop))
        elif x >= T_tdstop:
            dd_day = 0
        return dd_day
###############################################################################
class GDD_model:
    '''
    The GDD model class. 
    It is mainly designed to be called inside the phenology_SM_from_budburst(), which calculates 
    the daily effective degree day based on the classic GDD function with one parameter that corresponds to the base temperature
    
    Parameter
    ----------
    x : expected mean temperature on a given day, which is mainly applied in .apply() function.
    Tdmin: float, base temperature below which development rate is null. Default to 10 for grapevine.
    '''
    def __init__(self, Tdmin):
        self.tdmin =  Tdmin
    
    def predict(self, x): 
        T_dmin = self.tdmin
        if x <= T_dmin:
            dd_day = 0
        else:
            dd_day = x - T_dmin
        return dd_day
###############################################################################
class GDD_model_Richardson:
    '''
    The GDD model class. 
    It is mainly designed to be called inside the phenology_SM_from_budburst(), which calculates 
    the daily effective degree day based on the modified version of GDD function with two cardinal parameters.
    
    Parameter
    ----------
    x : expected mean temperature on a given day, which is mainly applied in .apply() function.
    Tdmin: float, base temperature below which development rate is null. Default to 10 for grapevine.
    '''
    def __init__(self, Tdmin, Tdmax):
        self.tdmin =  Tdmin
        self.tdmax =  Tdmax
    
    def predict(self, x): 
        T_dmin = self.tdmin
        T_dmax = self.tdmax
        if x <= T_dmin:
            dd_day = 0
        else:
            dd_day= max(min(x-T_dmin, T_dmax-T_dmin),0)
        return dd_day
###############################################################################
class wang_model:
    '''
    A beta model function class. 
    It is mainly designed to be called inside the phenology_SM_from_budburst(), which calculates 
    the daily effective degree day based on the wang model function with three cardinal parameters.
    Here the model is based on wang´s paper not based on the PMP documentation. 
    Note here we are not considering the photoperiod (which we should but in a later stage) and 
    vernalization (already computed using BRIN model during the dormancy phase) effects 
    
    Source: https://doi.org/10.1016/S0308-521X(98)00028-6
    
    Parameter
    ----------
    x : expected mean temperature on a given day, which is mainly applied in .apply() function.
    Tdmin: float, base temperature below which development rate is null.
    Tdopt: float, optimum temperature at which development rate is optimum.
    Tdmax: float, maximum temperature beyond which development rate is null.
    '''
    def __init__(self, Tdmin, Tdopt, Tdmax):
        self.tdmin =  Tdmin
        self.tdopt =  Tdopt
        self.tdmax =  Tdmax
        
    def predict(self, x):  
        T_dmin = self.tdmin
        T_dopt = self.tdopt
        T_dmax = self.tdmax
        # Compute the alpha that is going to be used as the exponent parameter in wang function
        alpha = math.log(2) / math.log((T_dmax-T_dmin)/(T_dopt-T_dmin))
        if (T_dmin <= x) & (x <= T_dmax):
            fwang_numerator = (2 * math.pow(x-T_dmin, alpha) *  math.pow(T_dopt-T_dmin, alpha)) - math.pow(x-T_dmin, 2 * alpha)
            fwang_denominator = math.pow(T_dopt-T_dmin, 2 * alpha)
            dd_day = float(fwang_numerator/fwang_denominator)
        else: # In case x < T_dmin or x > T_dmax
            dd_day = 0 
        return dd_day
###############################################################################
def phenology_SM_from_budburst(T_input, budburst_ser, thermal_threshold = 290, module = "STICS_GDD", DOY_format=True, **kwargs):
    '''
    Implement different phenology models to compute a target phenology stage from the budburst onset for grapevine.
    
    Parameter
    ----------
    T_input : series, a time series of daily mean temperature. It can be either mean crop temperature or mean air temperature. A minimum of two years is required.
    budburst_ser: series, a time series of yearly budburst date.
    thermal_threshold: float, a target thermal threshold to realize a given stage for a given thermal forcing model
    module: str, the choice of phenology model. They are generally the callable functions in this script. 
    DOY_format: bool, the output format of predictions. Either in DOY or datetime object. Default to DOY format.
    kwargs: any additional keyword arguments that can be passed to this function. Mostly here it is required to provide phenolgoy module-specific parameters. 
    '''
    if not isinstance(T_input, pd.core.series.Series):
        raise TypeError("The input temperature does not follow a series format, the format of {} is found".format(T_input.__class__)) 
    # Obtain a unique list of study years
    Years =  T_input.index.year.unique()
    # Obtain a list of two year combination list as the iteration list 
    two_year_combo = [[Year,Year+1] for Year in Years if Year != Years[-1]] 
    # Pre-define an empty series to be filled with data
    Yearly_date_ser = pd.Series(dtype= float)
    # Iterate over the two-year combo list 
    for index, two_year_tuple in enumerate(two_year_combo):
        # Extract the budburst date for this specific year
        budburst_date = budburst_ser.iloc[index] # Here the year is the index of a series, leading to a specific budburst date of a year prior predicted by the BRIN model.
        # Extract the consecutive two-year climate dataframe
        two_year_climate = T_input.loc[T_input.index.year.isin(two_year_tuple)]
        # Filter the 2-year climate so that the first year date starts from the budburst date
        two_year_climate = two_year_climate.loc[~np.logical_and(two_year_climate.index < budburst_date, 
                                                            two_year_climate.index.year == min(two_year_tuple))] 
        # Initialize the model class
        if len(kwargs) !=0: # If the dictionary is not empty, meaning additional keyword arguments are supplied. This is most likely when the phenology models are not yet calibrated
            if module=="classic_GDD":
                model = GDD_model(kwargs["Tdmin"])
            elif module=="GDD_Richardson":
                model = GDD_model_Richardson(kwargs["Tdmin"], kwargs["Tdmax"])
            elif module=="wang":
                model = wang_model(kwargs["Tdmin"], kwargs["Tdopt"], kwargs["Tdmax"])
            elif module=="triangular_STICS":
                model = triangular_STICS_model(kwargs["Tdmin"], kwargs["Tdmax"], kwargs["Tdstop"])
            elif module=="triangular":
                model = triangular_model(kwargs["Tdmin"], kwargs["Tdopt"], kwargs["Tdmax"])                
            elif module=="sigmoid":
                model = sigmoid_model(kwargs["a"],kwargs["b"])
        else: # This is most likely in the scenario where each phenology has already been calibrated and run with calibrated parameters
            if module=="classic_GDD":
                model = GDD_model()
            elif module=="GDD_Richardson":
                model = GDD_model_Richardson()
            elif module=="wang":
                model = wang_model()
            elif module=="triangular_STICS":
                model = triangular_STICS_model()
            elif module=="triangular":
                model = triangular_model()                
            elif module=="sigmoid":
                model = sigmoid_model()
        # elif len(kwargs) ==0: # If the dictionary is not empty, meaning additional keyword arguments are not supplied. This is most likely when the phenology models are already calibrated and started to be implemented
        #     if module=="STICS_GDD":
        #         dd_daily_ser = two_year_climate.apply(triangular_STICS_model)
        #     elif module=="classic_GDD":
        #         dd_daily_ser = two_year_climate.apply(dd_daily)
        #     elif module=="triangular":
        #         dd_daily_ser = two_year_climate.apply(triangular_model)
        #     elif module=="sigmoid":
        #         dd_daily_ser = two_year_climate.apply(sigmoid_model)
        # Apply the chosen phenology models into the two year climate series
        dd_daily_ser = two_year_climate.apply(model.implement)
        # Convert daily value into cumulative value
        cdd_daily = dd_daily_ser.cumsum()
        # Find out the date where the threshold is firstly exceeded 
        if not cdd_daily[cdd_daily >= thermal_threshold].empty:
            date_index = cdd_daily[~(cdd_daily >= thermal_threshold)].argmax(skipna=True) + 1 # Return the last date where the condition of CCU is still not met. While the next day, the CCU must be achieved.
            target_date = cdd_daily.index[date_index]
        else:
            target_date = np.nan # In case the required thermal demand is not reached, assign the NaN value 
        if DOY_format:
            # Create a series with a single entry representing the yearly DOY
            Yearly_date = pd.Series(target_date.dayofyear, index=[target_date.year])
        else:
            Yearly_date = pd.Series(target_date, index=[target_date.year])
        # Concatenate the resultant series into the empty series provided before 
        Yearly_date_ser = pd.concat([Yearly_date_ser,Yearly_date], axis=0, join="outer",ignore_index = False)
    # Return the predicted DOY by GDD
    return Yearly_date_ser
############################################################################################################






