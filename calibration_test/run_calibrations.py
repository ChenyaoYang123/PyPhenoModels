from PyPhenoModels.Multi_phenology_model_classes import *
import pandas as pd
import time
from datetime import datetime,timedelta
######################################################################################################################################################################
class Timer():
    # This timer class can be ignored as it is only for test not planned to be implemented
    def start(self):
        print(f"[BENCHMARK] Start Time - {datetime.now()}")
        self._start_time = time.perf_counter()

    def _duration(self):
        duration = timedelta(seconds=time.perf_counter() - self._start_time)
        print(f"[BENCHMARK] Total Duration - {duration}")

    def end(self):
        self._duration()
        print(f"[BENCHMARK] End Time - {datetime.now()}")
######################################################################################################################################################################
# 1. Load observed phenology data and weather data
budburst_raw = pd.read_csv("sample_data.csv") # Here the sample data only contains the budburst date for a given variety
budburst_ob = pd.Series(budburst_raw.iloc[:, 1].values, index = budburst_raw["Year"], name = "budburst_ob") # Obtain the series for observation
study_years = [min(budburst_ob.index)-1] + budburst_ob.index.to_list()  # Obtain the inferred study years. Get -1 year so that it includes the preceding year of the first starting year
weather_OB = pd.read_csv("Lisboa region_no_NAN.csv") # Load the weather station data for Lisbon
weather_OB_datetime_index = pd.to_datetime(weather_OB.loc[:, weather_OB.columns.isin(["day", "month", "Year"])]) # Generate the datetime index
T_min_series = pd.Series(weather_OB["tasmin"].values, index = weather_OB_datetime_index, name = "tasmin") # Obtain the time series of daily minimum temperature, the name should be named as tasmin. 
T_max_series = pd.Series(weather_OB["tasmax"].values, index = weather_OB_datetime_index, name = "tasmax") # Obtain the time series of daily maximum temperature,  the name should be named as tasmax. 
# Subset the series for which only study years are analyzed
T_min_series = T_min_series.loc[T_min_series.index.year.isin(study_years)]
T_max_series = T_max_series.loc[T_max_series.index.year.isin(study_years)]
# 2. Run BRIN model to generate output for dormancy and budburst date
dormancy_out, budburst_out = run_BRIN_model(T_min_series, T_max_series, CCU_dormancy=130, CGDH_budburst = 7500.0, 
                   TMBc_budburst= 25.0, TOBc_budburst = 8.19)
# A list of BRIN model parameters that need to be calibrated (the docst can be retrieved by calling run_BRIN_model.__doc__)
# CCU_dormancy: float, a BRIN model parameter that represents the cumulative chilling unit to break the dormancy with calculations starting from the starting_DOY.
# CGDH_budburst: float, a BRIN model parameter that represents the cumulative growing degree hours to reach the budbust onset from the dormancy break DOY.
# TMBc_budburst: float, a BRIN model parameter that represents the upper temperatue threshold for the linear response function.
# TOBc_budburst: float, a BRIN model parameter that represents the base temperature for the post-dormancy period.

# 3.Compare predictions vs observations # Note here we only need to analyze the budburst DOY
budburst_pred = pd.Series(budburst_out.apply(lambda x: x.dayofyear).values, index = budburst_out.index, name="budburst_pred") # Obtain the prediction vector for budburst

# The next step should be to compare budburst_pred with budburst_ob