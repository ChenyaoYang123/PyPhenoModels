
# title: "Main file for Implementation of the SCE-UA Algorithm for calibration of the run_BRIN_model"
# author: "J. Arturo Torres-Matallana(1)"
# organization (1): Luxembourg Institute of Science and Technology (LIST)
# date: 12.02.2022 - 03.03.2022

# import pandas as pd
# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri
# pandas2ri.activate()
# from rpy2.robjects.packages import importr
# xts= importr('xts', lib_loc="/usr/local/lib/R/site-library",
#              robject_translations = {".subset.xts": "_subset_xts2",
#                                      "to.period": "to_period2"})

# import numpy as np
# from scipy import stats
# from vineyard import *

# from spotpy.parameter import Uniform
# from spotpy.objectivefunctions import rmse
# import os
# import func
# import sce_ua

import os

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def my_sce_ua(d_ini, id_cpu, path_data, path_obs, cv_i):
    # importing  all the functions defined in func.py
    from func import f_data_in, f_subset, f_data_obs, f_subset_obs, f_subset_2
    from sce_ua import f

    # Define output folder
    folder_output = os.path.basename(path_obs)
    folder_output = os.path.splitext(folder_output)
    folder_output = folder_output[0]

    # 'Create output folder
    try:
        mkdir(folder_output)
    except OSError as error:
        print(error)

    # # ' Setup input data filepaths
    # data_file = path_data
    #
    # # ' Reading dataset Remich
    # # here \s+ for sep means any one or more white space character
    # data = f_data_in(dat_file=data_file, dat_sep=',', dat_day='day', dat_month='month', dat_year='Year')
    # # print(type(data))
    # # print(data.head())
    #
    # # ' Subset data by date
    # if len(d_ini) == 2:
    #     data_subset = f_subset(dat=data, date_ini=d_ini[0], date_end=d_ini[1])
    #     # print(type(data_subset))
    #     # print(data_subset)
    # elif len(d_ini) == 4:
    #     data_subset = f_subset_2(dat=data, date_ini=d_ini[0], date_end=d_ini[1], date_ini_2=d_ini[2], date_end_2=d_ini[3])
    #
    # # ' Export data_subset as csv file
    # data_subset.to_csv(folder_output + '/tmp_data_subset_' + str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) + '.csv')
    #
    # # ' Plotting time series
    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(16, 9))
    # plt.plot(data_subset['date'], data_subset['tas'], linewidth=0.5)
    # # plt.show()
    #
    # # show current monitor dpi https://www.infobyip.com/detectmonitordpi.php
    # plt.savefig(folder_output + '/tas_' + str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) + '.png', dpi=175)
    #
    # # Read observational data
    # data_file_obs = path_obs
    #
    # obs = f_data_obs(dat_file=data_file_obs, dat_sep=',')
    # print(obs)
    # print(data_subset)
    #
    # year = data_subset['Year'].unique()  # TODO: define for generic 'jahr' i.e 'year' instead
    #
    # # obs_subset = f_subset_obs(dat=obs, year_ini=np.int64(d_ini[0:4]), year_end=np.int64(d_end[0:4]))
    # obs_subset = f_subset_obs(dat=obs, year_ini=year[0], year_end=year[len(year) - 1])
    # print(obs_subset)
    # obs_bbch09 = obs_subset[["Years", "budburst.DOY."]]
    # print(obs_bbch09)
    # obs_bbch09.to_csv(folder_output + '/tmp_obs_' + str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) + '.csv')


    f(d_ini, id_cpu, folder_output, path_data, path_obs, cv_i+1)