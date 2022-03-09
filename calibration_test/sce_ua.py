
# title: "Implementation of the SCE-UA Algorithm for calibration of the run_BRIN_model"
# author: "J. Arturo Torres-Matallana(1); Chenyao Yang(2)"
# organization (1): Luxembourg Institute of Science and Technology (LIST)
# organization (2): ... (UTAD)
# date: 12.02.2022 - 03.03.2022

# importing  all the functions required
import spotpy
from spotpy.parameter import Uniform
# from spotpy.parameter import Constant
from spotpy.objectivefunctions import rmse

# import random

import numpy
import pandas

# import time
# from datetime import datetime, timedelta

from Multi_phenology_model_classes import *
import pandas as pd
from collections.abc import Iterable 

def f(d_ini, id_cpu, folder_out, path_data, path_obs, cv_i):

    class MySpotSetup(object):

        # par0: CCU_dormancy = 130
        par0_mean_low_mean = 110
        par0_mean_low_sd = par0_mean_low_mean*cv_i / 100
        par0_mean_low = numpy.random.normal(loc=par0_mean_low_mean, scale=par0_mean_low_sd, size= 1)
        # par0_mean_rng_5th  = numpy.percentile(a=par0_mean_rng, q = 5)
        # par0_mean_rng_95th = numpy.percentile(a=par0_mean_rng, q=95)
        par0_mean_high_mean = 150
        par0_mean_high_sd = par0_mean_high_mean*cv_i / 100
        par0_mean_high = numpy.random.normal(loc=par0_mean_high_mean, scale=par0_mean_high_sd, size= 1)
        par0_mean = Uniform(low=par0_mean_low, high=par0_mean_high)

        par0_sd_low_mean = par0_mean_low_sd
        par0_sd_low_sd = par0_sd_low_mean * cv_i / 100
        par0_sd_low = numpy.random.normal(loc=par0_sd_low_mean, scale=par0_sd_low_sd, size=1)
        par0_sd_high_mean = par0_mean_high_sd
        par0_sd_high_sd = par0_sd_high_mean * cv_i / 100
        par0_sd_high = numpy.random.normal(loc=par0_sd_high_mean, scale=par0_sd_high_sd, size=1)
        par0_sd = Uniform(low=par0_sd_low, high=par0_sd_high)


        # par1: CGDH_budburst = 7500.0 # for hourly
        # par1: CGDH_budburst = 2000 # for dialy
        par1_mean_low_mean = 500
        par1_mean_low_sd = par1_mean_low_mean * cv_i / 100
        par1_mean_low = numpy.random.normal(loc=par1_mean_low_mean, scale=par1_mean_low_sd, size=1)
        par1_mean_high_mean = 4000
        par1_mean_high_sd = par1_mean_high_mean * cv_i / 100
        par1_mean_high = numpy.random.normal(loc=par1_mean_high_mean, scale=par1_mean_high_sd, size=1)
        par1_mean = Uniform(low=par1_mean_low, high=par1_mean_high)

        par1_sd_low_mean = par1_mean_low_sd
        par1_sd_low_sd = par1_sd_low_mean * cv_i / 100
        par1_sd_low = numpy.random.normal(loc=par1_sd_low_mean, scale=par1_sd_low_sd, size=1)
        par1_sd_high_mean = par1_mean_high_sd
        par1_sd_high_sd = par1_sd_high_mean * cv_i / 100
        par1_sd_high = numpy.random.normal(loc=par1_sd_high_mean, scale=par1_sd_high_sd, size=1)
        par1_sd = Uniform(low=par1_sd_low, high=par1_sd_high)

        # par2: TMBc_budburst = 25.0
        par2_mean_low_mean = 18 # 22
        par2_mean_low_sd = par2_mean_low_mean * cv_i / 100
        par2_mean_low = numpy.random.normal(loc=par2_mean_low_mean, scale=par2_mean_low_sd, size=1)
        par2_mean_high_mean = 32 # 28
        par2_mean_high_sd = par2_mean_high_mean * cv_i / 100
        par2_mean_high = numpy.random.normal(loc=par2_mean_high_mean, scale=par2_mean_high_sd, size=1)
        par2_mean = Uniform(low=par2_mean_low, high=par2_mean_high)

        par2_sd_low_mean = par2_mean_low_sd
        par2_sd_low_sd = par2_sd_low_mean * cv_i / 100
        par2_sd_low = numpy.random.normal(loc=par2_sd_low_mean, scale=par2_sd_low_sd, size=1)
        par2_sd_high_mean = par2_mean_high_sd
        par2_sd_high_sd = par2_sd_high_mean * cv_i / 100
        par2_sd_high = numpy.random.normal(loc=par2_sd_high_mean, scale=par2_sd_high_sd, size=1)
        par2_sd = Uniform(low=par2_sd_low, high=par2_sd_high)

        # par3: TOBc_budburst=8.19
        par3_mean_low_mean = 2 # 6
        par3_mean_low_sd = par3_mean_low_mean * cv_i / 100
        par3_mean_low = numpy.random.normal(loc=par3_mean_low_mean, scale=par3_mean_low_sd, size=1)
        par3_mean_high_mean = 18 # 10
        par3_mean_high_sd = par3_mean_high_mean * cv_i / 100
        par3_mean_high = numpy.random.normal(loc=par3_mean_high_mean, scale=par3_mean_high_sd, size=1)
        par3_mean = Uniform(low=par3_mean_low, high=par3_mean_high)

        par3_sd_low_mean = par3_mean_low_sd
        par3_sd_low_sd = par3_sd_low_mean * cv_i / 100
        par3_sd_low = numpy.random.normal(loc=par3_sd_low_mean, scale=par3_sd_low_sd, size=1)
        par3_sd_high_mean = par3_mean_high_sd
        par3_sd_high_sd = par3_sd_high_mean * cv_i / 100
        par3_sd_high = numpy.random.normal(loc=par3_sd_high_mean, scale=par3_sd_high_sd, size=1)
        par3_sd = Uniform(low=par3_sd_low, high=par3_sd_high)

        def __init__(self, obj_func=None):


            self.obj_func = obj_func
            self.d_ini = d_ini
            self.id_cpu = id_cpu
            self.path_data = path_data
            self.path_obs = path_obs

            # defining obs file to read
            #dat_file = folder_out + '/tmp_obs_' + str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) + '.csv'
            #dat_sep = ','
            #self.obs_bbch09 = f_data_obs(dat_file, dat_sep)

            # Beginning of code implemented by Chenyao!
            # 1. Load observed phenology data and weather data
            budburst_raw = pd.read_csv(self.path_obs)  # Here the sample data only contains the budburst date for a given variety
            budburst_ob = pd.Series(budburst_raw.iloc[:, 1].values, index=budburst_raw["Year"], name="budburst_ob")  # Obtain the series for observation
            study_years = [min(budburst_ob.index) - 1] + budburst_ob.index.to_list()  # Obtain the inferred study years. Get -1 year so that it includes the preceding year of the first starting year
            weather_OB = pd.read_csv(self.path_data)  # Load the weather station data for Lisbon

            # weather_OB.to_csv(
            #     folder_out + '/tmp_weather_OB_' + str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) + '.csv') # ato

            weather_OB_datetime_index = pd.to_datetime(weather_OB.loc[:, weather_OB.columns.isin(["day", "month", "Year"])]) # Generate the datetime index
            T_min_series = pd.Series(weather_OB["tasmin"].values, index=weather_OB_datetime_index,name="tasmin")  # Obtain the time series of daily minimum temperature, the name should be named as tasmin.
            T_max_series = pd.Series(weather_OB["tasmax"].values, index=weather_OB_datetime_index,name="tasmax")  # Obtain the time series of daily maximum temperature,  the name should be named as tasmax.
            # Subset the series for which only study years are analyzed
            T_min_series = T_min_series.loc[T_min_series.index.year.isin(study_years)]
            T_max_series = T_max_series.loc[T_max_series.index.year.isin(study_years)]
            # Ending of code implemented by Chenyao!

            self.obs_bbch09 = budburst_ob
            self.T_min_series = T_min_series
            self.T_max_series = T_max_series

        def simulation(self, x):
            # Defining parameter to be passed to vineyard
            print('... before running model ...')

            detect_nan = False

            while numpy.invert(detect_nan):

                d_ini_sim = self.d_ini
                id_cpu_sim = self.id_cpu

                par0 = numpy.random.normal(loc=x[0], scale=x[1], size=1)
                par0 = round(par0[0], 1)

                par1 = numpy.random.normal(loc=x[2], scale=x[3], size=1)
                par1 = round(par1[0], 1)

                par2 = numpy.random.normal(loc=x[4], scale=x[5], size=1)
                par2 = round(par2[0], 1)

                par3 = numpy.random.normal(loc=x[6], scale=x[7], size=1)
                par3 = round(par3[0], 1)

                pars = pd.DataFrame([par0, par1, par2, par3])
                pars.to_csv(
                    folder_out + '/tmp_pars_' + str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) + '.csv') # ato

                # Beginning of code implemented by Chenyao!
                # 2. Run BRIN model to generate output for dormancy and budburst date
                dormancy_out, budburst_out = run_BRIN_model(self.T_min_series, self.T_max_series, CCU_dormancy=par0,
                                                            CGDH_budburst=par1, TMBc_budburst=par2, TOBc_budburst=par3,
                                                            Richarson_model='daily')
                # A list of BRIN model parameters that need to be calibrated (the docst can be retrieved by calling run_BRIN_model.__doc__)
                # CCU_dormancy: float, a BRIN model parameter that represents the cumulative chilling unit to break the dormancy with calculations starting from the starting_DOY.
                # CGDH_budburst: float, a BRIN model parameter that represents the cumulative growing degree hours to reach the budbust onset from the dormancy break DOY.
                # TMBc_budburst: float, a BRIN model parameter that represents the upper temperatue threshold for the linear response function.
                # TOBc_budburst: float, a BRIN model parameter that represents the base temperature for the post-dormancy period.

                # here we add the check function of NA values
                if any(pandas.isnull(budburst_out)):
                    detect_nan = True
                    break

                # 3.Compare predictions vs observations # Note here we only need to analyze the budburst DOY
                budburst_pred = pd.Series(budburst_out.apply(lambda x: x.dayofyear).values, index=budburst_out.index,
                                          name="budburst_pred")  # Obtain the prediction vector for budburst

                # The next step should be to compare budburst_pred with budburst_ob
                # Ending of code implemented by Chenyao!


                bbch09_sim = pandas.DataFrame(budburst_pred) # TODO: check: pandas.Series(budburst_pred)

                bbch09_sim.to_csv(
                    folder_out + '/tmp_bbch09_sim_' + str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) + '.csv',
                    index=False, header=False)  # ato

                bbch09_sim1 = bbch09_sim['budburst_pred']

                return bbch09_sim1.values

        def evaluation(self):
            obs_bbch09 = pandas.DataFrame(self.obs_bbch09)

            obs_bbch09.to_csv(
                folder_out + '/tmp_obs_bbch09_' + str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) + '.csv',
                index=False, header=False)  # ato

            return obs_bbch09['budburst_ob']

        def objectivefunction(self, simulation, evaluation, params=None):
            if isinstance(simulation,Iterable):   
                check_NaN_bool = any(pd.isnull(simulation)) 
            elif ~isinstance(simulation,Iterable):
                check_NaN_bool = pd.isnull(simulation)
            if check_NaN_bool:
                like = np.nan
                return like

            #simulation.to_csv(folder_out + '/my_sim.csv', mode="a", index=False, header=False)  # ato

            # check if -999 values are present in evaluation (obs)
            if min(evaluation) == -999:
                id_missing = numpy.where(evaluation == -999)
                id_missing1 = list(id_missing)
                id_missing1 = pandas.DataFrame(id_missing1)
                id_missing1 = id_missing1.T
                id_missing1.columns = ['id_missing']
                id_missing1.to_csv(folder_out + '/tmp_evaluation_id_missing_' + str(d_ini[0]) + '_' +
                                   str(d_ini[len(d_ini) - 1]) + '.csv')

                evaluation.to_csv(folder_out + '/tmp_evaluation_' + str(d_ini[0]) + '_' +
                                  str(d_ini[len(d_ini) - 1]) + '.csv')
                # simulation.to_csv(folder_out + '/tmp_simulation_' + str(d_ini[0]) + '_' + str(d_ini[1]) + '.csv')

                id_no_missing = numpy.where(evaluation != -999)
                evaluation = evaluation.loc[id_no_missing]
                simulation = simulation.loc[id_no_missing] # No needed after including len(d_ini) = 4

            # Evaluating rmse
            like = rmse(evaluation, simulation)

            like_2csv = pandas.DataFrame([like])

            like_2csv.to_csv(
                folder_out + '/tmp_like_' + str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) + '.csv', mode='a',
                index=False, header=False)  # ato

            return like

    # ' Calibrating with SCE-UA
    import sys
    sys.stdout = open(folder_out + '/SCEUA_log_' + str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) + '_cv' +
                      str(cv_i) +'.log', 'w')

    my_spot_setup = MySpotSetup(spotpy.objectivefunctions.rmse)
    print('... before sampler...')
    sampler = spotpy.algorithms.sceua(my_spot_setup, dbname=folder_out + '/SCEUA_db_' + str(d_ini[0]) + '_' +
                                                            str(d_ini[len(d_ini) - 1]) + '_cv' + str(cv_i) + '.csv',
                                      dbformat='csv')


    # ' Select number of maximum repetitions
    rep = 5000 # 5000

    # ' We start the sampler and set some optional algorithm specific settings
    # ' (check out the publication of SCE-UA for details):
    sampler.sample(rep, ngs=7, kstop=3, peps=0.1, pcento=0.1)

    # sys.stdout.close()

    # Get the results of the sampler
    results = sampler.getdata()

    # Use the analyser to show the parameter interaction _' + str(d_ini[0]) + '_' + str(d_ini[1]) + '.csv'
    spotpy.analyser.plot_parameterInteraction(results, fig_name=folder_out + '/Parameterinteraction_' +
                                                                str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) +
                                                                '_cv' + str(cv_i) + '.png')

    posterior = spotpy.analyser.get_posterior(results, percentage=10)

    spotpy.analyser.plot_parameterInteraction(posterior, fig_name=folder_out + '/Parameterinteraction_post_' +
                                                                  str(d_ini[0]) + '_' + str(d_ini[len(d_ini) - 1]) +
                                                                  '_cv' + str(cv_i) + '.png')

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
