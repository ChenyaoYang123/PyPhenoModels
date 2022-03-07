
# title: "Run the SCE-UA Algorithm for calibration of the run_BRIN_model"
# author: "J. Arturo Torres-Matallana(1); Chenyao Yang(2)"
# organization (1): Luxembourg Institute of Science and Technology (LIST)
# organization (2): ... (UTAD)
# date: 12.02.2022 - 03.03.2022

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

from main import my_sce_ua
import random

time_start = datetime.now()
time_start_str = time_start.strftime("%Y-%m-%d %H:%M:%S")
print("Start time =", time_start_str)

path_data1 = ("./calibration_test/Lisboa region_no_NAN.csv")

path_obs1 = ("./calibration_test/sample_data.csv")


d_ini = ["1990-01-01", "2014-12-31"]
id_cpu = random.randint(100, 999)

# for cv_i in range(1): # run until 20
cv_i = 9
my_sce_ua(d_ini, id_cpu, path_data1, path_obs1, cv_i)

time_end = datetime.now()
time_end_str = time_end.strftime("%Y-%m-%d %H:%M:%S")
print("End time =", time_end_str)

time_elapsed = time_end - time_start
print("Elapsed time =", time_elapsed)