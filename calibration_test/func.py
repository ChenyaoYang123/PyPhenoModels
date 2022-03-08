
# title: "Auxiliary functions for the SCE-UA Algorithm for calibration"
# author: "J. Arturo Torres-Matallana(1)"
# organization (1): Luxembourg Institute of Science and Technology (LIST)
# date: 12.02.2022 - 03.03.2022

import pandas as pd


def f_data_obs(dat_file, dat_sep):
    # here \s+ for sep means any one or more white space character
    data = pd.read_csv(dat_file, sep=dat_sep)
    return data


def f_data_in(dat_file, dat_sep, dat_day, dat_month, dat_year):
    # here \s+ for sep means any one or more white space character
    data = pd.read_csv(dat_file, sep=dat_sep)

    # 'creating date column
    # date_1 = pd.DataFrame({'year': data[dat_year],
    #                        'month': data[dat_month],
    #                        'day': data[dat_day]})
    # cols = ["jahr", "mo", "ta"]
    cols = [dat_year, dat_month, dat_day]
    date_1 = data[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    date_1 = pd.to_datetime(date_1)

    # create date column in data
    data['date'] = date_1
    print(data)
    return data


def f_subset(dat, date_ini, date_end):
    df = dat[(dat['date'] >= date_ini) & (dat['date'] <= date_end)]
    return df

def f_subset_2(dat, date_ini, date_end, date_ini_2, date_end_2):
    df = dat[(dat['date'] >= date_ini) & (dat['date'] <= date_end)]
    df_2 = dat[(dat['date'] >= date_ini_2) & (dat['date'] <= date_end_2)]

    df_final = pd.concat([df, df_2])
    return df_final

def f_subset_obs(dat, year_ini, year_end):
    df = dat[(dat['Years'] >= year_ini) & (dat['Years'] <= year_end)]
    return df