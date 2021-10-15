"""
Utilities.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numpy as np
import pandas as pd
import matplotlib.dates as mdates


def date_ticks(ax, num_years, month, date_format='%Y-%m'):
    """
    Parameters
    ----------
    month: mathsci.constants.Month
    """

    ax.xaxis.set_major_locator(mdates.YearLocator(base=num_years,
                                                  month=month.value))

    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))


def namedtuple_asdf(nt):
    return pd.DataFrame(nt._asdict(), index=[""])


def format_date(mydate):
    return mydate.strftime('%d-%b-%Y')


def format_monthyear(mydate):
    return mydate.strftime('%b-%Y')


def format_monthyear_num(mydate):
    return mydate.strftime('%Y-%m')


def format_year(mydate):
    return mydate.strftime('%Y')


def datetime_to_float(some_date):
    """
    Returns year plus the fraction of year represented by the day in the year.

    eg: 01-JAN-2000 returns 2000 + 1/365

    TODO: account for leap years?
    """
    return some_date.year + (some_date.timetuple().tm_yday) / 365.0
#    return some_date.year + some_date.month / 12 - (some_date.day / 31) / 12


def single_timeseries_to_arrays(df):
    indep = np.array([datetime_to_float(x)
                      for x in df.index], dtype=float)
    dep = df.iloc[:, 0].values

    return indep, dep


def truncate(df, start_year=None, end_year=None):

    df_to_use = df

    if start_year is not None:
        df_to_use = truncate_startyear(df, start_year)
    if end_year is not None:
        df_to_use = truncate_endyear(df_to_use, end_year)

    return df_to_use


def truncate_endyear(df, end_year):
    """
    Returns input DataFrame truncating so it ends at end_year.  Includes
    the end_year.

    PARAMETERS
    ----------
    df: pandas DataFrame
        index should be date-like
    end_year: int
    """

    years = np.array([x.year for x in df.index])

    if end_year not in years:
        raise Exception("requested year: " + str(end_year) + " not in dataset")

    i = np.argwhere(years == end_year)[0][0]
    return df.iloc[0:i + 1, :]


def truncate_startyear(df, start_year):
    """
    Returns input DataFrame truncating so it starts at start_year.  Includes
    the start_year.

    PARAMETERS
    ----------
    df: pandas DataFrame
        index should be date-like
    start_year: int
    """
    years = np.array([x.year for x in df.index])

    if start_year not in years:
        raise Exception("requested year: " + str(start_year) +
                        " not in dataset")

    i = np.argwhere(years == start_year)[0][0]
    return df.iloc[i:, :]


def extend_dataframe(df, to_index=None):
    """
    Parameters
    ----------
    df: pandas DataFrame
    to_index: int
    """
    double = df.append(df)

    if to_index is not None:
        double = double.iloc[0:to_index, :]

    return double


def write_latex(df, out_path_name):

    my_index_names = []
    for index_name in df.index.names:
        my_index_names.append(index_name.replace("_", "\_"))

    # put index names on same row as columns
    latex = df.to_latex(multicolumn_format="c", multirow=True)
    lines = latex.split("\n")
    lines.pop(4)

    header_line = lines[3]
    tokens = header_line.split("&")

    for i, index_name in enumerate(my_index_names):
        tokens[i] = index_name

    header_line = "&".join(tokens)

    lines[3] = header_line

    with open(out_path_name, 'w') as f:
        for line in lines:
            f.write(line + "\n")
