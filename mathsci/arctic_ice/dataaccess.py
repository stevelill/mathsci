"""
Routines to access data.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


import datetime
from enum import Enum
import os

import pandas as pd
import numpy as np

from mathsci.utilities import general_utilities as gu
from mathsci.constants import Month, Dataset
from mathsci.arctic_ice.arctic_ice_constants import DATA_ROOT_DIR

_ICE_EXTENT_UNITS = "Sea Ice Extent ($10^6\ km^2$)"
_ICE_EXTENT_XLABEL = "Time"
_ICE_EXTENT_COL_NAME = "Extent"

_DAY_OF_MONTH_STR = "day-of-month"
_MONTHLY_AVG_STR = "monthly-avg"

_DAY_OF_MONTH_REPR = 15

_DATA_ROOT_DIR = DATA_ROOT_DIR


class DatasetRegion(Enum):
    NH_EXTENT = 1
    SH_EXTENT = 2


_MONTH_LONG_NAME = {Month.JAN: "January",
                    Month.FEB: "February",
                    Month.MAR: "March",
                    Month.APR: "April",
                    Month.MAY: "May",
                    Month.JUN: "June",
                    Month.JUL: "July",
                    Month.AUG: "August",
                    Month.SEP: "September",
                    Month.OCT: "October",
                    Month.NOV: "November",
                    Month.DEC: "December"}


class Region(Enum):
    Northern_Hemisphere = 1
    Beaufort_Sea = 2
    Chukchi_Sea = 3
    East_Siberian_Sea = 4
    Laptev_Sea = 5
    Kara_Sea = 6
    Barents_Sea = 7
    Greenland_Sea = 8
    Baffin_Bay_Gulf_of_St_Lawrence = 9
    Canadian_Archipelago = 10
    Hudson_Bay = 11
    Central_Arctic = 12

    # these have sparse data; we won't use them
#     Bering_Sea = 13
#     Baltic_Sea = 14
#     Sea_of_Okhotsk = 15
#     Yellow_Sea = 16
#     Cook_Inlet = 17


def _arctic_ice_extent_month_name(month, data_freq_str, datasource):
    result = ("Arctic Sea Ice Extent " +
              month.name + " " +
              datasource + " " +
              data_freq_str)

    return result


def print_data_message(df):
    print("data start: ", df.index[0])
    print("data end:   ", df.index[-1])
    print("num data:   ", len(df.index))


def arctic_ice_extent_nsidc_monthly_avg(month,
                                        dataset=DatasetRegion.NH_EXTENT,
                                        start_year=None,
                                        end_year=None,
                                        verbose=False):
    """
    This is 'good' satellite-based data from NSIDC.

    The data is sourced from:
    F. Fetterer et al. Sea Ice Index, Version 3. Dataset G02135. NSIDC: National 
    Snow and Ice Data Center. Boulder, Colorado USA, 2017. 
    doi: https://doi.org/10.7265/N5K072F8

    NOTES
    -----
    These source files contain monthly averages, and do not include months
    where the daily data did not contain a full month's data.  Examples are
    JAN-1988.

    We recommend using this method to fetch monthly averages, as it is the
    most conservative with the data, and likely what people expect for the
    monthly averages.  Moreover, we fetch source data and do not perform
    operations upon it.

    NOTES
    -----
    The returned dataframe has date index labels as 15th of month.

    TODO
    ----
    Support for DatasetRegion.SH_EXTENT

    PARAMETERS
    ----------
    month: mathsci.constants.Month enum item
    dataset: DatasetRegion
        currently only supports the default DatasetRegion.NH_EXTENT
    start_year: int
    end_year: int
    verbose: bool

    Returns
    -------
    tuple with:
    pandas DataFrame: represents the data
    str: name
    str: xlabel
    str: ylabel
    array-like of float: independent data values 
    array-like of float: dependent data values
    """
    mydir = os.path.join(_DATA_ROOT_DIR, "nsidc/export")
    if dataset == DatasetRegion.NH_EXTENT:
        filename = "Sea_Ice_Index_Monthly_Data_by_Year_G02135_v3_NH_Extent.csv"
    elif dataset == DatasetRegion.SH_EXTENT:
        filename = "Sea_Ice_Index_Monthly_Data_by_Year_G02135_v3_SH_Extent.csv"
    else:
        raise Exception("dataset not valid")

    orig_df = pd.read_csv(os.path.join(mydir, filename), index_col=0)

    ice_extent_df = orig_df.loc[:, [_MONTH_LONG_NAME[month]]]
    ice_extent_df = ice_extent_df.dropna()
    ice_extent_df.columns = [_ICE_EXTENT_COL_NAME]

    dates = []
    for year in ice_extent_df.index:
        curr_date = datetime.date(year=int(year),
                                  month=month.value,
                                  day=_DAY_OF_MONTH_REPR)
        dates.append(curr_date)
    ice_extent_df.index = dates
    ice_extent_df.sort_index(inplace=True, ascending=True)

    if start_year is not None:
        ice_extent_df = gu.truncate_startyear(ice_extent_df,
                                              start_year)

    if end_year is not None:
        ice_extent_df = gu.truncate_endyear(ice_extent_df,
                                            end_year)

    if verbose:
        print_data_message(ice_extent_df)

    name = _arctic_ice_extent_month_name(month=month,
                                         data_freq_str=_MONTHLY_AVG_STR,
                                         datasource=Dataset.NSIDC.name)

    indep, dep = gu.single_timeseries_to_arrays(ice_extent_df)

    ice_extent_df.index = pd.to_datetime(ice_extent_df.index)

    return (ice_extent_df,
            name,
            _ICE_EXTENT_XLABEL,
            _ICE_EXTENT_UNITS,
            indep,
            dep)


def arctic_ice_extent_sibt1850_midmonth(month,
                                        region=None,
                                        start_year=None,
                                        end_year=None,
                                        verbose=False):
    """
    Returns SIBT 1850 data.

    The data is sourced from:
    J. E. Walsh et al. Gridded Monthly Sea Ice Extent and Concentration, 
    1850 Onward, Version 2. Dataset G10010 V2. NSIDC: National Snow and Ice Data 
    Center. Boulder, Colorado USA, 2019. doi: https://doi.org/10.7265/jj4s-tq79

    NOTES
    -----
    The returned dataframe has date index labels as 15th of month.

    PARAMETERS
    ----------
    month: mathsci.constants.Month enum item
    region: mathsci.climate_change.arctic_ice.dataaccess.Region
    start_year: int
    end_year: int
    verbose: bool

    Returns
    -------
    tuple with:
    pandas DataFrame: represents the data
    str: name
    str: xlabel
    str: ylabel
    array-like of float: independent data values 
    array-like of float: dependent data values
    """
    mydir = os.path.join(_DATA_ROOT_DIR, "sibt_1850/export")
    filename = os.path.join(mydir, "sibt_extents_v2.csv")
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    df = df.dropna()
    df.sort_index(inplace=True, ascending=True)

    new_ind = [datetime.datetime.strptime(str(x), '%Y%j') for x in df.index]
    df.index = new_ind

    if region is None:
        region = Region.Northern_Hemisphere

    if region not in Region:
        raise Exception("region: " + region + " not found")

    long_ind = [x for x in df.index if x.month == month.value]
    long_df = df.loc[long_ind, [region.name]]
    long_df.columns = [_ICE_EXTENT_COL_NAME]

    scale = 1.0e6
    long_df = long_df / scale

    if start_year is not None:
        long_df = gu.truncate_startyear(long_df, start_year)

    if end_year is not None:
        long_df = gu.truncate_endyear(long_df, end_year)

    if verbose:
        print_data_message(long_df)

    # dropping data freq since there is no choice for it
    # data_freq_str=_DAY_OF_MONTH_STR
    data_freq_str = ""
    name = _arctic_ice_extent_month_name(month=month,
                                         data_freq_str=data_freq_str,
                                         datasource=Dataset.SIBT1850.name)

    indep, dep = gu.single_timeseries_to_arrays(long_df)

    return (long_df,
            name,
            _ICE_EXTENT_XLABEL,
            _ICE_EXTENT_UNITS,
            indep,
            dep)


def arctic_ice_extent_blend(month,
                            start_year=None,
                            end_year=None,
                            verbose=False,
                            use_nsidc_day_of_month=False):
    """
    Combines arctic ice extent data from the SIBT1850 and NSIDC satellite data.

    The motivation is to be able to extend SIBT1850 with new data going
    forward.  We would not use the blend for short-term series, such as
    starting at 1979.  For that we have the gold-standard satellite NSIDC data.

    NOTES
    -----
    The returned dataframe has date index labels as 15th of month.

    PARAMETERS
    ----------
    month: mathsci.constants.Month enum item
    start_year: int
    end_year: int
    verbose: bool
    use_nsidc_day_of_month: bool
        If True, uses mid-month data from the satellite data rather than the
        monthly averages.

    Returns
    -------
    tuple with:
    pandas DataFrame: represents the data
    str: name
    str: xlabel
    str: ylabel
    array-like of float: independent data values 
    array-like of float: dependent data values
    """
    (ice_extent_df,
     name,
     xlabel,
     ylabel,
     indep,
     dep) = arctic_ice_extent_sibt1850_midmonth(month, verbose=False)

    ##

    if use_nsidc_day_of_month:
        (nsidc_ice_extent_df,
         name,
         xlabel,
         ylabel,
         nsidc_indep,
         nsidc_dep) = arctic_ice_extent_nsidc_dayofmonth(month,
                                                         day_of_month=15,
                                                         verbose=False)
    else:
        (nsidc_ice_extent_df,
         name,
         xlabel,
         ylabel,
         nsidc_indep,
         nsidc_dep) = arctic_ice_extent_nsidc_monthly_avg(month,
                                                          verbose=False)

    overlap_start = 1979
    overlap_end = 2017

    nsidc_overlap = gu.truncate(nsidc_ice_extent_df,
                                start_year=overlap_start,
                                end_year=overlap_end)

    # note: the monthly avg might remove some months: eg: DEC-1987, JAN-1988
    # so we expect that nsidc may have fewer obs
    long_overlap = ice_extent_df.loc[nsidc_overlap.index, :]

    long_prior = gu.truncate(ice_extent_df, end_year=overlap_start - 1)

    nsidc_post = gu.truncate(nsidc_ice_extent_df, start_year=overlap_end + 1)

    def ramp_linear(x, N):
        return -(x - N) / (N + 1.0)

    def ramp_quadratic(x, N):
        a = 7 * 1.0e-4
        b = (a - 1.0 - a * N * N) / (1.0 + N)
        c = 1.0 - a + b
        return a * x * x + b * x + c

    def ramp_func(x1, x2):
        N = len(x1)
        domain = np.arange(N)

        w1 = ramp_linear(domain, N)

        w2 = (1.0 - w1)
        return w1 * x1 + w2 * x2

    def myfunc(x1, x2):
        # w1 will be weight for 1850sibt
        w1 = 0.5
        w2 = (1.0 - w1)

        return w1 * x1 + w2 * x2

    diff = long_overlap.values.ravel() - nsidc_overlap.values.ravel()

    drop_amt = np.median(diff)

    long_prior -= drop_amt
    long_overlap -= drop_amt

    overlap_blend = long_overlap.combine(nsidc_overlap,
                                         func=ramp_func)

    combined = pd.concat([long_prior,
                          overlap_blend,
                          nsidc_post], join="outer", axis=0)

    if start_year is not None:
        combined = gu.truncate_startyear(combined, start_year)

    if end_year is not None:
        combined = gu.truncate_endyear(combined, end_year)

    if verbose:
        print_data_message(combined)

    indep, dep = gu.single_timeseries_to_arrays(combined)

    if use_nsidc_day_of_month:
        name = _arctic_ice_extent_month_name(month=month,
                                             data_freq_str=_DAY_OF_MONTH_STR,
                                             datasource=Dataset.BLEND.name)
    else:
        name = _arctic_ice_extent_month_name(month=month,
                                             data_freq_str="",
                                             datasource=Dataset.BLEND.name)

    return (combined,
            name,
            xlabel,
            ylabel,
            indep,
            dep)


def arctic_ice_data_avgFROMDAILY(month, verbose=False):
    """
    NOTES
    -----
    We compute monthly averages directly from the daily data.  Some months do
    not contain a full month's data (eg: JAN-1988).  Currently, we average 
    whatever daily data is available for a given month, including the most
    recent month.  TODO: keep, or change?

    return index labels as 15th of month

    PARAMETERS
    ----------
    month: Month enum item

    Returns
    -------
    tuple with:
    pandas DataFrame: represents the data
    str: name
    str: xlabel
    str: ylabel
    array-like of float: independent data values 
    array-like of float: dependent data values
    """
    mydir = os.path.join(_DATA_ROOT_DIR, "nsidc/export")
    filename = "N_seaice_extent_daily_v3.0.csv"
    orig_df = pd.read_csv(os.path.join(mydir, filename))

    dates = []
    for row in orig_df.values:
        curr_date = datetime.date(year=int(row[0]),
                                  month=int(row[1]),
                                  day=int(row[2]))
        dates.append(curr_date)
    df = pd.DataFrame({_ICE_EXTENT_COL_NAME: orig_df.Extent.values},
                      index=dates)
    df.sort_index(inplace=True, ascending=True)

    all_years = set([x.year for x in df.index])

    month_means = []
    month_mean_dates = []

    month_ind = month.value

    for year in all_years:
        month_index = [x for x in df.index if x.month == month_ind and
                       x.year == year]
        month_df = df.loc[month_index, :]

        month_means.append(np.mean(month_df.values))
        month_mean_dates.append(datetime.date(year=year,
                                              month=month_ind,
                                              day=_DAY_OF_MONTH_REPR))

    ice_extent_df = pd.DataFrame({_ICE_EXTENT_COL_NAME: month_means},
                                 index=month_mean_dates)

    ice_extent_df = ice_extent_df.dropna()

    if verbose:
        print_data_message(ice_extent_df)

    # TODO: get better month name here
    name = _arctic_ice_extent_month_name(month=month,
                                         data_freq_str=_MONTHLY_AVG_STR,
                                         datasource=Dataset.NSIDC.name)

    indep, dep = gu.single_timeseries_to_arrays(ice_extent_df)

    return (ice_extent_df,
            name,
            _ICE_EXTENT_XLABEL,
            _ICE_EXTENT_UNITS,
            indep,
            dep)


def arctic_ice_extent_nsidc_dayofmonth(month,
                                       day_of_month=15,
                                       start_year=None,
                                       end_year=None,
                                       verbose=False):
    """
    NOTES
    -----
    Some months do not contain a full month's data (eg: JAN-1988).

    return index labels as 15th of month (even though data might be from a
    nearby date

    PARAMETERS
    ----------
    month: Month enum item
    day: int

    Returns
    -------
    tuple with:
    pandas DataFrame: represents the data
    str: name
    str: xlabel
    str: ylabel
    array-like of float: independent data values 
    array-like of float: dependent data values
    """
    mydir = os.path.join(_DATA_ROOT_DIR, "nsidc/export")
    filename = "N_seaice_extent_daily_v3.0.csv"
    orig_df = pd.read_csv(os.path.join(mydir, filename))

    dates = []
    for row in orig_df.values:
        curr_date = datetime.date(year=int(row[0]),
                                  month=int(row[1]),
                                  day=int(row[2]))
        dates.append(curr_date)
    df = pd.DataFrame({_ICE_EXTENT_COL_NAME: orig_df.Extent.values},
                      index=dates)
    df.sort_index(inplace=True, ascending=True)

    # some months do not have given day-of-month; will fetch nearby values
    # instead
    years = set([x.year for x in df.index if x.month == month.value])

    daymonth_ind = []
    for year in years:
        month_vals = [x for x in df.index if x.year == year and
                      x.month == month.value]

        month_result = None
        for val in month_vals:
            if val.day <= day_of_month:
                month_result = val
        if month_result is not None:
            daymonth_ind.append(month_result)

    daymonth_df = df.loc[daymonth_ind, :]
    daymonth_df.columns = [_ICE_EXTENT_COL_NAME]
    daymonth_df = daymonth_df.dropna()

    # make index be 15th of month
    new_ind = [datetime.date(year=x.year,
                             month=x.month,
                             day=_DAY_OF_MONTH_REPR) for x in daymonth_df.index]

    # convert to datetime
    daymonth_df.index = pd.to_datetime(new_ind)

    if start_year is not None:
        daymonth_df = gu.truncate_startyear(daymonth_df,
                                            start_year)

    if end_year is not None:
        daymonth_df = gu.truncate_endyear(daymonth_df, end_year)

    if verbose:
        print_data_message(daymonth_df)

    name = _arctic_ice_extent_month_name(month=month,
                                         data_freq_str=_DAY_OF_MONTH_STR,
                                         datasource=Dataset.NSIDC.name)

    indep, dep = gu.single_timeseries_to_arrays(daymonth_df)

    return (daymonth_df,
            name,
            _ICE_EXTENT_XLABEL,
            _ICE_EXTENT_UNITS,
            indep,
            dep)
