"""
Utilities to aid in data access.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


from collections import namedtuple

from mathsci.arctic_ice import dataaccess
from mathsci.constants import Month, Dataset


def fetch_month(dataset, month, start_year=None, end_year=None):

    if dataset == Dataset.NSIDC:
        (ice_extent_df,
         name,
         xlabel,
         ylabel,
         indep,
         dep) = dataaccess.arctic_ice_extent_nsidc_monthly_avg(
            month=month,
            start_year=start_year,
            end_year=end_year,
            verbose=False)
    elif dataset == Dataset.BLEND:
        (ice_extent_df,
         name,
         xlabel,
         ylabel,
         indep,
         dep) = dataaccess.arctic_ice_extent_blend(
            month=month,
            start_year=start_year,
            end_year=end_year,
            verbose=False)
    elif dataset == Dataset.SIBT1850:
        (ice_extent_df,
         name,
         xlabel,
         ylabel,
         indep,
         dep) = dataaccess.arctic_ice_extent_sibt1850_midmonth(
            month=month,
            start_year=start_year,
            end_year=end_year,
            verbose=False)

    return (ice_extent_df, name, xlabel, ylabel, indep, dep)


FetchedData = namedtuple("FetchedData", ["data_df",
                                         "name",
                                         "xlabel",
                                         "ylabel",
                                         "indep",
                                         "dep"])


def fetch_all_months(dataset,
                     months=None,
                     start_year=None,
                     end_year=None):

    data_map = {}

    month_set = Month
    if months is not None:
        month_set = months

    for month in month_set:

        (ice_extent_df,
         name,
         xlabel,
         ylabel,
         indep,
         dep) = fetch_month(dataset=dataset,
                            month=month,
                            start_year=start_year,
                            end_year=end_year)

        # keep this?
#        ice_extent_df.columns = [month.name]

        fetched_data = FetchedData(data_df=ice_extent_df,
                                   name=name,
                                   xlabel=xlabel,
                                   ylabel=ylabel,
                                   indep=indep,
                                   dep=dep)

        data_map[month] = fetched_data

    return data_map
