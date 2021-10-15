"""
Plots and tables for data for paper or otherwise.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import os

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from mathsci.constants import Month, Dataset
from mathsci.arctic_ice import data_util
from mathsci.arctic_ice.arctic_ice_constants import (
    PLOTS_TABLES_ROOTDIR)
from mathsci.utilities import text_util

text_util.format_output(4)

OUTDIR = PLOTS_TABLES_ROOTDIR


################################################################################
# Library methods
################################################################################


def hist_plots(dfs, name, ylabel):

    # remove month from name
    tokens = name.split(" ")
    tokens.pop(4)
    name = " ".join(tokens)

    f, ax = plt.subplots()

    for df in dfs:
        df.plot(ax=ax, style='-o')
        plt.title(name)
        plt.ylabel(ylabel)

    plt.legend(loc="lower left")


def data_by_month(dfs,
                  name,
                  ylabel,
                  to_index=12,
                  year_ind=None,
                  mean=False):
    """
    Parameters
    ----------
    to_index: int
        index to stop at, between 0 and 23
    """
    for df in dfs:
        df.reset_index(drop=True, inplace=True)

    comb = pd.concat(dfs, ignore_index=True, axis=1)
    comb.columns = [month.name for month in Month]

    if year_ind is not None:
        comb = comb.iloc[[year_ind], :]

    # remove month from name
    tokens = name.split(" ")
    tokens.pop(4)
    name = " ".join(tokens)

    df_to_plot = comb.T
    df_to_plot = df_to_plot.append(df_to_plot)
    df_to_plot = df_to_plot.iloc[0:to_index, :]

    if mean:
        df_to_plot = df_to_plot.mean(axis=1)

    df_to_plot.plot(legend=False, style='-o')
    plt.xticks(np.arange(len(df_to_plot)), df_to_plot.index)

    plt.title(name)
    plt.ylabel(ylabel)


################################################################################
# Plot and Table Creation
################################################################################


def plot_sii_hist_all_months(save=False):

    data_map = data_util.fetch_all_months(dataset=Dataset.NSIDC)

    dfs = []

    for month, fetched_data in data_map.items():
        ice_extent_df = fetched_data.data_df
        ice_extent_df.columns = [month.name]
        dfs.append(ice_extent_df)

        name = fetched_data.name
        ylabel = fetched_data.ylabel

    hist_plots(dfs=dfs,
               name=name,
               ylabel=ylabel)

    if save:
        plt.savefig(os.path.join(OUTDIR,
                                 "arctic_ice_series_months_nsidc.pdf"),
                    format="pdf")


def plot_sibt_hist_all_months(save=False):

    data_map = data_util.fetch_all_months(dataset=Dataset.SIBT1850)

    dfs = []

    for month, fetched_data in data_map.items():
        ice_extent_df = fetched_data.data_df
        ice_extent_df.columns = [month.name]
        dfs.append(ice_extent_df)

        name = fetched_data.name
        ylabel = fetched_data.ylabel

    hist_plots(dfs=dfs[2:8],
               name=name,
               ylabel=ylabel)

    if save:
        plt.savefig(os.path.join(OUTDIR,
                                 "arctic_ice_series_months_sibt_mar_aug.pdf"),
                    format="pdf")

    dfs_to_use = dfs[8:]
    dfs_to_use.extend(dfs[0:2])

    hist_plots(dfs=dfs_to_use,
               name=name,
               ylabel=ylabel)

    if save:
        plt.savefig(os.path.join(OUTDIR,
                                 "arctic_ice_series_months_sibt_sep_feb.pdf"),
                    format="pdf")


def plot_sibt_sii_hist_sep(save=False):

    month = Month.SEP

    (ice_extent_df_sii,
     name,
     xlabel,
     ylabel,
     indep,
     dep) = data_util.fetch_month(dataset=Dataset.NSIDC, month=month)

    (ice_extent_df_sibt,
     name,
     xlabel,
     ylabel,
     indep,
     dep) = data_util.fetch_month(dataset=Dataset.SIBT1850, month=month)

    f, ax = plt.subplots(figsize=(12, 6))

    ice_extent_df_sibt.plot(ax=ax, style='o', markersize=4)
    ice_extent_df_sii.plot(ax=ax, style='o', marker="x", markersize=8)

    # ylabels are the same for both datasets
    plt.ylabel(ylabel)

    plt.legend([Dataset.SIBT1850.name, Dataset.NSIDC.name])

    plt.title("SEP Arctic Sea Ice Extent")

    if save:
        plt.savefig(os.path.join(OUTDIR,
                                 "arctic_ice_nsidc_sibt_sep.pdf"),
                    format="pdf")


def plot_seasonality(dataset, save=False):

    data_map = data_util.fetch_all_months(dataset=dataset)

    dfs = []

    for month, fetched_data in data_map.items():
        ice_extent_df = fetched_data.data_df
        ice_extent_df.columns = [month.name]
        dfs.append(ice_extent_df)

        name = fetched_data.name
        ylabel = fetched_data.ylabel

    data_by_month(dfs=dfs,
                  name=name,
                  ylabel=ylabel,
                  to_index=18)

    if save:
        savename = "arctic_ice_by_month_" + dataset.name.lower() + ".pdf"
        plt.savefig(os.path.join(OUTDIR, savename),
                    format="pdf")


def plot_sibt_sii_blend_hist_sep_intersect(save=False):

    start_year = None
    end_year = None

    month = Month.SEP

    (ice_extent_df_nsidc,
     name_nsidc,
     xlabel,
     ylabel,
     indep,
     dep) = data_util.fetch_month(dataset=Dataset.NSIDC,
                                  month=month,
                                  start_year=start_year,
                                  end_year=end_year)

    (ice_extent_df_sibt,
     name_sibt,
     xlabel,
     ylabel,
     indep,
     dep) = data_util.fetch_month(dataset=Dataset.SIBT1850,
                                  month=month,
                                  start_year=start_year,
                                  end_year=end_year)

    (ice_extent_df_blend,
     name_blend,
     xlabel,
     ylabel,
     indep,
     dep) = data_util.fetch_month(dataset=Dataset.BLEND,
                                  month=month,
                                  start_year=start_year,
                                  end_year=end_year)

    ice_extent_df_sibt.columns = [name_sibt]
    ice_extent_df_nsidc.columns = [name_nsidc]
    ice_extent_df_blend.columns = [name_blend]

    comb1 = pd.concat([ice_extent_df_sibt, ice_extent_df_nsidc],
                      join="inner",
                      axis=1)
    comb1.plot(grid=True, style='-o')
    plt.title("Arctic Sea Ice Extent " + month.name + " " +
              Dataset.NSIDC.name + " and " + Dataset.SIBT1850.name)
    plt.ylabel(ylabel)

    if save:
        plt.savefig(os.path.join(OUTDIR,
                                 "arctic_ice_nsidc_sibt.pdf"),
                    format="pdf")

    comb2 = pd.concat([ice_extent_df_nsidc, ice_extent_df_blend],
                      join="inner",
                      axis=1)
    comb2.plot(grid=True, style='-o')
    plt.title("Arctic Sea Ice Extent " + month.name + " " +
              Dataset.NSIDC.name + " and " + Dataset.BLEND.name)

    plt.ylabel(ylabel)
    if save:
        plt.savefig(os.path.join(OUTDIR,
                                 "arctic_ice_nsidc_blend.pdf"),
                    format="pdf")

    comb3 = pd.concat([ice_extent_df_sibt, ice_extent_df_blend],
                      join="outer",
                      axis=1)
    comb3.plot(grid=True, style='-o', markersize=4)
    plt.title("Arctic Sea Ice Extent " + month.name + " " +
              Dataset.SIBT1850.name + " and " + Dataset.BLEND.name)
    plt.ylabel(ylabel)
    plt.legend(loc="lower left")

    if save:
        plt.savefig(os.path.join(OUTDIR,
                                 "arctic_ice_sibt_blend.pdf"),
                    format="pdf")
