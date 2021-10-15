"""
Produces plots and tables for paper or otherwise.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import os

from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from mathsci.arctic_ice import dataaccess, data_util
from mathsci.arctic_ice import estim_plots
from mathsci.arctic_ice import final_predictions
from mathsci.arctic_ice import prediction_methods
from mathsci.arctic_ice.arctic_ice_constants import (
    PLOTS_TABLES_ROOTDIR,
    NUM_END_TO_SKIP,
    NUM_BETWEEN_TO_SKIP,
    ZERO_ICE_LEVEL,
    SEED,
    NUM_BOOT_SIMS,
    TYPE_OF_BIC,
    NA_REP,
    FYZI,
    ROLLING_BLEND_DIR,
    ROLLING_SIBT_DIR
)
from mathsci.constants import Month, Dataset
from mathsci.segreg.model import Model
from mathsci.segreg.models import BkptModelFitter
from mathsci.utilities import general_utilities as gu
from mathsci.utilities import text_util

text_util.format_output(0)

IMAGES_OUTDIR = PLOTS_TABLES_ROOTDIR
TABLES_OUTDIR = PLOTS_TABLES_ROOTDIR

PREDICT_NAME = "Predict"


################################################################################
# Library methods
################################################################################

def format_columns_for_display(cols):
    cols = [x.replace("_", " ") for x in cols]
    cols = [x.title() for x in cols]
    cols = [x.replace("Bic", "BIC") for x in cols]
    return cols


def change_cols(df):
    newcols = list(df.columns)
    newcols[0] = "left"
    newcols[2] = "right"
    df.columns = format_columns_for_display(newcols)


def models_compute_predict_df(ice_extent_df,
                              indep,
                              dep,
                              models,
                              data_name,
                              num_boot_sims=None,
                              model_priors=None,
                              skip_interval=False,
                              format_dates=False):
    """
    Returns only the prediction for the BIC-selected model from the group of 
    models.
    """

    if num_boot_sims is None:
        num_boot_sims = NUM_BOOT_SIMS

    predict_df = final_predictions.predict_models(indep=indep,
                                                  dep=dep,
                                                  models=models,
                                                  zero_ice_level=ZERO_ICE_LEVEL,
                                                  seed=SEED,
                                                  num_boot_sims=num_boot_sims,
                                                  bic_type=TYPE_OF_BIC,
                                                  model_priors=model_priors,
                                                  skip_interval=skip_interval)

    predict_df.index = [""]

    #predict_df["data_start"] = gu.format_year(ice_extent_df.index[0])
    #predict_df["data_end"] = gu.format_year(ice_extent_df.index[-1])

    if format_dates:
        predict_df["data_start"] = gu.format_year(ice_extent_df.index[0])
        predict_df["data_end"] = gu.format_year(ice_extent_df.index[-1])
    else:
        predict_df["data_start"] = ice_extent_df.index[0]
        predict_df["data_end"] = ice_extent_df.index[-1]

    predict_df["data_name"] = data_name
    change_cols(predict_df)
    return predict_df


def compute_predict_df(ice_extent_df,
                       indep,
                       dep,
                       data_name,
                       num_boot_sims=None,
                       model_priors=None,
                       num_end_to_skip=NUM_END_TO_SKIP,
                       num_between_to_skip=NUM_BETWEEN_TO_SKIP,
                       skip_interval=False,
                       sub_bic=True,
                       format_dates=False,
                       bic_type=TYPE_OF_BIC):

    if num_boot_sims is None:
        num_boot_sims = NUM_BOOT_SIMS

    (predict_df,
     est_df,
     zero_ice_df) = final_predictions.predict_omnibus(
        indep=indep,
        dep=dep,
        num_end_to_skip=num_end_to_skip,
        num_between_to_skip=num_between_to_skip,
        zero_ice_level=ZERO_ICE_LEVEL,
        seed=SEED,
        num_boot_sims=num_boot_sims,
        bic_type=bic_type,
        model_priors=model_priors,
        skip_interval=skip_interval,
        sub_bic=sub_bic)

    predict_df.index = [""]
    if format_dates:
        predict_df["data_start"] = gu.format_year(ice_extent_df.index[0])
        predict_df["data_end"] = gu.format_year(ice_extent_df.index[-1])
    else:
        predict_df["data_start"] = ice_extent_df.index[0]
        predict_df["data_end"] = ice_extent_df.index[-1]
    predict_df["data_name"] = data_name
    change_cols(predict_df)
    return predict_df


def predictions_all_months(dataset,
                           start_year=None,
                           end_year=None,
                           month_to_priors=None,
                           skip_interval=False,
                           months=None,
                           verbose=False,
                           sub_bic=True,
                           format_dates=False,
                           models=None):

    dfs = []
    name = None
    ylabel = None

    month_set = Month
    if months is not None:
        month_set = months

    for month in month_set:

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

        # remove month from name
        tokens = name.split(" ")
        tokens.pop(3)
        name = " ".join(tokens)

        data_name = name.split(" ")[-2]

        model_priors = None

        if month_to_priors is not None:

            model_priors = month_to_priors.get(month)
            if model_priors is not None and verbose:
                print()
                print(month.name + " model_priors: ", model_priors)
                print()

        if models is not None:
            predict_df = models_compute_predict_df(ice_extent_df=ice_extent_df,
                                                   indep=indep,
                                                   dep=dep,
                                                   models=models,
                                                   data_name=data_name,
                                                   model_priors=model_priors,
                                                   skip_interval=skip_interval,
                                                   format_dates=format_dates)
        else:
            predict_df = compute_predict_df(ice_extent_df=ice_extent_df,
                                            indep=indep,
                                            dep=dep,
                                            data_name=data_name,
                                            model_priors=model_priors,
                                            skip_interval=skip_interval,
                                            sub_bic=sub_bic,
                                            format_dates=format_dates)

        dfs.append(predict_df)

#         except:
#             print("cannot fetch for: ", month, " ", end_year)
#             dfs.append(pd.DataFrame({"predict" : np.nan,
#                                      "bic_model_wt" : np.nan}, index=[""]))

    comb = pd.concat(dfs, join="inner", axis=0)
    comb.index = [x.name for x in month_set]

    comb.loc[:, "BIC Model Wt"] = [text_util.format_num(x, num_decimal=2)
                                   for x in comb.loc[:, "BIC Model Wt"]]

# final_predictions.format_bic_wt(comb)

    return comb


def plot_predictions(predict_df,
                     name,
                     to_index=18,
                     ax=None,
                     plot_interval=False,
                     title_extra="",
                     skip_title=False,
                     formatted_dates=True):

    double = gu.extend_dataframe(predict_df,
                                 to_index=to_index)

    # original repeated index messes up fill_between
    foo = double.reset_index(drop=True)
    #####foo.iloc[:, [1]].plot(grid=True, style='-o', ax=ax);
    foo.loc[:, [PREDICT_NAME]].plot(grid=True, style='-o', ax=ax)

    plt.xticks(np.arange(len(double)), double.index)

    if plot_interval:
        plt.fill_between(foo.index,
                         foo.iloc[:, 0],
                         foo.iloc[:, 2],
                         alpha=0.2)

    plt.ylabel(FYZI)

    if plot_interval:
        mid_title = "95% Prediction Intervals"
    else:
        mid_title = ""

    if not skip_title:

        if formatted_dates:
            # if we formatted dates, this is a string
            data_start = predict_df.loc[:, "Data Start"]
            data_start = data_start[0]
        else:
            data_start = data_start.year

        title = (FYZI + " Predictions    " + name +
                 "\n" + mid_title + " " +
                 title_extra + "    "
                 "Data Start: " + data_start +
                 "    Extent: " + str(int(ZERO_ICE_LEVEL)))

        plt.title(title)


def plot_predictions2(predict_df,
                      to_index=18,
                      ax=None):

    double = gu.extend_dataframe(predict_df,
                                 to_index=to_index)

    # original repeated index messes up fill_between
    foo = double.reset_index(drop=True)

    ax = foo.plot(grid=True, style='-o', ax=ax)

    plt.xticks(np.arange(len(double)), double.index)

    return ax

# DUPE ALERT: put this in graphing_util


def plot_series(df, month, markers=True, ax=None, legend=True):
    if markers:
        ax = df.plot(style='-o', markersize=3, ax=ax, legend=legend)
    else:
        ax = df.plot(ax=ax, legend=legend)

    gu.date_ticks(ax=ax, num_years=5, month=month, date_format='%Y')

    plt.xticks(rotation=67.5)
    plt.xlabel("Start Year")

#    ax.set_xticks(rotation=67.5);
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=67.5)
    #ax.set_xlabel("start year");

    # ax.yaxis.set_major_locator(plt.MaxNLocator(8))

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return ax


def plot_years(dfs, formatted_dates=True):
    """
    Same dataset, but different end years.
    """

    if formatted_dates:
        end_years = [x.loc[:, "Data End"][0] for x in dfs]
    else:
        end_years = [x.loc[:, "Data End"][0].year for x in dfs]
    dataset_name = dfs[0].loc[:, "Data Name"][0]

    fig, ax = plt.subplots()

    interval_ind = 0
    plot_predictions(dfs[interval_ind],
                     name=dataset_name,
                     plot_interval=True,
                     ax=ax,
                     title_extra="for " + str(end_years[interval_ind]))

    for df in dfs[1:]:
        plot_predictions(df,
                         name=dataset_name,
                         ax=ax,
                         skip_title=True)

    ax.yaxis.set_major_locator(MaxNLocator(15))

    legend = ["data end " + str(x) for x in end_years]
    plt.legend(legend, loc="lower right")


def plot_datasets(dfs):
    """
    Same end year, different datasets.
    """

    start_years = [x.loc[:, "data_start"][0].year for x in dfs]
    end_year = dfs[0].loc[:, "data_end"][0].year
    dataset_names = [x.loc[:, "data_name"][0] for x in dfs]

    fig, ax = plt.subplots()

#     interval_ind = 0
#     plot_predictions(dfs[interval_ind],
#                      name=dataset_name,
#                      plot_interval=True,
#                      ax=ax,
#                      title_extra="for " + str(end_years[interval_ind]))

    for df, dataset_name in zip(dfs, dataset_names):
        plot_predictions(df,
                         name=dataset_name,
                         ax=ax,
                         skip_title=True)

    ax.yaxis.set_major_locator(MaxNLocator(15))

    legend = [x + " data start " + str(y)
              for x, y in zip(dataset_names, start_years)]
    plt.legend(legend, loc="lower right")


def predictions_for_month(month, models=None, verbose=False):

    start_year = 2002

    months = [month]
    month_to_priors = None
    ###########################################################
    dataset = Dataset.BLEND
    end_years = np.arange(start_year, 2021)

    blend_dfs = []
    for end_year in end_years:
        if end_year % 10 == 0 and verbose:
            print(end_year)
        pred = predictions_all_months(dataset=dataset,
                                      end_year=end_year,
                                      month_to_priors=month_to_priors,
                                      skip_interval=True,
                                      months=months,
                                      models=models)
        pred.replace(to_replace=10000, value=np.nan, inplace=True)
        blend_dfs.append(pred)
    ###########################################################
    dataset = Dataset.SIBT1850
    end_years = np.arange(start_year, 2018)

    sibt_dfs = []
    for end_year in end_years:
        if end_year % 10 == 0 and verbose:
            print(end_year)
        pred = predictions_all_months(dataset=dataset,
                                      end_year=end_year,
                                      month_to_priors=month_to_priors,
                                      skip_interval=True,
                                      months=months,
                                      models=models)
        pred.replace(to_replace=10000, value=np.nan, inplace=True)
        sibt_dfs.append(pred)
    ###########################################################
    dataset = Dataset.NSIDC
    end_years = np.arange(2010, 2021)

    nsidc_dfs = []
    for end_year in end_years:
        if end_year % 10 == 0 and verbose:
            print(end_year)
        try:
            # if we run without priors for AUG and OCT, we get 1bkpt; so we put
            # 1bkpt for SEP for consistency
            if end_year >= 2018 and month == Month.SEP:
                if models is None:
                    month_to_priors = {month: [0.0, 1.0, 0.0]}

            pred = predictions_all_months(dataset=dataset,
                                          end_year=end_year,
                                          month_to_priors=month_to_priors,
                                          skip_interval=True,
                                          months=months,
                                          models=models)
            pred.replace(to_replace=10000, value=np.nan, inplace=True)
            nsidc_dfs.append(pred)
        except:
            print("baddie: ", end_year)
    ###########################################################
    return blend_dfs, sibt_dfs, nsidc_dfs


def create_predict_df(dfs, month):

    vals = []
    index = []

    name = None

    for df in dfs:
        name = df.loc[:, "Data Name"].values[0]

        start_year = df.loc[:, "Data Start"].values[0]
        # crud way to get year here; numpy date
        start_year = str(start_year).split("-")[0]

        name += "_" + start_year

        vals.append(df.loc[month, PREDICT_NAME])
        index.append(df.loc[month, "Data End"])

    month_df = pd.DataFrame({name: vals}, index=index)
    return month_df


def predictions(sourcedir_root, year):

    all_pred = {}
    for month in Month:
        sourcedir = os.path.join(sourcedir_root, month.name)
        predict_filename = os.path.join(sourcedir, "bic_predict.csv")

        predict_df = pd.read_csv(predict_filename, index_col=0, parse_dates=True)
        predict_df.sort_index(inplace=True, ascending=True)

        curr_predict = predict_df.loc[str(year), "predict"].values
        all_pred[month.name] = curr_predict

    predictions_df = pd.DataFrame(all_pred, index=["predict"])
    return predictions_df.T


def predict_mean(sourcedir_root, start_year=None, end_year=None):

    index = []
    mean_predictions = []
    for month in Month:
        sourcedir = os.path.join(sourcedir_root, month.name)
        predict_filename = os.path.join(sourcedir, "bic_predict.csv")

        predict_df = pd.read_csv(predict_filename, index_col=0, parse_dates=True)
        predict_df.sort_index(inplace=True, ascending=True)
        predict_df = gu.truncate(df=predict_df,
                                 start_year=start_year,
                                 end_year=end_year)

        mean_prediction = predict_df.predict.mean()
        mean_predictions.append(mean_prediction)

        index.append(month.name)

    predictions_df = pd.DataFrame({"mean_predict": mean_predictions},
                                  index=index)
    return predictions_df


def table_for_year(sourcedir_root, year):

    cols = ["left",
            "predict",
            "right",
            "bic_model",
            "bic_model_wt",
            "data_start",
            "data_end",
            "data_name"]

    index = []
    rows = []
    for month in Month:
        sourcedir = os.path.join(sourcedir_root, month.name)
        predict_filename = os.path.join(sourcedir, "bic_predict.csv")
        meta_filename = os.path.join(sourcedir, "meta.csv")

        predict_df = pd.read_csv(predict_filename, index_col=0, parse_dates=True)
        predict_df.sort_index(inplace=True, ascending=True)

        meta_df = pd.read_csv(meta_filename)

        # display(meta_df.end_year)

        curr_predict_row = predict_df.loc[str(year), :]

        # print(curr_predict_row)

        # data_start gets parsed as date, but no data_end
        data_start = curr_predict_row.index[0]
        data_end = meta_df.loc[0, "end_year"]

        data_end = data_end.split("-")[0]

        # print(data_end)

        curr_output_row = list(curr_predict_row.iloc[:, 0:5].values[0])
        curr_output_row.append(data_start)
        curr_output_row.append(data_end)
        curr_output_row.append(meta_df.dataset.values[0])

        model_name = curr_output_row[3]
        curr_output_row[3] = model_name.lower()
        # print(model_name)

        index.append(month.name)
        rows.append(curr_output_row)

    predictions_df = pd.DataFrame(rows, columns=cols, index=index)
    final_predictions.format_bic_wt(predictions_df)

    # formatting
    newcols = list(predictions_df.columns)
    predictions_df.columns = format_columns_for_display(newcols)

    predictions_df.loc[:, "Data Start"] = [gu.format_year(x) for x in
                                           predictions_df.loc[:, "Data Start"]]
#     predictions_df.loc[:, "Data End"] = [gu.format_year(x) for x in
#                                            predictions_df.loc[:, "Data End"]]

    return predictions_df

##############################################################


def pred_by_month(sourcedir_root,
                  start_year=None,
                  end_year=None):
    index = []
    median_predictions = []
    mean_predictions = []

    predictions_1850 = []
    predictions_1964 = []
    predictions_1974 = []

    dataset = None

    for month in Month:

        sourcedir = os.path.join(sourcedir_root, month.name)

        predict_filename = os.path.join(sourcedir, "bic_predict.csv")
        meta_filename = os.path.join(sourcedir, "meta.csv")

        # get dataset name
        meta_df = pd.read_csv(meta_filename)
        if month == Month.JAN:
            dataset = meta_df.dataset.values[0]

        predict_df = pd.read_csv(predict_filename,
                                 index_col=0,
                                 parse_dates=True)
        predict_df.sort_index(inplace=True, ascending=True)

        predict_df = gu.truncate(df=predict_df,
                                 start_year=start_year,
                                 end_year=end_year)

        source_start = predict_df.index[0].year
        source_end = predict_df.index[-1].year

        ind = [x for x in predict_df.index if x.year == 1850]
        if ind:
            predictions_1850.append(predict_df.loc[ind, "predict"].to_numpy()[0])
        else:
            predictions_1850 = None

        show_recent = False
        if show_recent:
            ind = [x for x in predict_df.index if x.year == 1964]
            predictions_1964.append(predict_df.loc[ind, "predict"].to_numpy()[0])

            ind = [x for x in predict_df.index if x.year == 1974]
            predictions_1974.append(predict_df.loc[ind, "predict"].to_numpy()[0])

        median_prediction = predict_df.predict.median()
        mean_prediction = predict_df.predict.mean()

        median_predictions.append(median_prediction)
        mean_predictions.append(mean_prediction)
        index.append(month.name)

    prediction_df = pd.DataFrame({"median": median_predictions,
                                  "mean": mean_predictions,
                                  "1850": predictions_1850},
                                 index=index)

    if show_recent:
        prediction_df["1964"] = predictions_1964
        prediction_df["1974"] = predictions_1974

    return prediction_df, source_start, source_end, dataset


def create_colname(name, start, end):
    result = ("Data_" + str(start) + "_to_" + str(end))
    return result


def prediction_plot(dataset,
                    month,
                    start_year=None,
                    end_year=None,
                    show_zero_ice=False,
                    verbose=False,
                    xaxis_num_ticks=None,
                    yaxis_num_ticks=None,
                    models=None,
                    marker="o",
                    scatter_color="black",
                    scatter_size=5,
                    ax=None):

    (ice_extent_df,
     name,
     xlabel,
     ylabel,
     indep,
     dep) = data_util.fetch_month(dataset=dataset,
                                  month=month,
                                  start_year=start_year,
                                  end_year=end_year)

    if models is None:
        models = [Model.OLS,
                  Model.ONE_BKPT,
                  Model.TWO_BKPT
                  ]
    # TODO: needs bic_type
    estim_plots.plot_fit(indep=indep,
                         dep=dep,
                         name=name,
                         xlabel=xlabel,
                         ylabel=ylabel,
                         num_end_to_skip=NUM_END_TO_SKIP,
                         num_between_to_skip=NUM_BETWEEN_TO_SKIP,
                         models=models,
                         show_zero_ice=show_zero_ice,
                         zero_ice_level=ZERO_ICE_LEVEL,
                         xaxis_num_ticks=xaxis_num_ticks,
                         yaxis_num_ticks=yaxis_num_ticks,
                         scatter_size=scatter_size,
                         scatter_color=scatter_color,
                         marker=marker,
                         ax=ax)


def fetch_predictions(filename):

    df = pd.read_csv(filename,
                     index_col=0,
                     parse_dates=True)
    df.sort_index(inplace=True, ascending=True)
    df.replace(prediction_methods.ZERO_ICE_INFINITY_PROXY,
               np.nan,
               inplace=True)

    return df


def plot_rolling_blend(month, show_bic_choice=False, outdir=None):

    sourcedir = os.path.join(ROLLING_BLEND_DIR, month.name)

    bic_filename = "bic_predict.csv"
    ols_filename = "ols_predict.csv"
    one_bkpt_filename = "one_bkpt_predict.csv"
    two_bkpt_filename = "two_bkpt_predict.csv"

    bic_predict_filename = os.path.join(sourcedir, bic_filename)
    ols_predict_filename = os.path.join(sourcedir, ols_filename)
    one_bkpt_predict_filename = os.path.join(sourcedir, one_bkpt_filename)
    two_bkpt_predict_filename = os.path.join(sourcedir, two_bkpt_filename)
    meta_filename = os.path.join(sourcedir, "meta.csv")

    meta_df = pd.read_csv(meta_filename, parse_dates=["end_year"])
    end_year = meta_df.loc[:, ["end_year"]].iloc[0, 0]
    end_year = end_year.year

    bic_predict_df = fetch_predictions(bic_predict_filename)
    ols_predict_df = fetch_predictions(ols_predict_filename)
    one_bkpt_predict_df = fetch_predictions(one_bkpt_predict_filename)
    two_bkpt_predict_df = fetch_predictions(two_bkpt_predict_filename)

    if show_bic_choice:
        f, (ax1, ax2, ax3) = plt.subplots(3, 1,
                                          sharex=True,
                                          figsize=(12, 8))
    else:
        f, (ax1, ax3) = plt.subplots(2, 1,
                                     sharex=True,
                                     figsize=(12, 7))

    ############################

    dataset = meta_df.loc[:, ["dataset"]].iloc[0, 0]

    plt.suptitle(month.name +
                 "    Rolling Start Dates    " + dataset +
                 "    End Year: " + str(end_year))  # , y=0.93)

    plot_series(bic_predict_df.loc[:, ["predict"]],
                month=month,
                ax=ax1,
                legend=False)
    ax1.set_title("BIC Model    " + FYZI + "    (Extent=1)")

    ############################

    if show_bic_choice:
        predict_models = [Model[x].value for x in bic_predict_df.bic_model]
        predict_models_df = pd.DataFrame({"model": predict_models},
                                         index=bic_predict_df.index)

        plot_series(predict_models_df, month=month, ax=ax2, legend=False)
        ax2.set_yticks([1, 2, 3])
        ax2.set_yticklabels([Model.OLS.name,
                             Model.ONE_BKPT.name,
                             Model.TWO_BKPT.name])

        ax2.set_ylim([0.5, 3.5])
        ax2.set_title("Model")

    ############################

    maxval = max(one_bkpt_predict_df.predict.max(),
                 two_bkpt_predict_df.predict.max())

    ols_capped = []
    for val in ols_predict_df.predict.values:
        if val > maxval:
            result = np.nan
        else:
            result = val
        ols_capped.append(result)

    ols_capped_df = pd.DataFrame({"predict": ols_capped},
                                 index=ols_predict_df.index)

    plot_series(ols_capped_df.loc[:, ["predict"]],
                month=month,
                ax=ax3,
                legend=True)

    ############################

    plot_series(one_bkpt_predict_df.loc[:, ["predict"]],
                month=month,
                ax=ax3,
                legend=True)

    ############################

    plot_series(two_bkpt_predict_df.loc[:, ["predict"]],
                month=month,
                ax=ax3,
                legend=True)

    ax3.set_title("All Models    " + FYZI + "    (Extent=1)")

    markersize = 8
    color = "red"
    marker = "x"
    for currdate in bic_predict_df.index:
        model_str = bic_predict_df.loc[currdate].bic_model
        model = Model[model_str]

        if model == Model.OLS:
            ax3.plot(currdate, ols_capped_df.loc[currdate].predict,
                     'o', markersize=markersize, color=color, marker=marker)
        elif model == Model.ONE_BKPT:
            ax3.plot(currdate, one_bkpt_predict_df.loc[currdate].predict,
                     'o', markersize=markersize, color=color, marker=marker)
        elif model == Model.TWO_BKPT:
            ax3.plot(currdate, two_bkpt_predict_df.loc[currdate].predict,
                     'o', markersize=markersize, color=color, marker=marker)

    ax3.legend([Model.OLS.display,
                Model.ONE_BKPT.display,
                Model.TWO_BKPT.display,
                "BIC model"])

    outname = (month.name.lower() + "_" +
               meta_df.loc[:, ["dataset"]].iloc[0, 0] + "_" +
               "first_zero_ice_prediction_meta.pdf"
               )

    plt.tight_layout()

    if outdir is not None:
        out = os.path.join(outdir, outname)
        plt.savefig(out, format="pdf")


def myformatter(precision=0):
    def format_arr(arr):
        result = [text_util.format_num(x, num_decimal=precision) for x in arr]
        return result

    return format_arr


def fyzi(indep, dep, models):

    fyzis = []
    for model in models:
        estimator = model.estimator(num_end_to_skip=NUM_END_TO_SKIP,
                                    num_between_to_skip=NUM_BETWEEN_TO_SKIP)
        estimator.fit(indep, dep)

        (first_zero,
         zero_time) = prediction_methods.predict_first_zero(indep=indep,
                                                            dep=dep,
                                                            estimator=estimator,
                                                            num_boot_sims=NUM_BOOT_SIMS,
                                                            seed=SEED,
                                                            level=ZERO_ICE_LEVEL)

        fyzis.append(first_zero)

    df = pd.DataFrame({"fyzi": fyzis}, index=models)

    return df


################################################################################
# Plot and Table Creation
################################################################################


def plot_sii_end_years(save=False):
    dataset = Dataset.NSIDC
    end_years = [2017, 2018, 2019, 2020]

    month_to_priors = {Month.SEP: [0.0, 1.0, 0.0]}

    nsidc_dfs = []
    for end_year in end_years:
        print(end_year)
        pred = predictions_all_months(dataset=dataset,
                                      end_year=end_year,
                                      month_to_priors=month_to_priors,
                                      format_dates=True)
        nsidc_dfs.append(pred)
    plot_years(nsidc_dfs)

    if save:
        plt.savefig(os.path.join(IMAGES_OUTDIR,
                                 "predict_all_nsidc_end_years.pdf"),
                    format="pdf")


def table_sii_predict(save=False):
    dataset = Dataset.NSIDC
    nsidc_pred_orig = predictions_all_months(dataset=dataset, format_dates=True)

    nsidc_pred_orig.iloc[:, 0:3] = nsidc_pred_orig.iloc[:, 0:3].apply(np.floor)

    formatter = myformatter(precision=0)
    nsidc_pred_orig.iloc[:, 0:3] = nsidc_pred_orig.iloc[:, 0:3].apply(formatter)

    display(nsidc_pred_orig)

    out = os.path.join(TABLES_OUTDIR, "predict_all_months_nsidc_orig.tex")

    if save:
        nsidc_pred_orig.to_latex(out)


def _sii_sep_mod_df():
    dataset = Dataset.NSIDC
    month = Month.SEP
    month_to_priors = {month: [0.0, 1.0, 0.0]}

    sub_bic = True

    nsidc_pred_sep_1bkpt = predictions_all_months(dataset=dataset,
                                                  month_to_priors=month_to_priors,
                                                  sub_bic=sub_bic,
                                                  format_dates=True)
    nsidc_pred_sep_1bkpt.loc[month.name, "BIC Model Wt"] = NA_REP

    # for presentation -- use the year the forecast lands in
    nsidc_pred_sep_1bkpt.iloc[:, 0:3] = nsidc_pred_sep_1bkpt.iloc[:, 0:3].apply(np.floor)

    return nsidc_pred_sep_1bkpt


def plot_and_table_sii_predict_mod_sep(save=False):

    nsidc_pred_sep_1bkpt = _sii_sep_mod_df()

    dataset_name = nsidc_pred_sep_1bkpt.loc[:, "Data Name"][0]

    fig, ax = plt.subplots()
    plot_predictions(nsidc_pred_sep_1bkpt,
                     name=dataset_name,
                     plot_interval=True,
                     ax=ax)
    ax.get_legend().remove()

    # cannot format years before plotting or won't plot
    display(nsidc_pred_sep_1bkpt)
    formatter = myformatter(precision=0)
    nsidc_pred_sep_1bkpt.iloc[:, 0:3] = nsidc_pred_sep_1bkpt.iloc[:, 0:3].apply(formatter)

    if save:
        out = os.path.join(TABLES_OUTDIR,
                           "predict_all_months_nsidc_force_sep.tex")
        nsidc_pred_sep_1bkpt.to_latex(out)
        plt.savefig(os.path.join(IMAGES_OUTDIR,
                                 "predict_all_months_nsidc.pdf"),
                    format="pdf")


# TODO: finish fixing -- trouble with models and priors
def plot_rolling_end_date(include_sibt=False,
                          save=False,
                          models=None,
                          month=None):
    if month is None:
        month = Month.SEP

    (blend_dfs,
     sibt_dfs,
     nsidc_dfs) = predictions_for_month(month, models=models)

    blend_df = create_predict_df(blend_dfs, month.name)
    sibt_df = create_predict_df(sibt_dfs, month.name)
    nsidc_df = create_predict_df(nsidc_dfs, month.name)

    ax = blend_df.plot(style='-o', grid=True)
    nsidc_df.plot(style='-o', grid=True, ax=ax)

    if include_sibt:
        sibt_df.plot(style='-o', grid=True, ax=ax)

    name = Dataset.BLEND.name + " and " + Dataset.NSIDC.name
    title = (month.name + "    " + FYZI + " Predictions    " + name +
             "\n" +
             "Data End: varies"
             "    Extent: " + str(int(ZERO_ICE_LEVEL)))

    if models is not None:
        model_names = [x.name for x in models]
        model_names_str = " ".join(model_names)

        title += "    BIC from group: [" + model_names_str + "]"

    plt.title(title)
    plt.xlabel("Estimation End Year")
    plt.ylabel(FYZI)

    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=14, integer=True))

    if save:
        if include_sibt:
            plt.savefig(os.path.join(IMAGES_OUTDIR,
                                     "predict_blend_vary_end_years_all.pdf"),
                        format="pdf")
        else:
            plt.savefig(os.path.join(IMAGES_OUTDIR, "predict_blend_vary_end_years.pdf"),
                        format="pdf")


def table_sibt_blend_ranges(save=False):
    dataset = Dataset.BLEND
    start_year = None
    end_year = 2017

    blend_end_2017_df = predictions_all_months(dataset=dataset,
                                               start_year=start_year,
                                               end_year=end_year,
                                               format_dates=True,
                                               skip_interval=True)

    dataset = Dataset.SIBT1850

    sibt_end_2017_df = predictions_all_months(dataset=dataset,
                                              start_year=start_year,
                                              end_year=end_year,
                                              format_dates=True,
                                              skip_interval=True)

    ind = 1
    dfs = [blend_end_2017_df.iloc[:, ind],
           sibt_end_2017_df.iloc[:, ind]]
    comb = pd.concat(dfs, join="inner", axis=1)

    comb.columns = [blend_end_2017_df.iloc[0, -1],
                    sibt_end_2017_df.iloc[0, -1]]

    # round to nearest year so the ranges is correct in display
    #comb = comb.apply(np.around)
    comb = comb.apply(np.floor)

    ranges_df = pd.DataFrame(comb.max(axis=1) - comb.min(axis=1))
    ranges_df.columns = ["predict_range"]
    ranges_df = ranges_df.T
    ranges_df

    comb["Range"] = ranges_df.T

    formatter = myformatter(precision=0)
    comb = comb.apply(formatter)

    out = os.path.join(TABLES_OUTDIR, "predict_all_months_blend_sibt_2017.tex")

    if save:
        comb.to_latex(out)

    display(comb)


def plot_fyzi_and_ranges_table_sii_blend(save=False):
    year = 1850

    blend_1850_df = predictions(ROLLING_BLEND_DIR, year)
    blend_1850_name = Dataset.BLEND.name + "_" + str(year)

    cutoff_year = 1957

    blend_cutoff_df = predictions(ROLLING_BLEND_DIR, cutoff_year)
    blend_cutoff_name = Dataset.BLEND.name + "_" + str(cutoff_year)

    blend_cutoff_df = blend_cutoff_df.loc[:, ["predict"]]

    nsidc_pred_sep_1bkpt = _sii_sep_mod_df()

    nsidc_pred = nsidc_pred_sep_1bkpt.loc[:, [PREDICT_NAME]]
    nsidc_pred_name = Dataset.NSIDC.name + "_" + str(1979)

    # capitalize to have consistent column name
    blend_1850_df.columns = [PREDICT_NAME]
    blend_cutoff_df.columns = [PREDICT_NAME]

    dfs = [blend_1850_df,
           blend_cutoff_df,
           nsidc_pred]
    names = [blend_1850_name,
             blend_cutoff_name,
             nsidc_pred_name]

    comb = pd.concat(dfs,
                     join="inner", axis=1)

    comb.columns = names

    # round to nearest year so the ranges is correct in display
    comb = comb.apply(np.floor)

    fig, ax = plt.subplots()

    for df in dfs:

        plot_predictions(df,
                         name=None,
                         ax=ax,
                         skip_title=True)

    ax.yaxis.set_major_locator(MaxNLocator(15))
    plt.legend(names)

    name = Dataset.NSIDC.name + " and " + Dataset.BLEND.name
    title = (FYZI + " Predictions    " + name +
             "    Extent: " + str(int(ZERO_ICE_LEVEL)))

    plt.title(title)

    ranges_df = pd.DataFrame(comb.max(axis=1) - comb.min(axis=1))
    ranges_df.columns = ["predict_range"]
    ranges_df = ranges_df.T
    ranges_df

    comb["Range"] = ranges_df.T

    formatter = myformatter(precision=0)
    comb = comb.apply(formatter)

    display(comb)

    out = os.path.join(TABLES_OUTDIR, "predict_all_months_blend_nsidc.tex")

    if save:
        plt.savefig(os.path.join(IMAGES_OUTDIR,
                                 "predict_all_months_nsidc_blend.pdf"),
                    format="pdf")

        comb.to_latex(out)


def _create_predict_table(source_dir, year, show=True):
    predictions_df = table_for_year(source_dir, year)
    predictions_df.iloc[:, 0:3] = predictions_df.iloc[:, 0:3].apply(np.floor)

    if show:
        display(predictions_df)

    return predictions_df


def tables_predict_from_precompute(save=False):

    formatter = myformatter(precision=0)

    pred_df = _create_predict_table(source_dir=ROLLING_SIBT_DIR, year=1850)
    pred_df.iloc[:, 0:3] = pred_df.iloc[:, 0:3].apply(formatter)

    if save:
        pred_df.to_latex(os.path.join(TABLES_OUTDIR,
                                      "predict_all_months_sibt_1850.tex"))

    pred_df = _create_predict_table(source_dir=ROLLING_SIBT_DIR, year=1957)
    pred_df.iloc[:, 0:3] = pred_df.iloc[:, 0:3].apply(formatter)
    if save:
        pred_df.to_latex(os.path.join(TABLES_OUTDIR,
                                      "predict_all_months_sibt_1957.tex"))

    pred_df = _create_predict_table(source_dir=ROLLING_BLEND_DIR, year=1850)
    pred_df.iloc[:, 0:3] = pred_df.iloc[:, 0:3].apply(formatter)
    if save:
        pred_df.to_latex(os.path.join(TABLES_OUTDIR,
                                      "predict_all_months_blend_1850.tex"))

    pred_df = _create_predict_table(source_dir=ROLLING_BLEND_DIR, year=1957)
    pred_df.iloc[:, 0:3] = pred_df.iloc[:, 0:3].apply(formatter)
    if save:
        pred_df.to_latex(os.path.join(TABLES_OUTDIR,
                                      "predict_all_months_blend_1957.tex"))


def plot_mean_compare_halves_blend(save=False):
    source = ROLLING_BLEND_DIR

    cutoff_year = 1957
    (pred_blend_df_lhs,
     source_start_lhs,
     source_end_lhs,
     dataset) = pred_by_month(source, end_year=cutoff_year)

    (pred_blend_df_rhs,
     source_start_rhs,
     source_end_rhs,
     dataset) = pred_by_month(source, start_year=cutoff_year)

    colname = "mean"

    comb = pd.concat([pred_blend_df_lhs.loc[:, [colname]],
                      pred_blend_df_rhs.loc[:, [colname]]],
                     join="inner", axis=1)
    comb.columns = [create_colname(colname, source_start_lhs, source_end_lhs),
                    create_colname(colname, source_start_rhs, source_end_rhs)]

    plot_predictions2(comb)

    title = ("Mean of Rolling " + FYZI + " Predictions    "
             + dataset +
             "    Extent: " + str(int(ZERO_ICE_LEVEL)))

    plt.title(title)
    plt.ylabel(FYZI)

    plt.legend(loc="lower right")

    out = os.path.join(IMAGES_OUTDIR, "predict_all_months_blend_mean_1957.pdf")
    if save:
        plt.savefig(out, format="pdf")


def prediction_plot_sep_sii(save=False):
    month = Month.SEP
    show_zero_ice = True

    models = [Model.ONE_BKPT]
    prediction_plot(dataset=Dataset.NSIDC,
                    month=month,
                    start_year=None,
                    end_year=None,
                    show_zero_ice=show_zero_ice,
                    verbose=False,
                    xaxis_num_ticks=30,
                    yaxis_num_ticks=None,
                    models=models)

    # plt.show()

    if save:
        plt.savefig(os.path.join(IMAGES_OUTDIR, "scatter_predict_nsidc.pdf"),
                    format="pdf")


def prediction_plot_sep_blend(save=False):
    month = Month.SEP
    show_zero_ice = True

    models = [Model.ONE_BKPT]

    prediction_plot(dataset=Dataset.BLEND,
                    month=month,
                    start_year=None,
                    end_year=None,
                    show_zero_ice=show_zero_ice,
                    verbose=False,
                    xaxis_num_ticks=30,
                    models=models)
    if save:
        plt.savefig(os.path.join(IMAGES_OUTDIR,
                                 "scatter_predict_blend_1850.pdf"),
                    format="pdf")


def prediction_plot_sep_sii_blend(save=False):
    month = Month.SEP
    show_zero_ice = True

    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   sharey=True,
                                   figsize=(12, 10))
    # gridspec_kw={'height_ratios': [1, 2]})

    models = [Model.ONE_BKPT]

    prediction_plot(dataset=Dataset.BLEND,
                    month=month,
                    start_year=None,
                    end_year=None,
                    show_zero_ice=show_zero_ice,
                    verbose=False,
                    xaxis_num_ticks=30,
                    models=models,
                    scatter_color="black",
                    ax=ax2)

    prediction_plot(dataset=Dataset.NSIDC,
                    month=month,
                    start_year=None,
                    end_year=None,
                    show_zero_ice=show_zero_ice,
                    verbose=False,
                    xaxis_num_ticks=30,
                    yaxis_num_ticks=None,
                    models=models,
                    # marker="x",
                    scatter_color="black",
                    # scatter_size=20,
                    ax=ax1)

    plt.tight_layout()

    # plt.show()

    if save:
        plt.savefig(os.path.join(IMAGES_OUTDIR,
                                 "scatter_predict_nsidc_blend.pdf"),
                    format="pdf")


def rolling_predictions_backwards_blend(save=False):

    if save:
        outdir = IMAGES_OUTDIR
    else:
        outdir = None

    for month in dataaccess.Month:
        plot_rolling_blend(month=month,
                           show_bic_choice=False,
                           outdir=outdir)


def table_sii_predict_poly(save=False):
    dataset = Dataset.NSIDC

    models = [Model.OLS, Model.QUAD, Model.CUBIC]

    #models = [Model.OLS, Model.EXP]

    nsidc_pred_orig = predictions_all_months(dataset=dataset,
                                             format_dates=True,
                                             models=models)

    # print(nsidc_pred_orig)

    nsidc_pred_orig.iloc[:, 0:3] = nsidc_pred_orig.iloc[:, 0:3].apply(np.floor)

    formatter = myformatter(precision=0)
    nsidc_pred_orig.iloc[:, 0:3] = nsidc_pred_orig.iloc[:, 0:3].apply(formatter)

    display(nsidc_pred_orig)

    if save:
        nsidc_pred_orig.to_latex(os.path.join(TABLES_OUTDIR,
                                              "predict_all_months_nsidc_POLY.tex"))

# NEW, KEEP?


def table_predictions(months,
                      dataset,
                      models,
                      start_year=None,
                      end_year=None,
                      save=False,
                      bic_bkpt=False,
                      month_to_priors=None):

    dfs = []
    index_months = []
    index_list = []
    for month in months:
        (ice_extent_df,
         name,
         xlabel,
         ylabel,
         indep,
         dep) = data_util.fetch_month(dataset=dataset,
                                      month=month,
                                      start_year=start_year,
                                      end_year=end_year)

        models_to_use = list(models)

        if bic_bkpt:
            if month_to_priors is not None:

                model_priors = month_to_priors.get(month)
                if model_priors is not None:
                    print()
                    print(month.name + " model_priors: ", model_priors)
                    print()
            else:
                model_priors = None

            fitter = BkptModelFitter(indep=indep,
                                     dep=dep,
                                     bic_type=TYPE_OF_BIC,
                                     priors=model_priors,
                                     num_end_to_skip=NUM_END_TO_SKIP,
                                     num_between_to_skip=NUM_BETWEEN_TO_SKIP)

            bic_model = fitter.bic_model()

            models_to_use.insert(0, bic_model)

        df = fyzi(indep=indep,
                  dep=dep,
                  models=models_to_use)

        if bic_bkpt:
            new_index = list(df.index)
            new_index[0] = "bic_bkpt"
            df.index = new_index

        dfs.append(df)

        for dummy in df.index:
            index_months.append(month.name)
        index_list.extend(df.index)

    df = pd.concat(dfs, join="outer", axis=1)
    df.columns = [x.name for x in months]

    df = df.apply(np.floor)
    formatter = myformatter(precision=0)
    df = df.apply(formatter)

#    df = pd.concat(dfs, join="outer", axis=0)
#    index = pd.MultiIndex.from_arrays([index_months, index_list])
#    df.index = index

    df["Data Name"] = dataset.name
    df.columns.name = "Model"

    display(df)

    if save:
        if len(months) > 1:
            month_piece = ""
        else:
            month_piece = "_" + months[0].name

        piece = dataset.name.lower()
        if start_year is not None:
            piece += "_" + str(start_year)
        outname = "raw_predict" + month_piece + "_" + piece + ".tex"

        df.to_latex(os.path.join(TABLES_OUTDIR, outname))
