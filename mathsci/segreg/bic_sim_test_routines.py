"""
Helps notebook BIC sim tests.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

from collections import defaultdict
import multiprocessing
from multiprocessing import Pool

import numpy as np
import pandas as pd

from mathsci.utilities import text_util

from mathsci.segreg import models
from mathsci.segreg.model import Model


def _bic_selects(wts_df):
    bic_selects = wts_df.bic_model.values

    tot = len(bic_selects)
    num_ols = len(bic_selects[bic_selects == Model.OLS])
    num_one_bkpt = len(bic_selects[bic_selects == Model.ONE_BKPT])
    num_two_bkpt = len(bic_selects[bic_selects == Model.TWO_BKPT])
    assert(num_ols + num_one_bkpt + num_two_bkpt == tot)

    if tot == 0:
        frac_ols = np.nan
        frac_one_bkpt = np.nan
        frac_two_bkpt = np.nan
    else:
        frac_ols = num_ols / tot
        frac_one_bkpt = num_one_bkpt / tot
        frac_two_bkpt = num_two_bkpt / tot

    return (frac_ols, frac_one_bkpt, frac_two_bkpt)


def _bic_selects_str(wts_df):

    (frac_ols, frac_one_bkpt, frac_two_bkpt) = _bic_selects(wts_df)

    frac_str = ("[" +
                text_util.format_num(frac_ols, num_decimal=2) +
                ", " +
                text_util.format_num(frac_one_bkpt, num_decimal=2) +
                ", " +
                text_util.format_num(frac_two_bkpt, num_decimal=2) +
                "]")

    return frac_str


def hist_title(wts_df, num_data):
    frac_str = _bic_selects_str(wts_df)
    title = "N=" + str(num_data) + "        " + frac_str
    return title


################################################################################
# Good code below here
################################################################################


# given truth func, resid type, create sims, then for each sim
# fit models, compute bic wts and chosen model

def compute_rmse(predictions_df,
                 truth_prediction,
                 remove_outliers=False):

    if remove_outliers:
        lower_quartile = predictions_df.quantile([0.25])
        upper_quartile = predictions_df.quantile([0.75])
        # need same index or cannot subtract them
        lower_quartile.index = [0]
        upper_quartile.index = [0]

        iqr = upper_quartile - lower_quartile
        outlier = iqr * 3.0 + upper_quartile
        outlier_vals = outlier.to_numpy().ravel()

        num = len(predictions_df)
        outlier_df = pd.DataFrame()
        for i, colname in enumerate(outlier.columns):
            outlier_df[colname] = np.repeat(outlier_vals[i], num)

        predictions_df = predictions_df[predictions_df.le(outlier_df)]

    predictions_errors = predictions_df - truth_prediction
    predictions_errors = predictions_errors ** 2

    rmse_df = pd.DataFrame(np.sqrt(predictions_errors.mean())).T

    return rmse_df


def _bic_predictions(bic_type_to_wts_df, predictions_df):
    """
    Collects the predictions corresponding to the BIC-chosen model.

    Parameters
    ----------
    bic_type_to_wts_df: dict
        keys: bic_type
        values: weights pandas DataFrame
    predictions_df: pandas DataFrame
    """

    bic_pred_df = pd.DataFrame()

    model_col_str = "bic_model"

    for bic_type, wts_df in bic_type_to_wts_df.items():

        curr_model_predictions = []
        for i, model in enumerate(wts_df[model_col_str]):

            curr_prediction = predictions_df.loc[i, model.display]
            curr_model_predictions.append(curr_prediction)

        bic_pred_df[bic_type.name] = curr_model_predictions

    return bic_pred_df


def _add_score(bic_sim_summary_df, truth_model):
    """
    Adds a ``score`` column to each row of the dataframe.

    Returns
    -------
    dict
        keys: bic_types
        values: pandas DataFrame
            columns: [OLS, ONE_BKPT, TWO_BKPT, Score]
            rows: [frac_significant, frac_chosen]
            The ``Score`` is the fraction correctly chosen minus the fraction
            incorrectly chosen.  It ranges between -1 and 1.  A score of one
            indicates the correct model was chosen 100% of the time.  A score
            of minus one indicates the correct model was never chosen.    

    """
    score_str = "score"

    truth_significant = bic_sim_summary_df.loc[:, truth_model.display]
    tot_significant = bic_sim_summary_df.sum(axis=1)
    non_truth_significant = tot_significant - truth_significant
    score = truth_significant - non_truth_significant

    bic_sim_summary_df[score_str] = score

    return bic_sim_summary_df


def bic_sim_summary(bic_types,
                    indep,
                    deps,
                    num_end_to_skip,
                    num_between_to_skip,
                    truth_model=None,
                    skip_fit_at_boundary=True,
                    significance=0.95,
                    predict_horizon=30,
                    predict_level=0.0,
                    only_boundary=False):
    """
    Iterates over datasets.  For each dataset, fits models, and computes BICs.

    Next, computes summary for each BIC type.  One summary is the fraction of
    each model chosen (number chosen divided by number datasets).  The other
    summary is the fraction of each model where the BIC-probability of the
    chosen model is significant, meaning greater than or equal to 
    ``significance`` (eg: >= 0.95 ).

    Option to include or not fits where some parameters are at the boundary of
    parameter space.

    Returns
    -------
    dict
        keys: bic_types
        values: pandas DataFrame
            columns: [OLS, ONE_BKPT, TWO_BKPT]
            rows: [frac_significant, frac_chosen]

        if ``truth_model`` is not None, adds a score column.
            The ``Score`` is the fraction correctly chosen minus the fraction
            incorrectly chosen.  It ranges between -1 and 1.  A score of one
            indicates the correct model was chosen 100% of the time.  A score
            of minus one indicates the correct model was never chosen.
    """

    (bic_type_to_wts_df,
     model_pred_df,
     model_level_pred_df) = _bic_for_sims(bic_types=bic_types,
                                          indep=indep,
                                          deps=deps,
                                          num_end_to_skip=num_end_to_skip,
                                          num_between_to_skip=num_between_to_skip,
                                          skip_fit_at_boundary=skip_fit_at_boundary,
                                          predict_horizon=predict_horizon,
                                          predict_level=predict_level,
                                          only_boundary=only_boundary)

    # begin erase
    # print(len(model_pred_df))
    # end erase

    result = {}
    for bic_type, wts_df in bic_type_to_wts_df.items():

        (frac_ols,
         frac_one_bkpt,
         frac_two_bkpt) = _bic_selects(wts_df)

        # frac amounts can be nan in the event that zero sims were processed,
        # which can happen for ``skip_fit_at_boundary`` or ``only_boundary``
        if not np.isnan(frac_ols):
            assert(abs(frac_ols + frac_one_bkpt + frac_two_bkpt - 1.0)
                   < 1.0e-14)

        summary_df = pd.DataFrame(columns=[Model.OLS.display,
                                           Model.ONE_BKPT.display,
                                           Model.TWO_BKPT.display])

        # compute num significant at 95% probability
        num_significant = (wts_df[wts_df.iloc[:, 0:3] >=
                                  significance].count().values[0:3])
        num = len(wts_df)

        if num == 0:
            frac_significant = np.nan
        else:
            frac_significant = num_significant / num

        summary_df.loc["frac_significant", :] = frac_significant

        summary_df.loc["frac_chosen"] = [frac_ols, frac_one_bkpt, frac_two_bkpt]

        summary_df.columns.name = bic_type.name

        if truth_model is not None:
            _add_score(summary_df, truth_model)

        result[bic_type] = summary_df

    # predictions
    bic_pred_df = _bic_predictions(bic_type_to_wts_df, model_pred_df)

    bic_level_pred_df = _bic_predictions(bic_type_to_wts_df, model_level_pred_df)

    return (result,
            model_pred_df,
            bic_pred_df,
            model_level_pred_df,
            bic_level_pred_df)

# TODO: add prediction ahead 30y for each fit to fit_map


def _bic_for_sims(bic_types,
                  indep,
                  deps,
                  num_end_to_skip,
                  num_between_to_skip,
                  skip_fit_at_boundary=False,
                  predict_horizon=30,
                  predict_level=0.0,
                  only_boundary=False):
    """
    Iterates over datasets.  For each dataset, fits models, and computes BICs.

    Option to include or not fits where some parameters are at the boundary of
    parameter space.

    Returns
    -------
    dict:
        keys: bic_types
        values: pandas DataFrame: cols = [bic_wt_ols, 
                                          bic_wt_1bkpt, 
                                          bic_wt_2bkpt, 
                                          chosen_model]
                                one row per sim 

    predictions_df: dict
        cols = [ols, one_bkpt, two_bkpt]
        one row per sim
    """

    # multiprocessing: we divide deps into chunks and process each separately

    num_mc_sims = len(deps)
    pool_size = multiprocessing.cpu_count()
    num_per_pool = int(np.ceil(num_mc_sims / pool_size))

    all_params = []
    for i in range(pool_size):

        begin_ind = i * num_per_pool
        end_ind = (i + 1) * num_per_pool

        curr_deps = deps[begin_ind:end_ind]

        curr_params = [indep,
                       curr_deps,
                       bic_types,
                       num_end_to_skip,
                       num_between_to_skip,
                       predict_horizon,
                       predict_level]

        all_params.append(curr_params)

    pool = Pool(processes=pool_size)
    result = pool.map(_run_estimations_wrapper, all_params)
    pool.close()
    pool.join()

    all_fit_maps = []
    # dict: values are rows that we will make into dataframe later
    all_bic_type_to_rows = defaultdict(list)

    for result_arr in result:
        fit_maps, bic_type_to_rows = result_arr

        all_fit_maps.extend(fit_maps)

        for bic_type, rows in bic_type_to_rows.items():
            all_bic_type_to_rows[bic_type].extend(rows)

    # keep track of predictions, too, for each model
    prediction_rows = []
    predict_level_rows = []

    # if we skip any rows due to ``only_boundary`` or ``at_boundary``, then
    # we also need to remove them from the result dict below; so we keep track
    # of indices here
    included_indices = []
    for i, fit_map in enumerate(all_fit_maps):

        at_boundary = (fit_map[Model.ONE_BKPT].at_boundary or
                       fit_map[Model.TWO_BKPT].at_boundary)

        ##########
        # POSSIBLY CONTINUE LOOP
        ##########
        if only_boundary and not at_boundary:
            continue

        if skip_fit_at_boundary and at_boundary:
            continue

        included_indices.append(i)

        prediction_row = [fit_map[Model.OLS].prediction,
                          fit_map[Model.ONE_BKPT].prediction,
                          fit_map[Model.TWO_BKPT].prediction]
        prediction_rows.append(prediction_row)

        predict_level_row = [fit_map[Model.OLS].level_prediction,
                             fit_map[Model.ONE_BKPT].level_prediction,
                             fit_map[Model.TWO_BKPT].level_prediction]
        predict_level_rows.append(predict_level_row)

    # build result dict
    result = {}
    for bic_type, rows in all_bic_type_to_rows.items():
        wts_df = _build_bic_sims_df(rows)

        # we may have skipped some due to ``skip_fit_at_boundary`` or
        # ``only_boundary``
        wts_df = wts_df.iloc[included_indices, :]

        result[bic_type] = wts_df

    predictions_df = _build_predictions_df(prediction_rows)
    level_predictions_df = _build_predictions_df(predict_level_rows)

    return result, predictions_df, level_predictions_df


def _run_estimations_wrapper(params):
    return _run_estimations(*params)


def _run_estimations(indep,
                     deps,
                     bic_types,
                     num_end_to_skip,
                     num_between_to_skip,
                     predict_horizon,
                     predict_level):
    fit_maps = []
    # dict: values are rows that we will make into dataframe later
    bic_type_to_rows = defaultdict(list)

    for dep in deps:
        # model estimation happens here
        fit_map = models.create_fit_map(indep=indep,
                                        dep=dep,
                                        num_end_to_skip=num_end_to_skip,
                                        num_between_to_skip=num_between_to_skip,
                                        predict_horizon=predict_horizon,
                                        predict_level=predict_level)
        fit_maps.append(fit_map)

        # compute bic stuff for each bic_type
        for bic_type in bic_types:
            (curr_wts,
             curr_bic_model) = models.bic_wts_and_choice(bic_type=bic_type,
                                                         indep=indep,
                                                         dep=dep,
                                                         fit_map=fit_map,
                                                         num_end_to_skip=num_end_to_skip,
                                                         num_between_to_skip=num_between_to_skip)
            # create one row
            curr_wts.append(curr_bic_model)

            bic_type_to_rows[bic_type].append(curr_wts)

    return fit_maps, bic_type_to_rows


def _build_predictions_df(rows):
    cols = [Model.OLS.display,
            Model.ONE_BKPT.display,
            Model.TWO_BKPT.display]

    return pd.DataFrame(rows, columns=cols)


def _build_bic_sims_df(rows):
    #name = "_bma_wt"

    cols = [Model.OLS.display,
            Model.ONE_BKPT.display,
            Model.TWO_BKPT.display,
            "bic_model"]

    return pd.DataFrame(rows, columns=cols)
