"""
Predictions based on chosen methodology.  Considered FINAL for paper.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numpy as np
import pandas as pd

from mathsci.arctic_ice import first_ice_prediction_interval
from mathsci.arctic_ice import model_fit
from mathsci.segreg.model import Model
from mathsci.utilities import text_util


# TODO: perhaps add returning the bic weight of chosen model?
def format_bic_wt(df):
    """
    Assumes dataframe has a column named ``bic_model_wt``.
    """
    df.bic_model_wt = [text_util.format_num(x, num_decimal=2)
                       for x in df.bic_model_wt]


def _bic_model_info(est_df):
    bic_model = est_df.loc[:, "bic_model"].values[0]
    bic_model_name = bic_model.display

    bic_wts = [x for x in est_df.columns if "bma_wt" in x]

    bic_wt_col_list = [x for x in bic_wts if bic_model_name in x]
    bic_wt_col = bic_wt_col_list[0]
    bic_wt = est_df.loc[:, bic_wt_col].values[0]

    return bic_wt


def predict_omnibus(indep,
                    dep,
                    num_end_to_skip=10,
                    num_between_to_skip=10,
                    zero_ice_level=1.0,
                    seed=None,
                    num_boot_sims=10000,
                    bic_type=None,
                    model_priors=None,
                    skip_interval=False,
                    sub_bic=True):
    """
    ONLY for OLS, 1BKPT, 2BKPT.
    """
    (est_df,
     zero_ice_df) = model_fit.fit_models(indep=indep,
                                         dep=dep,
                                         num_end_to_skip=num_end_to_skip,
                                         num_between_to_skip=num_between_to_skip,
                                         zero_ice_level=zero_ice_level,
                                         seed=seed,
                                         bic_type=bic_type,
                                         model_priors=model_priors)

    bic_wt = _bic_model_info(est_df)

    bic_first_ice = zero_ice_df.loc["first_zero_ice_time", "bic"]

    # TODO: add bic_type for prediction_interval_bic

    if skip_interval:
        interval = [np.nan, np.nan]
    else:
        interval = first_ice_prediction_interval.prediction_interval_bic(
            indep=indep,
            dep=dep,
            width=95,
            num_boot_sims=num_boot_sims,
            show_plot=False,
            verbose=False,
            seed=seed,
            expanded_percentiles=True,
            return_dist=False,
            zero_ice_level=zero_ice_level,
            sub_bic=sub_bic,
            num_end_to_skip=num_end_to_skip,
            num_between_to_skip=num_between_to_skip,
            models=None,
            model_priors=model_priors,
            bic_type=bic_type)

    predict_df = pd.DataFrame({"predict_interval_left": interval[0],
                               "predict": bic_first_ice,
                               "predict_interval_right": interval[1],
                               "bic_model": est_df.bic_model.values,
                               "bic_model_wt": bic_wt},
                              index=["prediction"])
    return predict_df, est_df, zero_ice_df


def predict_poly(indep,
                 dep,
                 zero_ice_level=1.0,
                 seed=None,
                 num_boot_sims=10000,
                 bic_type=None,
                 model_priors=None,
                 skip_interval=False):
    """
    ONLY for OLS, QUAD, CUBIC.
    """
    models = [Model.OLS, Model.QUAD, Model.CUBIC]

    return predict_models(indep=indep,
                          dep=dep,
                          models=models,
                          zero_ice_level=zero_ice_level,
                          seed=seed,
                          num_boot_sims=num_boot_sims,
                          bic_type=bic_type,
                          model_priors=model_priors,
                          skip_interval=skip_interval)


def predict_models(indep,
                   dep,
                   models,
                   zero_ice_level=1.0,
                   seed=None,
                   num_boot_sims=10000,
                   bic_type=None,
                   model_priors=None,
                   skip_interval=False):

    (first_zero_time,
     zero_time,
     bic_model,
     bic_model_wt) = model_fit.fit_models_group(indep=indep,
                                                dep=dep,
                                                models=models,
                                                zero_ice_level=zero_ice_level,
                                                model_priors=model_priors)

    bic_first_ice = first_zero_time

    sub_bic = True

    if skip_interval:
        interval = [np.nan, np.nan]
    else:
        interval = first_ice_prediction_interval.prediction_interval_bic(
            indep=indep,
            dep=dep,
            width=95,
            num_boot_sims=num_boot_sims,
            show_plot=False,
            verbose=False,
            seed=seed,
            expanded_percentiles=True,
            return_dist=False,
            zero_ice_level=zero_ice_level,
            sub_bic=sub_bic,
            models=models,
            model_priors=model_priors)

    predict_df = pd.DataFrame({"predict_interval_left": interval[0],
                               "predict": bic_first_ice,
                               "predict_interval_right": interval[1],
                               "bic_model": bic_model,
                               "bic_model_wt": bic_model_wt},
                              index=["prediction"])
    return predict_df
