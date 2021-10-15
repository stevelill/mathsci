"""
Analyze model fits and zero ice times.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numpy as np
import pandas as pd

from mathsci.arctic_ice import prediction_methods
from mathsci.segreg.model import Model
from mathsci.segreg.models import BkptModelFitter, MultiModelEstimator


def _create_zero_ice_df(zero_ice_level,
                        ols_zero_ice_time,
                        one_bkpt_zero_ice_time,
                        two_bkpt_zero_ice_time,
                        bkpt_model_fitter,
                        bic_model,
                        bic_lwz_model,
                        bic_hos_model,
                        aicc_model,
                        index_name=None):
    if index_name is None:
        index_name = "zero_ice_time"

    zero_ice_df = pd.DataFrame(index=[index_name])
    zero_ice_df["level"] = zero_ice_level
    zero_ice_df[Model.OLS.display] = ols_zero_ice_time
    zero_ice_df[Model.ONE_BKPT.display] = one_bkpt_zero_ice_time
    zero_ice_df[Model.TWO_BKPT.display] = two_bkpt_zero_ice_time

    zero_ice_vals = [ols_zero_ice_time,
                     one_bkpt_zero_ice_time,
                     two_bkpt_zero_ice_time]

    bma_wts = [bkpt_model_fitter.ols_bma_wt,
               bkpt_model_fitter.one_bkpt_bma_wt,
               bkpt_model_fitter.two_bkpt_bma_wt]

    zero_ice_df["bma"] = np.vdot(np.array(bma_wts), np.array(zero_ice_vals))

    bic_model_zero_ice_time = zero_ice_df.loc[:, bic_model.display].to_numpy()[0]
    zero_ice_df["bic"] = bic_model_zero_ice_time

    bic_lwz_model_zero_ice_time = zero_ice_df.loc[:, bic_lwz_model.display].to_numpy()[0]
    zero_ice_df["bic_lwz"] = bic_lwz_model_zero_ice_time

    bic_hos_model_zero_ice_time = zero_ice_df.loc[:, bic_hos_model.display].to_numpy()[0]
    zero_ice_df["bic_hos"] = bic_hos_model_zero_ice_time

    # EXPERIMENTAL
    throw_out_worst_bma_wts = bkpt_model_fitter.throw_out_worst_bma_wts

    zero_ice_df["remove_worst_bma"] = np.vdot(np.array(throw_out_worst_bma_wts),
                                              np.array(zero_ice_vals))
    # EXPERIMENTAL

    aicc_model_zero_ice_time = zero_ice_df.loc[:, aicc_model.display].to_numpy()[0]
    zero_ice_df["aicc_model"] = aicc_model_zero_ice_time

    return zero_ice_df


def fit_models(indep,
               dep,
               num_end_to_skip=10,
               num_between_to_skip=10,
               zero_ice_level=0.0,
               seed=None,
               num_boot_sims=10000,
               bic_type=None,
               model_priors=None):
    """
    ONLY for OLS, 1BKPT, 2BKPT.

    TODO: add param for list of models.

    Returns
    -------
    estim_df: pandas DataFrame
        Has one row, multiple columns with fitted values for bkpt models as well
        as BIC, AICC.
    zero_ice_df: pandas DataFrame
        Has two rows: first row for first zero ice time, second for zero ice 
        time.
    """

    fitter = BkptModelFitter(indep=indep,
                             dep=dep,
                             num_end_to_skip=num_end_to_skip,
                             num_between_to_skip=num_between_to_skip,
                             bic_type=bic_type,
                             priors=model_priors)

    est_df = fitter.to_dataframe()

    (ols_first_zero,
     ols_zero_ice_time) = prediction_methods.predict_first_zero(indep=indep,
                                                                dep=dep,
                                                                estimator=fitter.ols_estimator,
                                                                num_boot_sims=num_boot_sims,
                                                                seed=seed,
                                                                level=zero_ice_level)
    (one_bkpt_first_zero,
     one_bkpt_zero_ice_time) = prediction_methods.predict_first_zero(indep=indep,
                                                                     dep=dep,
                                                                     estimator=fitter.one_bkpt_estimator,
                                                                     num_boot_sims=num_boot_sims,
                                                                     seed=seed,
                                                                     level=zero_ice_level)
    (two_bkpt_first_zero,
     two_bkpt_zero_ice_time) = prediction_methods.predict_first_zero(indep=indep,
                                                                     dep=dep,
                                                                     estimator=fitter.two_bkpt_estimator,
                                                                     num_boot_sims=num_boot_sims,
                                                                     seed=seed,
                                                                     level=zero_ice_level)

    bic_model = fitter.bic_model()
    bic_lwz_model = fitter.bic_lwz_model()
    bic_hos_model = fitter.bic_hos_model()
    aicc_model = fitter.aicc_model()

    first_zero_ice_df = _create_zero_ice_df(zero_ice_level=zero_ice_level,
                                            ols_zero_ice_time=ols_first_zero,
                                            one_bkpt_zero_ice_time=one_bkpt_first_zero,
                                            two_bkpt_zero_ice_time=two_bkpt_first_zero,
                                            bkpt_model_fitter=fitter,
                                            bic_model=bic_model,
                                            bic_lwz_model=bic_lwz_model,
                                            bic_hos_model=bic_hos_model,
                                            aicc_model=aicc_model,
                                            index_name="first_zero_ice_time")

    zero_ice_df = _create_zero_ice_df(zero_ice_level=zero_ice_level,
                                      ols_zero_ice_time=ols_zero_ice_time,
                                      one_bkpt_zero_ice_time=one_bkpt_zero_ice_time,
                                      two_bkpt_zero_ice_time=two_bkpt_zero_ice_time,
                                      bkpt_model_fitter=fitter,
                                      bic_model=bic_model,
                                      bic_lwz_model=bic_lwz_model,
                                      bic_hos_model=bic_hos_model,
                                      aicc_model=aicc_model,
                                      index_name="zero_ice_time")

    predict_df = pd.concat([first_zero_ice_df, zero_ice_df], join="inner", axis=0)

    return est_df, predict_df


def fit_models_group(indep,
                     dep,
                     models,
                     zero_ice_level=0.0,
                     seed=None,
                     num_boot_sims=10000,
                     bic_type=None,
                     model_priors=None):
    """
    Returns
    -------
    estim_df: pandas DataFrame
        Has one row, multiple columns with fitted values for bkpt models as well
        as BIC, AICC.
    zero_ice_df: pandas DataFrame
        Has two rows: first row for first zero ice time, second for zero ice 
        time.
    """

    zero_map = {}

    for model in models:
        estimator = model.estimator()

        (first_zero,
         zero_time) = prediction_methods.predict_first_zero(indep=indep,
                                                            dep=dep,
                                                            estimator=estimator,
                                                            num_boot_sims=num_boot_sims,
                                                            seed=seed,
                                                            level=zero_ice_level)

        zero_data = [first_zero, zero_time]
        zero_map[model] = zero_data

    mme = MultiModelEstimator(models=models, priors=model_priors)
    mme.fit(indep=indep, dep=dep)

    bic_model = mme.bic_model()

    bic_zero_data = zero_map[bic_model]
    first_zero_time = bic_zero_data[0]
    zero_time = bic_zero_data[1]

    bic_wts = mme.bma_weights_df()

    bic_model_wt_col = [x for x in bic_wts.columns
                        if bic_model.name.lower() in x]

    bic_model_wt_col = bic_model_wt_col[0]

    bic_model_wt = bic_wts.loc[:, bic_model_wt_col].values[0]

    return first_zero_time, zero_time, bic_model, bic_model_wt
