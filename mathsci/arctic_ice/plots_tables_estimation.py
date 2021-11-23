"""
Plots and tables for model estimation for paper or otherwise.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numbers
import os

import numpy as np
import pandas as pd
from IPython.core.display import display
import scipy.optimize
import scipy.stats
import statsmodels.api as sm

from segreg import bootstrap
from mathsci.utilities import text_util

from mathsci.arctic_ice import prediction_methods
from mathsci.constants import Month, Dataset
from mathsci.segreg.model import Model
from mathsci.segreg.models import BkptModelFitter
from mathsci.utilities import general_utilities as gu
from mathsci.arctic_ice import data_util
from mathsci.statistics.stats_util import BicCalculator, BIC_TYPE
from mathsci.statistics import stats_util

from mathsci.arctic_ice.arctic_ice_constants import (
    PLOTS_TABLES_ROOTDIR,
    NUM_END_TO_SKIP,
    NUM_BETWEEN_TO_SKIP,
    ZERO_ICE_LEVEL,
    TYPE_OF_BIC,
    NA_REP,
    NUM_BOOT_SIMS,
    SEED
)

text_util.format_output(4)

_DEFAULT_NUM_ITER = 1000000

outdir = PLOTS_TABLES_ROOTDIR

################################################################################
# Library methods
################################################################################


def est_ar1(series):
    indep = series[0:-1]
    dep = series[1:]
    model = sm.OLS(dep, indep)
    results = model.fit()
    b = results.params[0]
    residuals = results.resid
    #rss = np.vdot(residuals, residuals)
    rss = results.ssr

    return b, rss, residuals


def format_col(values, num_decimal=3):
    """
    Assumes NA_REP not number
    """
    new_vals = []
    for val in values:
        if np.isnan(val):
            new_val = NA_REP
        elif isinstance(val, numbers.Number):
            new_val = text_util.format_num(val, num_decimal=num_decimal)
        else:
            new_val = val
        new_vals.append(new_val)

    return new_vals


def add_params_ols_1bkt(df, bic_model, func_est_params, indep):
    nan = np.nan

    if bic_model == Model.OLS:
        u = indep[0]
        v = func_est_params[0] + func_est_params[1] * u
        m1 = nan
        m2 = func_est_params[1]
    elif bic_model == Model.ONE_BKPT:
        u, v, m1, m2 = func_est_params

    df["$u$"] = u
    df["$v$"] = v
    df["$m_1$"] = m1
    df["$m_2$"] = m2


def add_params(df, bic_model, func_est_params, indep):
    nan = np.nan

    if bic_model == Model.OLS:
        u1 = nan
        v1 = nan
        u2 = indep[0]
        v2 = func_est_params[0] + func_est_params[1] * u2
        m1 = nan
        m2 = func_est_params[1]
    elif bic_model == Model.ONE_BKPT:
        u, v, m1, m2 = func_est_params
        u1 = nan
        v1 = nan
        u2 = u
        v2 = v
    elif bic_model == Model.TWO_BKPT:
        u1, v1, u2, v2, m1, m2 = func_est_params

    df["$u_1$"] = u1
    df["$v_1$"] = v1
    df["$u_2$"] = u2
    df["$v_2$"] = v2
    df["$m_1$"] = m1
    df["$m_2$"] = m2


def estimation_df(indep,
                  dep,
                  model_priors=None,
                  show_ar1_resid=False,
                  ice_extent_df=None,
                  only_ols_1bkpt=False):

    fitter = BkptModelFitter(indep=indep,
                             dep=dep,
                             bic_type=TYPE_OF_BIC,
                             priors=model_priors,
                             num_end_to_skip=NUM_END_TO_SKIP,
                             num_between_to_skip=NUM_BETWEEN_TO_SKIP)
    estimator = fitter.bic_estimator()
    bic_model = fitter.bic_model()
    bic_wt = fitter.bic_weight()

    est_params = estimator.fit(indep, dep)

    func_est_params = est_params[0:-1]

    df = pd.DataFrame(index=["stats"])
    df["BIC Model"] = bic_model.display
    #####df["BIC_Model_Weight"] = bic_wt

    if only_ols_1bkpt:
        add_params_ols_1bkt(df, bic_model, func_est_params, indep)
    else:
        add_params(df, bic_model, func_est_params, indep)

    resid = estimator.residuals
    r_sq = estimator.r_squared

    dw = sm.stats.stattools.durbin_watson(resid)

    stddev = est_params[-1]
    ##stddev = np.std(resid)

    skew = scipy.stats.skew(resid)
    kurtosis = scipy.stats.kurtosis(resid, fisher=False)

    #####df["$R^2$"] = r_sq
    #dw_name = "Durbin-Watson"
    dw_name = "DW"
    df[dw_name] = sm.stats.stattools.durbin_watson(resid)
    ##df["Stddev"] = stddev
    df["Skew"] = skew
    df["Kur"] = kurtosis

    if ice_extent_df is not None:
        df["Data Start"] = gu.format_year(ice_extent_df.index[0])
        df["Data End"] = gu.format_year(ice_extent_df.index[-1])

    if show_ar1_resid:
        #############################################
        # add AR(1) residual estimation
        #############################################
        def objective_func(params):
            #print("inside obj func: ", params)
            bkpt_func = estimator.get_func_for_params(params)
            resid1 = dep - bkpt_func(indep)
            b, rss, residuals = est_ar1(resid1)
            return rss

        result = scipy.optimize.minimize(objective_func,
                                         func_est_params,
                                         method="Nelder-Mead",
                                         # method="BFGS",
                                         options={'maxiter': 100000,
                                                  'maxfev': 100000,
                                                  'fatol': 1.0e-6,
                                                  'disp': False},
                                         tol=1.0e-6)
    #res = minimize(func, init, method="BFGS", options={'gtol': 1e-4, 'disp': True})

        mle_params = result.x
        # print("-"*50)
        # print(result)
        # print(func_est_params)
        # print(mle_params)

        if bic_model == Model.OLS:
            zero_ice_time = prediction_methods.zero_ice_date_ols(func_est_params,
                                                                 level=ZERO_ICE_LEVEL)
            zero_ice_time_mle = prediction_methods.zero_ice_date_ols(mle_params,
                                                                     level=ZERO_ICE_LEVEL)
        elif bic_model == Model.ONE_BKPT:
            zero_ice_time = prediction_methods.zero_ice_date_one_bkpt(func_est_params,
                                                                      level=ZERO_ICE_LEVEL)
            zero_ice_time_mle = prediction_methods.zero_ice_date_one_bkpt(mle_params,
                                                                          level=ZERO_ICE_LEVEL)
        elif bic_model == Model.TWO_BKPT:
            zero_ice_time = prediction_methods.zero_ice_date_two_bkpt(func_est_params,
                                                                      level=ZERO_ICE_LEVEL)
            zero_ice_time_mle = prediction_methods.zero_ice_date_two_bkpt(mle_params,
                                                                          level=ZERO_ICE_LEVEL)
        df["ZERO ICE"] = zero_ice_time
        df["ZERO ICE AR1"] = zero_ice_time_mle

    return df


def estimations_all_months(dataset,
                           start_year=None,
                           end_year=None,
                           month_to_priors=None,
                           months=None,
                           verbose=True,
                           only_ols_1bkpt=False):

    data_map = data_util.fetch_all_months(dataset=dataset,
                                          months=months,
                                          start_year=start_year,
                                          end_year=end_year)

    dfs = []
    month_set = []

    for month, fetched_data in data_map.items():

        month_set.append(month)

        name = fetched_data.name
        ice_extent_df = fetched_data.data_df
        indep = fetched_data.indep
        dep = fetched_data.dep

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

        df = estimation_df(indep=indep,
                           dep=dep,
                           model_priors=model_priors,
                           ice_extent_df=ice_extent_df,
                           only_ols_1bkpt=only_ols_1bkpt)
        dfs.append(df)

    comb = pd.concat(dfs, join="inner", axis=0)
    comb.index = [x.name for x in month_set]

    if only_ols_1bkpt:
        format_inds_three = [3, 4]
    else:
        format_inds_three = [5, 6]

    for ind in format_inds_three:
        comb.iloc[:, ind] = format_col(comb.iloc[:, ind].values, num_decimal=3)

    if only_ols_1bkpt:
        format_inds_one = [2]
    else:
        format_inds_one = [2, 4]

    for ind in format_inds_one:
        comb.iloc[:, ind] = format_col(comb.iloc[:, ind].values, num_decimal=1)

    if only_ols_1bkpt:
        format_inds_two = [1, 5, 6, 7]
    else:
        format_inds_two = [1, 3, 7, 8, 9]

    for ind in format_inds_two:
        comb.iloc[:, ind] = format_col(comb.iloc[:, ind].values, num_decimal=2)

    comb.replace("_", "\_", inplace=True, regex=True)
    comb.replace(np.nan, NA_REP, inplace=True)

    return comb


def myformatter(precision=0):
    def format_arr(arr):
        result = [text_util.format_num(x, num_decimal=precision) for x in arr]
        return result

    return format_arr


################################################################################
# conf int

# TODO: to util: this is duped
def format_row(values, num_decimal=3):
    """
    Assumes NA_REP not number
    """
    new_vals = []
    for val in values:
        if np.isnan(val):
            new_val = NA_REP
        elif isinstance(val, numbers.Number):
            new_val = text_util.format_num(val, num_decimal=num_decimal)
        else:
            new_val = val
        new_vals.append(new_val)

    return new_vals


def estimation_results(ice_extent_df,
                       name,
                       xlabel,
                       ylabel,
                       indep,
                       dep,
                       model_priors=None,
                       month=None,
                       num_iter=_DEFAULT_NUM_ITER,
                       verbose=False,
                       significance=0.05,
                       seed=None):

    fitter = BkptModelFitter(indep=indep,
                             dep=dep,
                             bic_type=TYPE_OF_BIC,
                             priors=model_priors)
    estimator = fitter.bic_estimator()
    bic_model = fitter.bic_model()

    est_params = estimator.fit(indep, dep)

    resid_stddev = est_params[-1]

    resid = estimator.residuals
    resid_df = pd.DataFrame({"resid": resid}, index=indep)

    if verbose:
        print("-" * 75)
        print()
        print(month.name, "    ", bic_model.name)

        print()
        print("R^2: ", estimator.r_squared)
        print()

    month_str = ""
    if month is not None:
        month_str = month.name

    name_to_use = bic_model.name + "    " + name + "    " + month_str

#     segmented_regression.bootstrap(indep,
#                                    dep,
#                                    estimator=estimator,
#                                    display_name=name,
#                                    save_dir=None,
#                                    resample_cases=resample_cases,
#                                    significance=significance,
#                                    num_iter=num_iter,
#                                    graph=graph,
#                                    preserve_cond_var=preserve_cond_var,
#                                    add_percentile_ci=add_percentile_ci,
#                                    add_basic_ci=add_basic_ci,
#                                    precision=6,
#                                    verbose=True,
#                                    latex=False);

    # orig put back
    # one million for paper
    #num_iter = 1000000

    graph = True
    resample_cases = False

    add_percentile_ci = True
    add_basic_ci = False

    (bca_ci_df,
     percentile_ci_df,
     basic_ci_df) = bootstrap.boot_conf_intervals(indep=indep,
                                                  dep=dep,
                                                  estimator=estimator,
                                                  display_name=name,
                                                  resample_cases=resample_cases,
                                                  significance=significance,
                                                  num_sims=num_iter,
                                                  precision=6,
                                                  verbose=False,
                                                  seed=seed)
    return bca_ci_df, bic_model


def conf_int_all_months(dataset,
                        start_year=None,
                        end_year=None,
                        month_to_priors=None,
                        months=None,
                        verbose=True,
                        num_iter=_DEFAULT_NUM_ITER,
                        significance=0.05,
                        seed=None):

    month_set = Month
    if months is not None:
        month_set = months

    data_map = data_util.fetch_all_months(dataset=dataset,
                                          months=month_set,
                                          start_year=start_year,
                                          end_year=end_year)

    bca_ci_df_map = {}
    model_map = {}

    for month, fetched_data in data_map.items():

        name = fetched_data.name

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

        ice_extent_df = fetched_data.data_df
        xlabel = fetched_data.xlabel
        ylabel = fetched_data.ylabel
        indep = fetched_data.indep
        dep = fetched_data.dep

        bca_ci_df, bic_model = estimation_results(ice_extent_df=ice_extent_df,
                                                  name=data_name,
                                                  xlabel=xlabel,
                                                  ylabel=ylabel,
                                                  indep=indep,
                                                  dep=dep,
                                                  model_priors=model_priors,
                                                  month=month,
                                                  num_iter=num_iter,
                                                  significance=significance,
                                                  seed=seed)

        # change index name to col name for better latex
        ind_name = bca_ci_df.index.name
        bca_ci_df.columns.name = ind_name
        bca_ci_df.index.name = None

        bca_ci_df_map[month] = bca_ci_df
        model_map[month] = bic_model
    return bca_ci_df_map, model_map


def info_criteria(indep,
                  dep,
                  models,
                  add_aicc=False):
    """
    Compares BIC and AIC for a set of models.  Displays raw numbers and weights.
    """
    num_data = len(indep)

    bic_calc = BicCalculator(bic_type=BIC_TYPE.STANDARD)
    bic_calc_bkpt = BicCalculator(bic_type=BIC_TYPE.HYBRID)

    #bic_calc_bkpt = bic_calc
    #####bic_calc_bkpt = BicCalculator(bic_type = BIC_TYPE.HOS)

    bics = []
    aics = []
    aiccs = []
    for model in models:

        if model in [Model.ONE_BKPT, Model.TWO_BKPT]:
            bic_calc_to_use = bic_calc_bkpt
        else:
            bic_calc_to_use = bic_calc

        estimator = model.estimator(num_end_to_skip=NUM_END_TO_SKIP,
                                    num_between_to_skip=NUM_BETWEEN_TO_SKIP)
        estimator.fit(indep, dep)

        loglikelihood = estimator.loglikelihood()
        num_params = estimator.num_params

        bic = bic_calc_to_use.bic(num_params=num_params,
                                  loglikelihood=loglikelihood,
                                  num_data=num_data)
        aic = stats_util.aic(num_params=num_params,
                             loglikelihood=loglikelihood)
        aicc = stats_util.aicc(num_params=num_params,
                               loglikelihood=loglikelihood,
                               num_data=num_data)

        bics.append(bic)
        aics.append(aic)
        aiccs.append(aicc)
    ic_df = pd.DataFrame({"BIC": bics, "AIC": aics}, index=models)
    if add_aicc:
        ic_df["AICC"] = aiccs

    wts_df = ic_df.apply(stats_util.bma_weights, axis=0)
    wts_cols = [x + " Model Wt" for x in wts_df.columns]
    wts_df.columns = wts_cols
    both = pd.concat([ic_df, wts_df], join="outer", axis=1)
    return both


################################################################################
# Plot and Table Creation
################################################################################


def table_estim_sii_all_months(save=False):
    month_to_priors = {Month.SEP: [0, 1, 0]}
    df = estimations_all_months(dataset=Dataset.NSIDC,
                                start_year=None,
                                end_year=None,
                                month_to_priors=month_to_priors,
                                months=None,
                                verbose=True,
                                only_ols_1bkpt=True)

    #formatter = myformatter(precision=2)
    #df.iloc[:, [1]] = df.iloc[:, [1]].apply(formatter)

    # Don't need this to fit page, but maybe want to be consistent with blend?
    # This changes "Data Start" to "Start".
    newcols = list(df.columns)
    for ind in [-2, -1]:
        newcols[ind] = newcols[ind].split("Data ")[1]
    df.columns = newcols

    display(df)

    if save:
        df.to_latex(os.path.join(outdir, "estim_results_all_months_nsidc.tex"),
                    escape=False)


def table_estim_blend_all_months(start_year=None, save=False):

    df = estimations_all_months(dataset=Dataset.BLEND,
                                start_year=start_year,
                                end_year=None,
                                months=None,
                                verbose=True)

    # change "Data Start" to "Start" because table too wide in latex
    newcols = list(df.columns)
    for ind in [-2, -1]:
        newcols[ind] = newcols[ind].split("Data ")[1]
    df.columns = newcols

    display(df)

    if save:
        outname = "estim_results_all_months_blend_" + str(start_year) + ".tex"
        df.to_latex(os.path.join(outdir, outname), escape=False)


def table_bca_conf_int(months,
                       dataset,
                       month_to_priors=None,
                       start_year=None,
                       end_year=None,
                       save=False,
                       verbose=False,
                       num_iter=_DEFAULT_NUM_ITER,
                       significance=0.05,
                       seed=None):

    bca_ci_df_map, model_map = conf_int_all_months(dataset=dataset,
                                                   start_year=start_year,
                                                   end_year=end_year,
                                                   month_to_priors=month_to_priors,
                                                   months=months,
                                                   verbose=verbose,
                                                   num_iter=num_iter,
                                                   significance=significance,
                                                   seed=seed)
    dfs = []
    index_months = []
    index_list = []

    for month, df in bca_ci_df_map.items():

        model = model_map[month]
        if model == Model.ONE_BKPT:
            formatter = myformatter(precision=2)
            df.iloc[[0, 1, 4], :] = df.iloc[[0, 1, 4], :].apply(formatter)
            formatter = myformatter(precision=3)
            df.iloc[[2, 3], :] = df.iloc[[2, 3], :].apply(formatter)
        elif model == Model.TWO_BKPT:
            formatter = myformatter(precision=2)
            df.iloc[[0, 1, 2, 3, 6], :] = df.iloc[[0, 1, 2, 3, 6], :].apply(formatter)
            formatter = myformatter(precision=3)
            df.iloc[[4, 5], :] = df.iloc[[4, 5], :].apply(formatter)

        dfs.append(df)
        for dummy in df.index:
            index_months.append(month.name)
        index_list.extend(df.index)

    df = pd.concat(dfs, join="outer", axis=0)
    index = pd.MultiIndex.from_arrays([index_months, index_list])
    df.index = index

    display(df)

    if save:
        if len(months) > 1:
            month_piece = ""
        else:
            month_piece = "_" + months[0].name

        piece = dataset.name.lower()
        if start_year is not None:
            piece += "_" + str(start_year)
        outname = "estim_conf_int" + month_piece + "_" + piece + ".tex"

        df.to_latex(os.path.join(outdir, outname))


def table_info_criteria(months,
                        dataset,
                        models,
                        start_year=None,
                        end_year=None,
                        save=False,
                        bic_bkpt=False,
                        month_to_priors=None,
                        formatting=True):

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

        df = info_criteria(indep=indep,
                           dep=dep,
                           models=models_to_use)
        #df.index.name = "Model"

        dfs.append(df)

        for dummy in df.index:
            index_months.append(month.name)
        index_list.extend(df.index)

    df = pd.concat(dfs, join="outer", axis=0)
    index = pd.MultiIndex.from_arrays([index_months, index_list])
    df.index = index

    if formatting:
        formatter = myformatter(precision=2)
        df = df.apply(formatter)

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
        outname = "info_criteria" + month_piece + "_" + piece + ".tex"

        df.to_latex(os.path.join(outdir, outname))
