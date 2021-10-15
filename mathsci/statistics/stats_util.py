"""
Generally small and useful routines.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

from enum import Enum

import numpy as np
import scipy.stats


class BIC_TYPE(Enum):
    STANDARD = 1
    LWZ = 2
    HOS = 3
    BOOT = 4
    UNIF_BKPT = 5
    HOS_2 = 6
    HYBRID = 7


def expanded_quantiles(alpha, num):
    """
    See Hesterberg.
    """
    df = num - 1.0
    term = np.sqrt(num / df)
    t_quantile = scipy.stats.t.ppf(alpha, df=df)

    return scipy.stats.norm.cdf(term * t_quantile)


def create_t_variates(num_sims, num_data, df, scale):
    """
    Creates draws from Student t distribution.

    Parameters
    ----------
    df: int
        degrees of freedom
    """
    variates = []
    for i in range(num_sims):
        row = scipy.stats.t.rvs(df, scale=scale, size=num_data)
        variates.append(row)
    return np.array(variates)


def bma_weights(bic_arr, priors=None):
    """
    WARNING
    -------
    TODO: quantify this.
    Method is unstable if BIC values are too large.
    """

    if np.ndim(bic_arr) == 0:
        return 1.0

    if priors is not None:
        if abs(np.sum(priors) - 1.0) > 1.0e-12:
            raise Exception("prior probabilities do not sum to one")

    dim = np.ndim(bic_arr)

    if dim == 1:
        bic_to_use = [bic_arr]
    else:
        bic_to_use = bic_arr

    result = _bma_weights(np.array(bic_to_use), priors=priors)

    if dim == 1:
        result = result[0]

    return result


def _bma_weights(bic_arr, priors=None):

    assert(np.ndim(bic_arr) > 1)

    result = []
    for row in bic_arr:
        bma_wts = _bma_weights_impl(row, priors)
        result.append(bma_wts)
    return result


def _bma_weights_impl(bic_arr, priors=None):

    num = len(bic_arr)

    if priors is not None:
        priors = np.array(priors)
    else:
        priors = np.ones(num)

    bma_wts = []
    for i in range(num):
        curr_rel_bics = bic_arr - bic_arr[i]

        curr_denom = np.vdot(np.exp(-0.5 * curr_rel_bics), priors)

        bma_wts.append(priors[i] / curr_denom)

    return bma_wts


def aic(num_params, loglikelihood):
    k = num_params
    l = loglikelihood

    aic = 2.0 * (k - l)

    return aic


def aicc(num_params, loglikelihood, num_data):
    k = num_params
    l = loglikelihood
    n = num_data

    aic = 2.0 * (k - l)

    numer = 2.0 * k * (k + 1.0)
    denom = n - k - 1.0

    aicc = aic + numer / denom

    return aicc


class BicCalculator():

    def __init__(self, bic_type):
        self._bic_type = bic_type

    def bic(self, num_params, loglikelihood, num_data):

        if self._bic_type == BIC_TYPE.STANDARD:
            result = bic(num_params=num_params,
                         loglikelihood=loglikelihood,
                         num_data=num_data)
        elif self._bic_type == BIC_TYPE.LWZ:
            result = bic_lwz(num_params=num_params,
                             loglikelihood=loglikelihood,
                             num_data=num_data)
        elif self._bic_type == BIC_TYPE.HOS:
            result = bic_hos(num_params=num_params,
                             loglikelihood=loglikelihood,
                             num_data=num_data)
        elif self._bic_type == BIC_TYPE.HOS_2:
            result = bic_hos_two(num_params=num_params,
                                 loglikelihood=loglikelihood,
                                 num_data=num_data)
        elif self._bic_type == BIC_TYPE.HYBRID:
            result = bic_hybrid(num_params=num_params,
                                loglikelihood=loglikelihood,
                                num_data=num_data)

        else:
            raise Exception("BIC_TYPE: ", self._bic_type, " not recognized.")

        return result


def bic(num_params, loglikelihood, num_data):
    k = num_params
    l = loglikelihood
    n = num_data

    result = k * np.log(n) - 2.0 * l

    return result


def bic_lwz(num_params, loglikelihood, num_data):
    k = num_params
    l = loglikelihood
    n = num_data

    result = k * 0.299 * (np.log(n)) ** 2.1 - 2.0 * l

    return result


_CONST = np.log(40.0) / (np.log(40.0)**(2.1))


def bic_lwz_40(num_params, loglikelihood, num_data):
    k = num_params
    l = loglikelihood
    n = num_data

    result = k * _CONST * (np.log(n)) ** (2.1) - 2.0 * l

    return result


def bic_hos(num_params, loglikelihood, num_data):
    """
    WARNING: this only works for segmented regression models.

    Note that this does not modify the bic for OLS.
    """
    k = num_params
    l = loglikelihood
    n = num_data

    penalty = 0

    # 1bkpt: k equals 4 or 5 depending on whether resid variance is included
    if k == 4 or k == 5:
        penalty = 1
    # 2bkpt
    elif k == 6 or k == 7:
        penalty = 2

    result = (k + penalty) * np.log(n) - 2.0 * l

    return result


def bic_hos_two(num_params, loglikelihood, num_data):
    """
    WARNING: this only works for segmented regression models.
    """
    k = num_params
    l = loglikelihood
    n = num_data

    penalty = 0

    # 1bkpt: k equals 4 or 5 depending on whether resid variance is included
    if k == 4 or k == 5:
        penalty = 2
    # 2bkpt
    elif k == 6 or k == 7:
        penalty = 4

    result = (k + penalty) * np.log(n) - 2.0 * l

    return result


def bic_hybrid(num_params, loglikelihood, num_data):
    """
    Computes standard BIC if ``num_data`` is less than 100.  Otherwise,
    computes HOS2.

    WARNING: this only works for segmented regression models.
    """
    if num_data < 100:
        result = bic(num_params=num_params,
                     loglikelihood=loglikelihood,
                     num_data=num_data)
    else:
        result = bic_hos_two(num_params=num_params,
                             loglikelihood=loglikelihood,
                             num_data=num_data)

    return result


def rel_aic(panel_aic):
    """
    PARAMETERS
    ----------
    panel_aic: ndarray
        columns are aic (or aicc) for a given model
    """
    min_aic = np.amin(panel_aic, axis=1, keepdims=True)

    rel_aic_vals = np.exp(0.5 * (min_aic - panel_aic))

    return rel_aic_vals


def rel_bic(panel_bic):
    """
    PARAMETERS
    ----------
    panel_bic: ndarray
        columns are bic for a given model
    """
    rel_bic_vals = panel_bic - np.amin(panel_bic, axis=1, keepdims=True)

    return rel_bic_vals


def model_select_info_criterion(model_map):
    """
    Select model for AIC, AICC, BIC.

    Selects model with the minimum information criterion value.

    PARAMETERS
    ----------
    model_map: dict
        keys: Model, values: info criterion
    """
    minval = np.inf
    selected_model = None
    for model, info_criterion in model_map.items():
        if info_criterion < minval:
            selected_model = model
            minval = info_criterion
    return selected_model


def model_select_by_weight(model_map):
    """
    Select model for AIC, AICC, BIC when probability weights are given.

    Selects model with the maximum weight.

    PARAMETERS
    ----------
    model_map: dict
        keys: Model, values: float (weights)
    """
    maxval = np.NINF
    selected_model = None
    for model, weight in model_map.items():
        if weight > maxval:
            selected_model = model
            maxval = weight
    return selected_model
