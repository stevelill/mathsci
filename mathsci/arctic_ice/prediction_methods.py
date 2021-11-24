"""
General routines to predict zeros for segreg extrapolation.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numpy as np
from scipy import optimize

from mathsci.segreg.model import ExponentialEstimator
from mathsci.segreg.model import (QuadRegEstimator,
                                  CubicRegEstimator,
                                  GompertzEstimator)
from mathsci.segreg.models import MultiModelEstimator
from segreg.model import OLSRegressionEstimator
from segreg.model import OneBkptSegRegEstimator
from segreg.model import TwoBkptSegRegEstimator

try:
    from numba import jit
except ImportError as e:
    from segreg.mockjit import jit


ZERO_ICE_INFINITY_PROXY = 10000.0

ZERO_ICE_TIME_THRESHOLD = 500.0

_SLOPE_THRESHOLD = -1.0e-4

_NUM_YEARS_PAST_ZERO_ICE = 20


################################################################################
# ZERO ICE METHODS
################################################################################


def _zero_for_ols(params, level=0.0):
    intercept, slope = params

    result = ZERO_ICE_INFINITY_PROXY

    if slope < _SLOPE_THRESHOLD:
        zero = (level - intercept) / slope

        if zero >= ZERO_ICE_INFINITY_PROXY:
            result = ZERO_ICE_INFINITY_PROXY
        else:
            result = zero

    return result


def zero_ice_date_ols(functional_params, level=0.0):

    dim = np.ndim(functional_params)

    if dim == 1:
        return _zero_for_ols(functional_params, level=level)
    elif dim == 2:
        zeros = []
        for params in functional_params:
            zeros.append(_zero_for_ols(params, level=level))
        return np.array(zeros)


def _zero_for_second_line_one_bkpt(params, level=0.0):

    u, v, m1, m2 = params

    result = ZERO_ICE_INFINITY_PROXY

    if m2 < _SLOPE_THRESHOLD:
        zero = u + (level - v) / m2

        if zero >= ZERO_ICE_INFINITY_PROXY:
            result = ZERO_ICE_INFINITY_PROXY
        else:
            result = zero

    return result


def zero_ice_date_one_bkpt(functional_params, level=0.0):
    """
    PARAMETERS
    ----------
    functional_params: array-like
        [u, v, m1, m2]
    level: float
    """
    dim = np.ndim(functional_params)

    if dim == 1:
        return _zero_for_second_line_one_bkpt(functional_params, level=level)
    elif dim == 2:
        zeros = []
        for params in functional_params:
            zeros.append(_zero_for_second_line_one_bkpt(params, level=level))
        return np.array(zeros)


def _zero_for_second_line_two_bkpt(params, level=0.0):

    u1, v1, u2, v2, m1, m2 = params

    result = ZERO_ICE_INFINITY_PROXY

    if m2 < _SLOPE_THRESHOLD:
        zero = u2 + (level - v2) / m2

        if zero >= ZERO_ICE_INFINITY_PROXY:
            result = ZERO_ICE_INFINITY_PROXY
        else:
            result = zero

    return result


def zero_ice_date_two_bkpt(functional_params, level=0.0):

    dim = np.ndim(functional_params)

    if dim == 1:
        return _zero_for_second_line_two_bkpt(functional_params, level=level)
    elif dim == 2:
        zeros = []
        for params in functional_params:
            zeros.append(_zero_for_second_line_two_bkpt(params, level=level))
        return np.array(zeros)


def quad_zero(a, b, c):
    """
    ax^2 + bx + c
    """
    return (-b - np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


def _zero_ice_quad(functional_params, level=0.0):
    b0 = functional_params[0]
    b1 = functional_params[1]
    b2 = functional_params[2]

    # a,b,c of quadratic function above
    a = b2
    b = b1
    c = b0 - level

    term = b * b - 4.0 * a * c
    if term < 0.0:
        return ZERO_ICE_INFINITY_PROXY
        # return np.nan

    return quad_zero(a=a, b=b, c=c)


def zero_ice_date_quad(functional_params, level=0.0):
    dim = np.ndim(functional_params)

    if dim == 1:
        return _zero_ice_quad(functional_params, level=level)
    elif dim == 2:
        zeros = []
        for params in functional_params:
            zeros.append(_zero_ice_quad(params, level=level))
        return np.array(zeros)


def cubic_zero(a, b, c, d, left_bracket=None):
    """
    ax^3 + bx^2 + cx + d
    """
    if left_bracket is None:
        left_bracket = 2021

    def func(x):
        return a * x * x * x + b * x * x + c * x + d

    # begin erase
    if a > 0:
        print()
        print("CUBIC coefficient of x^3 is positive! a: ", a)
        print()
    # end erase

    if a > 0:
        result = ZERO_ICE_INFINITY_PROXY
    else:
        result = optimize.brentq(func, left_bracket, left_bracket + 1000)

    return result


def _zero_ice_cubic(functional_params, level=0.0, left_bracket=None):
    b0 = functional_params[0]
    b1 = functional_params[1]
    b2 = functional_params[2]
    b3 = functional_params[3]

    # a,b,c of quadratic function above
    a = b3
    b = b2
    c = b1
    d = b0 - level

    return cubic_zero(a=a, b=b, c=c, d=d, left_bracket=left_bracket)


def zero_ice_date_cubic(functional_params, level=0.0, left_bracket=None):
    dim = np.ndim(functional_params)

    if dim == 1:
        return _zero_ice_cubic(functional_params,
                               level=level,
                               left_bracket=left_bracket)
    elif dim == 2:
        zeros = []
        for params in functional_params:
            zeros.append(_zero_ice_cubic(params,
                                         level=level,
                                         left_bracket=left_bracket))
        return np.array(zeros)


def _zero_ice_exp(functional_params, shift, level=0.0):
    a = functional_params[0]
    b = functional_params[1]
    c = functional_params[2]

    condition = (c < 0 and b < 0) or (c > 0 and b > 0)

    # begin erase
    if condition:
        print()
        print("infty predict: ", b, c)
        print()
    # end erase

    if condition:
        result = ZERO_ICE_INFINITY_PROXY
    else:
        result = np.log((level - a) / b) / c

    return result + shift


def zero_ice_date_exp(functional_params, shift, level=0.0):
    dim = np.ndim(functional_params)

    if dim == 1:
        return _zero_ice_exp(functional_params, shift=shift, level=level)
    elif dim == 2:
        zeros = []
        for params in functional_params:
            zeros.append(_zero_ice_exp(params, shift=shift, level=level))
        return np.array(zeros)


def _zero_ice_gomp(functional_params, shift, level=0.0):
    a = functional_params[0]
    b = functional_params[1]
    c = functional_params[2]

    #condition = (c < 0 and b < 0) or (c > 0 and b > 0)

    # TODO: condition
    condition = False

    # begin erase
    if condition:
        print()
        print("infty predict: ", b, c)
        print()
    # end erase

    if condition:
        result = ZERO_ICE_INFINITY_PROXY
    else:
        term = -1.0 * np.log(level / a)
        result = b + c * np.log(term)

    return result + shift


def zero_ice_date_gomp(functional_params, shift, level=0.0):
    dim = np.ndim(functional_params)

    if dim == 1:
        return _zero_ice_gomp(functional_params=functional_params,
                              shift=shift,
                              level=level)
    elif dim == 2:
        zeros = []
        for params in functional_params:
            zeros.append(_zero_ice_gomp(functional_params=params,
                                        shift=shift,
                                        level=level))
        return np.array(zeros)


################################################################################
# FIRST ZERO ICE METHODS
################################################################################

def value_is_far(val, data_end):
    return val > data_end + ZERO_ICE_TIME_THRESHOLD


@jit(nopython=True)
def first_zero(domain, data, level=0.0):
    """
    Finds element of domain corresponding to index of first negative value in 
    data.

    PARAMETERS
    ----------
    domain: array-like
    data: array-like
    """
    assert(len(domain) == len(data))

    neg_vals_inds = np.where(data < level)

    negs_arr = neg_vals_inds[0]

    if negs_arr.size == 0:
        print()
        print(negs_arr)
        print()
        raise Exception("no fyzi!!!")

    first_neg = domain[neg_vals_inds[0][0]]
    return first_neg


@jit(nopython=True)
def find_first_zeros(domain, data, level=0.0):
    """
    Finds element of domain corresponding to index of first 
    negative value in each row of the data.

    PARAMETERS
    ----------
    domain: array-like
        in our context, domain would be some
        consecutive number of years (into the future)
    data_fitted: ndarray shape (num_sims, len(domain))
        if we let f(x) denote fitted model function, then
        data_fitted would be f(domain) + residuals; ie each
        row is a realization from the model
    """
    assert(len(domain) == data.shape[1])

    num_sims = len(data)

    first_zeros = []

    for i in range(num_sims):
        first_neg = first_zero(domain, data[i], level=level)
        first_zeros.append(first_neg)

    return np.array(first_zeros)


@jit(nopython=True)
def boot_first_zero_dist(domain,
                         fitted_domain,
                         resid_stddev,
                         num_sims=1000,
                         seed=None,
                         resids=None,
                         level=0.0):
    """
    TODO: fix up num_sims and resids

    TODO: resolve near dupe to boot_first_zero_draws

    Creates a distribution of draws from the first element of domain where
    the given model produces a value <= 0.

    Idea is that residuals are from a fit to data: indep, dep.  The parameter
    domain is thought of as model independent values beyond the end of indep.

    Eg: arctic ice, domain is years.

    Residual error drawn from normal distribution unless ``resids`` are given.
    """
    if seed is not None:
        np.random.seed(seed)

    num_data = len(domain)

    if resids is None:
        resids = resid_stddev * np.random.randn(num_sims, num_data)

    # create sim data
    sim_data = fitted_domain + resids

    first_zeros = []
    # need this form for numba
    for i in range(num_sims):
        sim_row = sim_data[i]
        curr_first_zero = first_zero(domain, sim_row, level=level)
        first_zeros.append(curr_first_zero)
    return np.array(first_zeros)


@jit(nopython=True)
def boot_first_zero_draws(domain,
                          fitted_domain,
                          residuals,
                          num_boot_sims=1000,
                          seed=None,
                          level=0.0):
    """
    Creates a distribution of draws from the first element of domain where
    the given model produces a value <= 0.

    Idea is that residuals are from a fit to data: indep, dep.  The parameter
    domain is thought of as model independent values beyond the end of indep.

    Eg: arctic ice, domain is years.
    """
    if seed is not None:
        np.random.seed(seed)

    num_data = len(domain)

    # generate data using boot samples
    num_resid = len(residuals)

    first_zeros = []
    for i in range(num_boot_sims):
        # generate residuals for the domain
        indices = np.random.choice(num_resid, num_data)
        sampled_residuals = residuals[indices]
        # create sim data
        sim_data = fitted_domain + sampled_residuals
        curr_first_zero = first_zero(domain, sim_data, level=level)
        first_zeros.append(curr_first_zero)

    return np.array(first_zeros)


@jit(nopython=True)
def boot_predict_first_zero_ice(domain,
                                fitted_domain,
                                residuals,
                                num_boot_sims=1000,
                                seed=None,
                                level=0.0):
    """
    Creates a distribution of draws from first zeros for boot simulations.
    Then returns the mean of that distribution.

    Idea is that residuals are from a fit to data: indep, dep.  The parameter
    domain is thought of as years after the end of indep.
    """
    first_zeros = boot_first_zero_draws(domain=domain,
                                        fitted_domain=fitted_domain,
                                        residuals=residuals,
                                        num_boot_sims=num_boot_sims,
                                        seed=seed,
                                        level=level)

    first_zero_prediction = np.mean(first_zeros)

    return first_zero_prediction


def zero_ice(estimator, indep, dep, level=0.0):
    """
    do by Model instead?
    TODO: get rid of isinstance checking
    """
    if isinstance(estimator, OneBkptSegRegEstimator):
        u, v, m1, m2, resid_stddev = estimator.fit(indep, dep)
        zero_time = zero_ice_date_one_bkpt([u, v, m1, m2],
                                           level=level)

    elif isinstance(estimator, OLSRegressionEstimator):
        intercept, slope, resid_stddev = estimator.fit(indep, dep)
        zero_time = zero_ice_date_ols([intercept, slope],
                                      level=level)

    elif isinstance(estimator, TwoBkptSegRegEstimator):
        u1, v1, u2, v2, m1, m2, resid_stddev = estimator.fit(indep, dep)
        zero_time = zero_ice_date_two_bkpt([u1, v1, u2, v2, m1, m2],
                                           level=level)
    elif isinstance(estimator, QuadRegEstimator):
        b0, b1, b2, resid_stddev = estimator.fit(indep, dep)
        zero_time = zero_ice_date_quad([b0, b1, b2], level=level)

    elif isinstance(estimator, CubicRegEstimator):
        b0, b1, b2, b3, resid_stddev = estimator.fit(indep, dep)
        zero_time = zero_ice_date_cubic([b0, b1, b2, b3], level=level)

    elif isinstance(estimator, ExponentialEstimator):
        a, b, c, resid_stddev = estimator.fit(indep, dep)
        shift = estimator.get_shift()
        zero_time = zero_ice_date_exp([a, b, c], shift=shift, level=level)

    elif isinstance(estimator, GompertzEstimator):
        a, b, c, resid_stddev = estimator.fit(indep, dep)
        shift = estimator.get_shift()
        zero_time = zero_ice_date_gomp([a, b, c], shift=shift, level=level)

    elif isinstance(estimator, MultiModelEstimator):
        estimator.fit(indep, dep)
        zero_times = [zero_ice(x, indep, dep, level)
                      for x in estimator.estimators]
        bma_wts = estimator.bma_weights()
        zero_time = np.vdot(bma_wts, zero_times)

    return zero_time


def _predict_first_zero(indep,
                        dep,
                        estimator,
                        num_boot_sims=1000,
                        seed=None,
                        convert_infinity=True,
                        level=0.0):
    """
    WARNING: converts far-away estimates to infinity.
    """

    # this will fit the estimator to the data
    zero_time = zero_ice(estimator=estimator,
                         indep=indep,
                         dep=dep,
                         level=level)

    func = estimator.model_function

    resid = estimator.residuals

    if value_is_far(zero_time, indep[-1]) and convert_infinity:

        first_zero_prediction = ZERO_ICE_INFINITY_PROXY
        # return infinity for both
        zero_time = ZERO_ICE_INFINITY_PROXY
    else:
        # if not convert_infinity, could be a lot of calculation here
        domain_begin = indep[-1] + 1.0
        domain_end = zero_time + _NUM_YEARS_PAST_ZERO_ICE
        domain = np.arange(domain_begin, domain_end)

        fitted_domain = func(domain)

        first_zero_prediction = boot_predict_first_zero_ice(domain=domain,
                                                            fitted_domain=fitted_domain,
                                                            residuals=resid,
                                                            num_boot_sims=num_boot_sims,
                                                            seed=seed,
                                                            level=level)

    return first_zero_prediction, zero_time


def predict_first_zero(indep,
                       dep,
                       estimator,
                       num_boot_sims=1000,
                       seed=None,
                       convert_infinity=True,
                       level=0.0):
    if isinstance(estimator, MultiModelEstimator):
        estimator.fit(indep, dep)

        first_zeros = []
        zero_times = []
        for component_estimator in estimator.estimators:
            (first_zero_prediction,
             zero_time) = _predict_first_zero(indep=indep,
                                              dep=dep,
                                              estimator=component_estimator,
                                              num_boot_sims=num_boot_sims,
                                              seed=seed,
                                              convert_infinity=convert_infinity,
                                              level=level)
            first_zeros.append(first_zero_prediction)
            zero_times.append(zero_time)

        bma_wts = estimator.bma_weights()
        first_zero_prediction = np.vdot(bma_wts, first_zeros)
        zero_time = np.vdot(bma_wts, zero_times)

    else:
        (first_zero_prediction,
         zero_time) = _predict_first_zero(indep=indep,
                                          dep=dep,
                                          estimator=estimator,
                                          num_boot_sims=num_boot_sims,
                                          seed=seed,
                                          convert_infinity=convert_infinity,
                                          level=level)
    return first_zero_prediction, zero_time


def infinity_to_max(arr, sort=True):
    """
    Puts the max finite element as probability mass for infinities.

    Ignoring the "infinite" elements of array, computes the maximum of the 
    finite elements.  Then replaces the infinite elements with this finite
    max value.
    """

    finite_arr = np.array(arr)

    inds = np.where(finite_arr > ZERO_ICE_INFINITY_PROXY - 1.0)[0]

    # just to compute max
    tmp = np.array(finite_arr)
    tmp[inds] = -100000
    max_element = max(tmp)
    finite_arr[inds] = max_element

    if sort:
        result = np.sort(finite_arr)
    else:
        result = finite_arr

    return result
