"""
Copies arctic_ice.prediction_methods; todo: reduce dupe
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numpy as np

from segreg.model import OLSRegressionEstimator
from segreg.model import OneBkptSegRegEstimator
from segreg.model import TwoBkptSegRegEstimator
from mathsci.segreg.model import QuadRegEstimator

INFINITY_PROXY = 10000.0

_SLOPE_THRESHOLD = -1.0e-4


def level_predict(estimator, indep, dep, level=0.0):
    """
    Predicts the time (or whatever units of ``indep``) for which the specified
    ``level`` is attained.

    do by Model instead?
    TODO: get rid of isinstance checking
    """
    if isinstance(estimator, OneBkptSegRegEstimator):
        u, v, m1, m2, resid_stddev = estimator.fit(indep, dep)
        zero_time = level_predict_one_bkpt([u, v, m1, m2],
                                           level=level)

    elif isinstance(estimator, OLSRegressionEstimator):
        intercept, slope, resid_stddev = estimator.fit(indep, dep)
        zero_time = level_predict_ols([intercept, slope],
                                      level=level)

    elif isinstance(estimator, TwoBkptSegRegEstimator):
        u1, v1, u2, v2, m1, m2, resid_stddev = estimator.fit(indep, dep)
        zero_time = level_predict_two_bkpt([u1, v1, u2, v2, m1, m2],
                                           level=level)
    elif isinstance(estimator, QuadRegEstimator):
        b0, b1, b2, resid_stddev = estimator.fit(indep, dep)
        zero_time = level_predict_quad([b0, b1, b2], level=level)
        # begin erase
        ##print("[b0, b1, b2]: ", b0, b1, b2)
        # end erase

    # TODO: circular dependency; put back

#     elif isinstance(estimator, MultiModelEstimator):
#         estimator.fit(indep, dep)
#         zero_times = [level_predict(x, indep, dep, level)
#                       for x in estimator.estimators]
#         bma_wts = estimator.bma_weights()
#         zero_time = np.vdot(bma_wts, zero_times)

    return zero_time


def _level_for_ols(params, level=0.0):
    intercept, slope = params

    # typical values would be v={4-8}, so m2=-1e-3 gives zero ice time
    # on order of 4000 - 8000; for the purposes of ice melt, this is
    # effectively infinity

    result = INFINITY_PROXY

    if slope < _SLOPE_THRESHOLD:
        zero = (level - intercept) / slope

        if zero >= INFINITY_PROXY:
            result = INFINITY_PROXY
        else:
            result = zero

    return result


def level_predict_ols(functional_params, level=0.0):

    dim = np.ndim(functional_params)

    if dim == 1:
        return _level_for_ols(functional_params, level=level)
    elif dim == 2:
        zeros = []
        for params in functional_params:
            zeros.append(_level_for_ols(params, level=level))
        return np.array(zeros)


def _level_for_second_line_one_bkpt(params, level=0.0):

    u, v, m1, m2 = params

    # typical values would be v={4-8}, so m2=-1e-3 gives zero ice time
    # on order of 4000 - 8000; for the purposes of ice melt, this is
    # effectively infinity

    result = INFINITY_PROXY

    if m2 < _SLOPE_THRESHOLD:
        zero = u + (level - v) / m2

        if zero >= INFINITY_PROXY:
            result = INFINITY_PROXY
        else:
            result = zero

    return result


def level_predict_one_bkpt(functional_params, level=0.0):
    """
    PARAMETERS
    ----------
    functional_params: array-like
        [u, v, m1, m2]
    """
    # TODO dim 1 and dim 2 versions
    #u, v, m1, m2 = functional_params
    # return u - v/m2

    # TODO filter out where m2 >= 0

    dim = np.ndim(functional_params)

    if dim == 1:
        return _level_for_second_line_one_bkpt(functional_params, level=level)
    elif dim == 2:
        zeros = []
        for params in functional_params:
            zeros.append(_level_for_second_line_one_bkpt(params, level=level))
        return np.array(zeros)


def _level_for_second_line_two_bkpt(params, level=0.0):

    u1, v1, u2, v2, m1, m2 = params

    # typical values would be v={4-8}, so m2=-1e-3 gives zero ice time
    # on order of 4000 - 8000; for the purposes of ice melt, this is
    # effectively infinity

    result = INFINITY_PROXY

    if m2 < _SLOPE_THRESHOLD:
        zero = u2 + (level - v2) / m2

        if zero >= INFINITY_PROXY:
            result = INFINITY_PROXY
        else:
            result = zero

    return result


def level_predict_two_bkpt(functional_params, level=0.0):
    #    u1, v1, u2, v2, m1, m2 = functional_params
    #    return u2 - v2 / m2

    dim = np.ndim(functional_params)

    if dim == 1:
        return _level_for_second_line_two_bkpt(functional_params, level=level)
    elif dim == 2:
        zeros = []
        for params in functional_params:
            zeros.append(_level_for_second_line_two_bkpt(params, level=level))
        return np.array(zeros)

# TODO: move to general?


def quad_zero(a, b, c):
    """
    ax^2 + bx + c
    """
    return (-b - np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


def _level_quad(functional_params, level=0.0):
    b0 = functional_params[0]
    b1 = functional_params[1]
    b2 = functional_params[2]

    # a,b,c of quadratic function above
    a = b2
    b = b1
    c = b0 - level

    term = b * b - 4.0 * a * c
    if term < 0.0:
        return INFINITY_PROXY

    return quad_zero(a=a, b=b, c=c)


def level_predict_quad(functional_params, level=0.0):
    dim = np.ndim(functional_params)

    if dim == 1:
        return _level_quad(functional_params, level=level)
    elif dim == 2:
        zeros = []
        for params in functional_params:
            zeros.append(_level_quad(params, level=level))
        return np.array(zeros)
