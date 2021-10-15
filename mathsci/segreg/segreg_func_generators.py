"""
Generates model function objects for testing.

Nearly all methods assume data starts at one, and is integral.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numpy as np

from segreg.model import OLSRegressionEstimator
from segreg.model import OneBkptSegRegEstimator
from segreg.model import TwoBkptSegRegEstimator

from mathsci.utilities import text_util
from mathsci.segreg.model import Model

fmt = text_util.get_decimal_formatter(precision=2)


def fmt_arr(arr):
    v = [fmt(x) for x in arr]

    return "[" + ", ".join(v) + "]"


def _compute_m2(m1, scale):
    """
    PARAMETERS
    ----------
    m1: float
        lhs slope in 1bkpt segreg
    scale: float
        angle between segments defined as: scale * pi
        0.5 < scale < 1.0
    """
    gamma = np.arctan(m1)
    return np.tan(gamma + np.pi * (scale - 1.0))


def _compute_m2_from_degree(m1, angle_deg):
    """
    The angle counter-clockwise relative to, and measured from first 
    segment.

    Parameters
    ----------
    m1: float
    angle_deg: float
    """
    # need second segment to not go backwards
    cutoff_low = (0.5 * np.pi - np.arctan(m1)) * 180 / np.pi
    cutoff_high = cutoff_low + 180
    if (angle_deg <= cutoff_low) or (angle_deg >= cutoff_high):
        raise Exception("angle out of bounds")

    scale = angle_deg / 180.0
    m2 = _compute_m2(m1, scale=scale)
    return m2


# def _compute_slopes(m1):
#     deg = np.arange(170, 180)
#     scale = deg / 180.0
#     m2 = _compute_m2(m1, scale=scale)
#     deg_to_m2 = {x: y for x, y in zip(deg, m2)}
#     return deg_to_m2
#
#
# # static calcs -- not using
# m1_0_deg_to_m2 = _compute_slopes(0.0)
# m1_05_deg_to_m2 = _compute_slopes(-0.05)
# m1_10_deg_to_m2 = _compute_slopes(-0.1)

################################################################################
# OLS
################################################################################


def _ols_bkpt_gen(intercept, slope, estimator):

    params = [intercept, slope]
    model_func = estimator.get_func_for_params(params)

    def ols_func_generator(num_data):
        """
        ``num_data`` not used here, but needs to be in interface of returned
        func
        """
        return model_func

    generator = ols_func_generator

    description = {}
    description["model"] = Model.OLS.name
    description["fixed_params"] = {"intercept": intercept,
                                   "slope": slope}

    return generator, description


def get_ols_func_generator(index=1):
    estimator = OLSRegressionEstimator()

    if index == 1:
        intercept = 20.0
        slope = 0.0

        generator, description = _ols_bkpt_gen(intercept, slope, estimator)

    elif index == 2:
        intercept = 20.0
        slope = -0.1

        generator, description = _ols_bkpt_gen(intercept, slope, estimator)
    else:
        raise Exception("no methods defined for this index")

    return generator, description


################################################################################
# ONE BKPT
################################################################################


def _frac_one_bkpt_gen(frac, v, m1, angle_deg, estimator):
    """
    This one is NOT appropriate to compare level predictions for different size
    data, because the length of the right-most line segment changes in length
    to the horizon with different sized data.
    """

    m2 = _compute_m2_from_degree(m1=m1, angle_deg=angle_deg)

    def one_bkpt_func_generator(num_data, indep_start=1, verbose=False):

        u = indep_start - 1 + np.around(frac * num_data)

        params = np.array([u, v, m1, m2])
        if verbose:
            print(params)
        return estimator.get_func_for_params(params)

    generator = one_bkpt_func_generator

    description = {}
    description["model"] = Model.ONE_BKPT.name
    description["fixed_params"] = {"v": v, "m1": m1, "m2": m2}
    description["angle"] = angle_deg
    description["desc"] = "bkpt located at frac of data"
    description["frac"] = frac

    return generator, description


def _fixed_end_one_bkpt_gen(v, m1, angle_deg, estimator, num_end=25):
    """
    This one is appropriate to compare level predictions for different size
    data.
    """

    m2 = _compute_m2_from_degree(m1=m1, angle_deg=angle_deg)

    def one_bkpt_func_generator(num_data, indep_start=1, verbose=False):

        frac = 1.0 - num_end / num_data

        u = indep_start - 1 + np.around(frac * num_data)

        params = np.array([u, v, m1, m2])

        if verbose:
            print(params)
        return estimator.get_func_for_params(params)

    generator = one_bkpt_func_generator

    description = {}
    description["model"] = Model.ONE_BKPT.name
    description["fixed_params"] = {"v": v, "m1": m1, "m2": m2}
    description["angle"] = angle_deg
    description["desc"] = "bkpt located at fixed num before end"
    description["num_end"] = num_end

    return generator, description


def get_one_bkpt_func_generator(index=1):
    estimator = OneBkptSegRegEstimator()

    generator = None
    description = None

    if index == 1:
        frac = 0.25
        v = 10
        m1 = 0.0
        angle_deg = 176

        generator, description = _frac_one_bkpt_gen(frac=frac,
                                                    v=v,
                                                    m1=m1,
                                                    angle_deg=angle_deg,
                                                    estimator=estimator)

    elif index == 2:
        frac = 0.5
        v = 10
        m1 = 0.0
        angle_deg = 176

        generator, description = _frac_one_bkpt_gen(frac=frac,
                                                    v=v,
                                                    m1=m1,
                                                    angle_deg=angle_deg,
                                                    estimator=estimator)

    elif index == 3:
        frac = 0.75
        v = 10
        m1 = 0.0
        angle_deg = 176

        generator, description = _frac_one_bkpt_gen(frac=frac,
                                                    v=v,
                                                    m1=m1,
                                                    angle_deg=angle_deg,
                                                    estimator=estimator)

    ############################################################################

    elif index == 4:
        frac = 0.25
        v = 10
        m1 = 0.0
        angle_deg = 174

        generator, description = _frac_one_bkpt_gen(frac=frac,
                                                    v=v,
                                                    m1=m1,
                                                    angle_deg=angle_deg,
                                                    estimator=estimator)

    elif index == 5:
        frac = 0.5
        v = 10
        m1 = 0.0
        angle_deg = 174

        generator, description = _frac_one_bkpt_gen(frac=frac,
                                                    v=v,
                                                    m1=m1,
                                                    angle_deg=angle_deg,
                                                    estimator=estimator)

    elif index == 6:
        frac = 0.75
        v = 10
        m1 = 0.0
        angle_deg = 174

        generator, description = _frac_one_bkpt_gen(frac=frac,
                                                    v=v,
                                                    m1=m1,
                                                    angle_deg=angle_deg,
                                                    estimator=estimator)

    ############################################################################

    elif index == 7:
        frac = 0.25
        v = 10
        m1 = 0.0
        angle_deg = 172

        generator, description = _frac_one_bkpt_gen(frac=frac,
                                                    v=v,
                                                    m1=m1,
                                                    angle_deg=angle_deg,
                                                    estimator=estimator)

    elif index == 8:
        frac = 0.5
        v = 10
        m1 = 0.0
        angle_deg = 172

        generator, description = _frac_one_bkpt_gen(frac=frac,
                                                    v=v,
                                                    m1=m1,
                                                    angle_deg=angle_deg,
                                                    estimator=estimator)

    elif index == 9:
        frac = 0.75
        v = 10
        m1 = 0.0
        angle_deg = 172

        generator, description = _frac_one_bkpt_gen(frac=frac,
                                                    v=v,
                                                    m1=m1,
                                                    angle_deg=angle_deg,
                                                    estimator=estimator)

    ############################################################################

    elif index == 10:
        v = 10
        m1 = 0.0
        angle_deg = 176
        num_end = 20

        generator, description = _fixed_end_one_bkpt_gen(v=v,
                                                         m1=m1,
                                                         angle_deg=angle_deg,
                                                         estimator=estimator,
                                                         num_end=num_end)

    elif index == 11:
        v = 10
        m1 = 0.0
        angle_deg = 174
        num_end = 20

        generator, description = _fixed_end_one_bkpt_gen(v=v,
                                                         m1=m1,
                                                         angle_deg=angle_deg,
                                                         estimator=estimator,
                                                         num_end=num_end)

    elif index == 12:
        v = 10
        m1 = 0.0
        angle_deg = 172
        num_end = 20

        generator, description = _fixed_end_one_bkpt_gen(v=v,
                                                         m1=m1,
                                                         angle_deg=angle_deg,
                                                         estimator=estimator,
                                                         num_end=num_end)

    elif index == 13:
        v = 10
        m1 = 0.0
        angle_deg = 176
        num_end = 30

        generator, description = _fixed_end_one_bkpt_gen(v=v,
                                                         m1=m1,
                                                         angle_deg=angle_deg,
                                                         estimator=estimator,
                                                         num_end=num_end)

    elif index == 14:
        v = 10
        m1 = 0.0
        angle_deg = 174
        num_end = 30

        generator, description = _fixed_end_one_bkpt_gen(v=v,
                                                         m1=m1,
                                                         angle_deg=angle_deg,
                                                         estimator=estimator,
                                                         num_end=num_end)

    elif index == 15:
        v = 10
        m1 = 0.0
        angle_deg = 172
        num_end = 30

        generator, description = _fixed_end_one_bkpt_gen(v=v,
                                                         m1=m1,
                                                         angle_deg=angle_deg,
                                                         estimator=estimator,
                                                         num_end=num_end)

    else:
        raise Exception("no methods defined for this index")

    return generator, description

################################################################################
# TWO BKPT
################################################################################


def _frac_two_bkpt_gen(frac1, frac2, v2, m1, angle_deg1, angle_deg2, estimator):

    m = _compute_m2_from_degree(m1=m1, angle_deg=angle_deg1)
    m2 = _compute_m2_from_degree(m1=m, angle_deg=angle_deg2)

    def two_bkpt_func_generator(num_data, indep_start=1, verbose=False):

        u1 = indep_start - 1 + np.around(frac1 * num_data)
        u2 = indep_start - 1 + np.around(frac2 * num_data)

        # v1 is determined here
        v1 = v2 - m * (u2 - u1)

        params = [u1, v1, u2, v2, m1, m2]
        if verbose:
            print(params)
            print("m: ", m)

        return estimator.get_func_for_params(params)

    generator = two_bkpt_func_generator

    description = {}
    description["model"] = Model.TWO_BKPT.name
    description["fixed_params"] = {"v2": v2, "m1": m1, "m": m, "m2": m2}
    description["angle1"] = angle_deg1
    description["angle2"] = angle_deg2
    description["desc"] = "bkpts located at frac1, frac2 of data"
    description["frac1"] = frac1
    description["frac2"] = frac2

    return generator, description


def _frac_two_bkpt_gen_flat_mid(frac1,
                                frac2,
                                v2,
                                angle_deg1,
                                angle_deg2,
                                estimator):
    m = 0.0

    ang1 = 360 - angle_deg1
    m1 = _compute_m2_from_degree(m1=0.0, angle_deg=ang1)
    m2 = _compute_m2_from_degree(m1=m, angle_deg=angle_deg2)

    def two_bkpt_func_generator(num_data, indep_start=1, verbose=False):

        u1 = indep_start - 1 + np.around(frac1 * num_data)
        u2 = indep_start - 1 + np.around(frac2 * num_data)

        # v1 is determined here
        v1 = v2 - m * (u2 - u1)

        params = [u1, v1, u2, v2, m1, m2]
        if verbose:
            print(params)
            print("m: ", m)

        return estimator.get_func_for_params(params)

    generator = two_bkpt_func_generator

    description = {}
    description["model"] = Model.TWO_BKPT.name
    description["fixed_params"] = {"v2": v2, "m1": m1, "m": m, "m2": m2}
    description["angle1"] = angle_deg1
    description["angle2"] = angle_deg2
    description["desc"] = "bkpts located at frac1, frac2 of data"
    description["frac1"] = frac1
    description["frac2"] = frac2

    return generator, description


def get_two_bkpt_func_generator(index=1):
    estimator = TwoBkptSegRegEstimator()

    generator = None
    description = None

    if index == 1:
        frac1 = 1 / 3
        frac2 = 2 / 3

        v2 = 6

        m1 = 0.0
        angle_deg1 = 176
        angle_deg2 = 360 - angle_deg1

        generator, description = _frac_two_bkpt_gen(frac1=frac1,
                                                    frac2=frac2,
                                                    v2=v2,
                                                    m1=m1,
                                                    angle_deg1=angle_deg1,
                                                    angle_deg2=angle_deg2,
                                                    estimator=estimator)

    elif index == 2:
        frac1 = 1 / 3
        frac2 = 2 / 3

        v2 = 6

        m1 = 0.0
        angle_deg1 = 174
        angle_deg2 = 360 - angle_deg1

        generator, description = _frac_two_bkpt_gen(frac1=frac1,
                                                    frac2=frac2,
                                                    v2=v2,
                                                    m1=m1,
                                                    angle_deg1=angle_deg1,
                                                    angle_deg2=angle_deg2,
                                                    estimator=estimator)

    elif index == 3:
        frac1 = 1 / 3
        frac2 = 2 / 3

        v2 = 6

        m1 = 0.0
        angle_deg1 = 172
        angle_deg2 = 360 - angle_deg1

        generator, description = _frac_two_bkpt_gen(frac1=frac1,
                                                    frac2=frac2,
                                                    v2=v2,
                                                    m1=m1,
                                                    angle_deg1=angle_deg1,
                                                    angle_deg2=angle_deg2,
                                                    estimator=estimator)
    elif index == 4:
        frac1 = 1 / 3
        frac2 = 2 / 3

        v2 = 6

        angle_deg1 = 180 + 4
        angle_deg2 = 360 - angle_deg1

        generator, description = _frac_two_bkpt_gen_flat_mid(frac1=frac1,
                                                             frac2=frac2,
                                                             v2=v2,
                                                             angle_deg1=angle_deg1,
                                                             angle_deg2=angle_deg2,
                                                             estimator=estimator)
    elif index == 5:
        frac1 = 1 / 3
        frac2 = 2 / 3

        v2 = 6

        angle_deg1 = 180 + 6
        angle_deg2 = 360 - angle_deg1

        generator, description = _frac_two_bkpt_gen_flat_mid(frac1=frac1,
                                                             frac2=frac2,
                                                             v2=v2,
                                                             angle_deg1=angle_deg1,
                                                             angle_deg2=angle_deg2,
                                                             estimator=estimator)
    elif index == 6:
        frac1 = 1 / 3
        frac2 = 2 / 3

        v2 = 6

        angle_deg1 = 180 + 8
        angle_deg2 = 360 - angle_deg1

        generator, description = _frac_two_bkpt_gen_flat_mid(frac1=frac1,
                                                             frac2=frac2,
                                                             v2=v2,
                                                             angle_deg1=angle_deg1,
                                                             angle_deg2=angle_deg2,
                                                             estimator=estimator)

    elif index == 7:
        frac1 = 0.25
        frac2 = 0.75

        v2 = 6

        angle_deg1 = 180 + 4
        angle_deg2 = 360 - angle_deg1

        generator, description = _frac_two_bkpt_gen_flat_mid(frac1=frac1,
                                                             frac2=frac2,
                                                             v2=v2,
                                                             angle_deg1=angle_deg1,
                                                             angle_deg2=angle_deg2,
                                                             estimator=estimator)
    elif index == 8:
        frac1 = 0.25
        frac2 = 0.75

        v2 = 6

        angle_deg1 = 180 + 6
        angle_deg2 = 360 - angle_deg1

        generator, description = _frac_two_bkpt_gen_flat_mid(frac1=frac1,
                                                             frac2=frac2,
                                                             v2=v2,
                                                             angle_deg1=angle_deg1,
                                                             angle_deg2=angle_deg2,
                                                             estimator=estimator)
    elif index == 9:
        frac1 = 0.25
        frac2 = 0.75

        v2 = 6

        angle_deg1 = 180 + 8
        angle_deg2 = 360 - angle_deg1

        generator, description = _frac_two_bkpt_gen_flat_mid(frac1=frac1,
                                                             frac2=frac2,
                                                             v2=v2,
                                                             angle_deg1=angle_deg1,
                                                             angle_deg2=angle_deg2,
                                                             estimator=estimator)

    else:
        raise Exception("no methods defined for this index")

    return generator, description
