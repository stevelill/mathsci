"""

"""

# Author: Steven Lillywhite
# License: BSD 3 clause

from collections import namedtuple

import numpy as np

from mathsci.statistics.ou import OU
from mathsci.statistics import stats_util

FakeData = namedtuple("FakeData", ["indep", "deps", "name", "func"])

_STUDENT_DF = 4
_HALF_LIVES = [0.5, 1.5]


def create_ou_generator(half_life, equilibrium_vol, mean_rev_level):

    def ou_generator(num_sims, num_data, dt):

        k = np.log(2.0) / half_life
        vol = equilibrium_vol * np.sqrt(2.0 * k)
        resid_ou = OU(mean_rev_rate=k, mean_rev_level=mean_rev_level, vol=vol)

        ou_sims = resid_ou.simulate(num_sims=num_sims,
                                    num_data=num_data,
                                    dt=dt,
                                    init_val=None)
        return ou_sims

    return ou_generator


# Warning: this must stay in sync with ``fake_data_suite`` !!!
# Use with caution
def fake_data_names():
    return ["Normal",
            "Student (" + str(_STUDENT_DF) + ")",
            "AR1 (" + str(_HALF_LIVES[0]) + ")",
            "AR1 (" + str(_HALF_LIVES[1]) + ")"]


def fake_data_suite(model_func,
                    num_data,
                    num_sims,
                    indep_start=1.0,
                    resid_stddev=0.5,
                    seed=None):
    """
    Creates simulated data.

    Creates four sets of simulated data corresponding to different residuals:
        normal, student T, AR1, AR1

    Details
    -------
    normal: mean: zero, stddev: ``resid_stddev``
    student T: 
    AR1: 
    AR1:

    Parameters
    ----------
    model_func: function object
    num_data: int
    num_sims: int
    indep_start: float
    resid_stddev: float
    seed: int

    Returns
    -------
    normal_data: mathsci.segreg.fake_data.FakeData
        normal residuals
    student_data: mathsci.segreg.fake_data.FakeData
        student T residuals, deg freedom 4
    ar1_data1: mathsci.segreg.fake_data.FakeData
        AR1 residuals, half-life 0.5
    ar2_data2: mathsci.segreg.fake_data.FakeData
        AR1 residuals, half-life 1.5
    """

    # 1. normal
    indep1, deps1 = generate_fake_data_even_space(func=model_func,
                                                  num_data=num_data,
                                                  indep_start=indep_start,
                                                  resid_stddev=resid_stddev,
                                                  num_sims=num_sims,
                                                  seed=seed,
                                                  student_df=None,
                                                  resid_generator=None)

    name1 = "Normal"
    normal_data = FakeData(indep=indep1,
                           deps=deps1,
                           name=name1,
                           func=model_func)

    # 2. student
    student_df = _STUDENT_DF
    indep2, deps2 = generate_fake_data_even_space(func=model_func,
                                                  num_data=num_data,
                                                  indep_start=indep_start,
                                                  resid_stddev=resid_stddev,
                                                  num_sims=num_sims,
                                                  seed=seed,
                                                  student_df=student_df,
                                                  resid_generator=None)

    name2 = "Student (" + str(student_df) + ")"
    student_data = FakeData(indep=indep2,
                            deps=deps2,
                            name=name2,
                            func=model_func)

    # 3. ar1
    half_life = _HALF_LIVES[0]

    ou_gen1 = create_ou_generator(half_life=half_life,
                                  equilibrium_vol=resid_stddev,
                                  mean_rev_level=0.0)
    indep3, deps3 = generate_fake_data_even_space(func=model_func,
                                                  num_data=num_data,
                                                  indep_start=indep_start,
                                                  resid_stddev=resid_stddev,
                                                  num_sims=num_sims,
                                                  seed=seed,
                                                  student_df=None,
                                                  resid_generator=ou_gen1)

    name3 = "AR1 (" + str(half_life) + ")"
    ou_data1 = FakeData(indep=indep3,
                        deps=deps3,
                        name=name3,
                        func=model_func)

    # 4. ar1
    half_life = _HALF_LIVES[1]

    ou_gen2 = create_ou_generator(half_life=half_life,
                                  equilibrium_vol=resid_stddev,
                                  mean_rev_level=0.0)
    indep4, deps4 = generate_fake_data_even_space(func=model_func,
                                                  num_data=num_data,
                                                  indep_start=indep_start,
                                                  resid_stddev=resid_stddev,
                                                  num_sims=num_sims,
                                                  seed=seed,
                                                  student_df=None,
                                                  resid_generator=ou_gen2)

    name4 = "AR1 (" + str(half_life) + ")"
    ou_data2 = FakeData(indep=indep4,
                        deps=deps4,
                        name=name4,
                        func=model_func)

    return normal_data, student_data, ou_data1, ou_data2


def generate_fake_data_even_space(func,
                                  num_data,
                                  indep_start,
                                  resid_stddev,
                                  num_sims=1,
                                  seed=None,
                                  student_df=None,
                                  resid_generator=None,
                                  dt=1.0):
    """
    Generates fake data where the ``indep`` data is evenly spaced.

    The generated data is:
        func(indep) + residuals

    Parameters
    ----------
    func: function object
        The underlying model, a univariate function.
    num_data: int
    indep_start: float
    num_sims: int
    seed: int or None
    student_df: int
    resid_generator: function object
    """
    indep = np.arange(indep_start, indep_start + num_data, dt, dtype=float)
    data_fitted = func(indep)

    if seed is not None:
        np.random.seed(seed)

    if student_df is not None and resid_generator is not None:
        raise Exception("cannot choose both student_df and resid_generator")

    if student_df is not None:
        resids = stats_util.create_t_variates(num_sims=num_sims,
                                              num_data=num_data,
                                              df=student_df,
                                              scale=resid_stddev)
    elif resid_generator is not None:
        resids = resid_generator(num_sims=num_sims,
                                 num_data=num_data,
                                 dt=dt)

    else:
        resids = resid_stddev * np.random.randn(num_sims, num_data)

    deps = data_fitted + resids

    return indep, deps
