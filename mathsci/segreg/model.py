"""
Basic Models.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

from enum import Enum

import numpy as np
import scipy.optimize

from segreg.model.alt import regression_alt
from segreg.model.estimator import Estimator
from segreg.model import OLSRegressionEstimator
from segreg.model import OneBkptSegRegEstimator
from segreg.model import TwoBkptSegRegEstimator


class Model(Enum):
    OLS = 1
    ONE_BKPT = 2
    TWO_BKPT = 3
    QUAD = 4
    CUBIC = 5
    EXP = 6
    GOMPERTZ = 7

    @property
    def display(self):
        return self.name.lower()

    def estimator(self, num_end_to_skip=10, num_between_to_skip=10):
        if self == Model.OLS:
            return OLSRegressionEstimator()
        elif self == Model.ONE_BKPT:
            return OneBkptSegRegEstimator(num_end_to_skip=num_end_to_skip)
        elif self == Model.TWO_BKPT:
            return TwoBkptSegRegEstimator(num_end_to_skip=num_end_to_skip,
                                          num_between_to_skip=num_between_to_skip)
        elif self == Model.QUAD:
            return QuadRegEstimator()
        elif self == Model.CUBIC:
            return CubicRegEstimator()
        elif self == Model.EXP:
            return ExponentialEstimator()
        elif self == Model.GOMPERTZ:
            return GompertzEstimator()

    # this only affects print()
    # does not affect ".name"
    def __str__(self):
        #         myname = self.name
        #         myname = myname.replace("ONE_", "1")
        #         myname = myname.replace("TWO_", "2")
        #         return myname
        return self.display


class QuadRegEstimator(Estimator):

    """
    classdocs
    """

    def __init__(self):

        self._num_params = 4

        self._indep = None
        self._dep = None
        self._params = None
        self._is_estimated = False

        self._fixed_params_indices = []
        self._estimated_params_indices = np.setdiff1d(np.arange(self._num_params),
                                                      self._fixed_params_indices)

        self._not_fit_message = "Need to call 'fit()' first"

    def _clear(self):
        self._indep = None
        self._dep = None
        self._params = None
        self._is_estimated = False

    def _set_data(self, indep, dep):
        self._clear()

        # we sort the data
        argsort_for_indep = indep.argsort()
        self._indep = indep[argsort_for_indep]
        self._dep = dep[argsort_for_indep]

    def has_restricted_params(self):
        pass

    @property
    def num_params(self):
        return self._num_params

    @property
    def param_names(self):
        return ["b0", "b1", "b2", "sigma"]

    @property
    def estimated_params_indices(self):
        return self._estimated_params_indices

    def fit(self, indep, dep):
        self._set_data(indep, dep)

        indep_to_use = np.vstack((self._indep, self._indep**2)).T
        est_const, est_mat, rss = regression_alt.ols_with_rss(indep_to_use,
                                                              self._dep)

        num_obs = len(self._indep)

        self._rss = rss
        # TODO: mle way or non-bias way here?
        #variance = self._rss / num_obs

        # TODO: use no-bias way
        # TODO: double check this formula
        # TODO: put in based on variable
        variance = self._rss / (num_obs - 2)
        resid_stddev = np.sqrt(variance)

        self._params = [est_const, est_mat[0], est_mat[1], resid_stddev]
        self._params = np.array(self._params)
        self._is_estimated = True

        return self._params

    def rss(self):
        return self._rss

    def get_func(self):
        if not self._is_estimated:
            raise Exception(self._not_fit_message)

        return self.get_func_for_params(self._params)

    def get_func_for_params(self, params):
        b0 = params[0]
        b1 = params[1]
        b2 = params[2]

        # TODO: handle for multivariate
        def ols_func(x):
            return b0 + b1 * x + b2 * x * x

        return ols_func


class CubicRegEstimator(Estimator):

    """
    classdocs
    """

    def __init__(self):

        self._num_params = 5

        self._indep = None
        self._dep = None
        self._params = None
        self._is_estimated = False

        self._fixed_params_indices = []
        self._estimated_params_indices = np.setdiff1d(np.arange(self._num_params),
                                                      self._fixed_params_indices)

        self._not_fit_message = "Need to call 'fit()' first"

    def _clear(self):
        self._indep = None
        self._dep = None
        self._params = None
        self._is_estimated = False

    def _set_data(self, indep, dep):
        self._clear()

        # we sort the data
        argsort_for_indep = indep.argsort()
        self._indep = indep[argsort_for_indep]
        self._dep = dep[argsort_for_indep]

    def has_restricted_params(self):
        pass

    @property
    def num_params(self):
        return self._num_params

    @property
    def param_names(self):
        return ["b0", "b1", "b2", "b3", "sigma"]

    @property
    def estimated_params_indices(self):
        return self._estimated_params_indices

    def fit(self, indep, dep):
        self._set_data(indep, dep)

        indep_to_use = np.vstack((self._indep,
                                  self._indep**2,
                                  self._indep**3)).T
        est_const, est_mat, rss = regression_alt.ols_with_rss(indep_to_use,
                                                              self._dep)

        num_obs = len(self._indep)

        self._rss = rss
        # TODO: mle way or non-bias way here?
        #variance = self._rss / num_obs

        # TODO: use no-bias way
        # TODO: double check this formula
        # TODO: put in based on variable
        variance = self._rss / (num_obs - 2)
        resid_stddev = np.sqrt(variance)

        self._params = [est_const,
                        est_mat[0],
                        est_mat[1],
                        est_mat[2],
                        resid_stddev]
        self._params = np.array(self._params)
        self._is_estimated = True

        return self._params

    def rss(self):
        return self._rss

    def get_func(self):
        if not self._is_estimated:
            raise Exception(self._not_fit_message)

        return self.get_func_for_params(self._params)

    def get_func_for_params(self, params):
        b0 = params[0]
        b1 = params[1]
        b2 = params[2]
        b3 = params[3]

        # TODO: handle for multivariate
        def ols_func(x):
            return b0 + b1 * x + b2 * x * x + b3 * x * x * x

        return ols_func


class ExponentialEstimator(Estimator):

    """
    y = a + b exp(cx)

    WARNING
    -------
    This class shifts the data by subtracting first data point.  After fitting,
    calling the method ``get_func`` and ``get_func_for_params`` will return a 
    function which shifts the data to the left.  When used outside this class, 
    subsequent re-adjustment will be necessary (eg: plotting, extrapolation).

    The reason is that the exponential function will quickly go beyond machine
    precision with large enough arguments.

    """

    def __init__(self):

        self._num_params = 4

        self._indep = None
        self._dep = None
        self._params = None
        self._is_estimated = False

        self._not_fit_message = "Need to call 'fit()' first"

        self._fixed_params_indices = []
        self._estimated_params_indices = np.setdiff1d(np.arange(self._num_params),
                                                      self._fixed_params_indices)

        self._shift = 0.0

    def _clear(self):
        self._indep = None
        self._dep = None
        self._params = None
        self._is_estimated = False
        self._shift = 0.0

    def _set_data(self, indep, dep):
        self._clear()

        # we sort the data
        argsort_for_indep = indep.argsort()
        self._indep = indep[argsort_for_indep]
        self._dep = dep[argsort_for_indep]

    def has_restricted_params(self):
        pass

    @property
    def num_params(self):
        return self._num_params

    @property
    def param_names(self):
        return ["a", "b", "c", "sigma"]

    @property
    def estimated_params_indices(self):
        return self._estimated_params_indices

    def fit(self, indep, dep):
        """
        TODO: provide the shift separately; not in return from fit since it 
        screws up other methods, eg: boot conf intervals
        """

        # shift
        shift = indep[0]

        scale = 0.0001

        self._set_data(indep, dep)

        self._shift = shift

        num_obs = len(self._indep)

        leven_marq = True

        if leven_marq:

            def resid_func(params):

                # fixing params in this specification can significantly degrade
                # fits
                #params[0] = 0.0
                #params[1] = -1.0
                # this matches Meier-Stroeve
                #params[2] = 100

                params_to_use = np.array([params[0],
                                          params[1],
                                          scale * params[2]])

                #print("inside obj func: ", params_to_use)

                func = self.get_func_for_params(params_to_use)
                resid = dep - func(indep)

                return resid

            init_params = np.array([10.0, -1.0, 1.0])

            result = scipy.optimize.least_squares(fun=resid_func,
                                                  x0=init_params,
                                                  max_nfev=100000)

        else:
            def rss_func(params):

                params_to_use = np.array([params[0],
                                          params[1],
                                          scale * params[2]])

                # print(params_to_use)

                func = self.get_func_for_params(params_to_use)
                resid = dep - func(indep)

                return np.vdot(resid, resid)

            init_params = np.array([10.0, -1.0, 1.0])

#             result = scipy.optimize.minimize(rss_func,
#                                              init_params,
#                                              method="BFGS",
#                                              )

            result = scipy.optimize.minimize(rss_func,
                                             init_params,
                                             method="Nelder-Mead",
                                             options={'maxiter': 100000,
                                                      'maxfev': 100000,
                                                      'fatol': 1.0e-6,
                                                      'disp': False},
                                             tol=1.0e-6)

        est_params = result.x

        if result.status < 1:
            # if True:
            print("-" * 50)
            # print(result.status)
            print(result.message)
            # print(result)
            # print(func_est_params)
            # print(mle_params)
            if leven_marq:
                print("RSS: ", 2.0 * result.cost)
            else:
                print("RSS: ", rss_func(est_params))
            print()

        est_params = np.array([est_params[0],
                               est_params[1],
                               scale * est_params[2]])

        est_resid = result.fun

        self._rss = np.vdot(est_resid, est_resid)

        # TODO: mle way or non-bias way here?
        #variance = self._rss / num_obs
        # TODO: use no-bias way
        # TODO: double check this formula
        # TODO: put in based on variable
        variance = self._rss / (num_obs - 2)
        resid_stddev = np.sqrt(variance)

        self._params = [est_params[0],
                        est_params[1],
                        est_params[2],
                        resid_stddev]
        self._params = np.array(self._params)
        self._is_estimated = True

        # print(self._params)

        return self._params

    def rss(self):
        return self._rss

    def get_func(self):
        if not self._is_estimated:
            raise Exception(self._not_fit_message)

        return self.get_func_for_params(self._params)

    def get_func_for_params(self, params):
        a = params[0]
        b = params[1]
        c = params[2]

        def func(x):
            # return a + b * np.exp(c * x)

            exponent = c * (x - self._shift)

            #print("max exponent: ", max(exponent))

            return a + b * np.exp(exponent)

        return func

    def get_shift(self):
        return self._shift


class GompertzEstimator(Estimator):

    """
    y = a exp( -exp( (x-b)/c ) )

    WARNING
    -------
    This class shifts the data by subtracting first data point.  After fitting,
    calling the method ``get_func`` and ``get_func_for_params`` will return a 
    function which shifts the data to the left.  When used outside this class, 
    subsequent re-adjustment will be necessary (eg: plotting, extrapolation).
    The ``fit`` method likewise returns an additional element giving the value
    of the shift that was used.

    The reason is that the exponential function will quickly go beyond machine
    precision with large enough arguments.

    """

    def __init__(self):

        self._num_params = 4

        self._indep = None
        self._dep = None
        self._params = None
        self._is_estimated = False

        self._not_fit_message = "Need to call 'fit()' first"

        self._fixed_params_indices = []
        self._estimated_params_indices = np.setdiff1d(np.arange(self._num_params),
                                                      self._fixed_params_indices)

        self._shift = 0.0

    def _clear(self):
        self._indep = None
        self._dep = None
        self._params = None
        self._is_estimated = False
        self._shift = 0.0

    def _set_data(self, indep, dep):
        self._clear()

        # we sort the data
        argsort_for_indep = indep.argsort()
        self._indep = indep[argsort_for_indep]
        self._dep = dep[argsort_for_indep]

    def has_restricted_params(self):
        pass

    @property
    def num_params(self):
        return self._num_params

    @property
    def param_names(self):
        return ["a", "b", "c", "sigma"]

    @property
    def estimated_params_indices(self):
        return self._estimated_params_indices

    def fit(self, indep, dep):
        """
        TODO: provide the shift separately; not in return from fit since it 
        screws up other methods, eg: boot conf intervals
        """

        # shift
        shift = indep[0]

        #scale = 0.0001
        #scale = 0.0001
        scale = 1.0

        self._set_data(indep, dep)

        self._shift = shift

        num_obs = len(self._indep)

        leven_marq = True

        if leven_marq:

            def resid_func(params):

                # fixing params in this specification can significantly degrade
                # fits
                #params[1] = -1.0
                #params[2] = 300.0

                params_to_use = np.array([params[0],
                                          params[1],
                                          scale * params[2]])

                #print("inside obj func: ", params_to_use)

                func = self.get_func_for_params(params_to_use)
                resid = dep - func(indep)

                return resid

            init_params = np.array([10.0, -1.0, 1.0])

            result = scipy.optimize.least_squares(fun=resid_func,
                                                  x0=init_params,
                                                  max_nfev=100000)

        else:
            def rss_func(params):

                params_to_use = np.array([params[0],
                                          params[1],
                                          scale * params[2]])

                # print(params_to_use)

                func = self.get_func_for_params(params_to_use)
                resid = dep - func(indep)

                return np.vdot(resid, resid)

            #init_params = np.array([0.0, 1.0, 1.0])
            init_params = np.array([10.0, -1.0, 1.0])

#             result = scipy.optimize.minimize(rss_func,
#                                              init_params,
#                                              method="BFGS",
#                                              )

            result = scipy.optimize.minimize(rss_func,
                                             init_params,
                                             method="Nelder-Mead",
                                             options={'maxiter': 100000,
                                                      'maxfev': 100000,
                                                      'fatol': 1.0e-6,
                                                      'disp': False},
                                             tol=1.0e-6)

        est_params = result.x

        if result.status < 1:
            # if True:
            print("-" * 50)
            # print(result.status)
            print(result.message)
            # print(result)
            # print(func_est_params)
            # print(mle_params)
            if leven_marq:
                print("RSS: ", 2.0 * result.cost)
            else:
                print("RSS: ", rss_func(est_params))
            print()

        est_params = np.array([est_params[0],
                               est_params[1],
                               scale * est_params[2]])

        est_resid = result.fun

        self._rss = np.vdot(est_resid, est_resid)

        # TODO: mle way or non-bias way here?
        #variance = self._rss / num_obs
        # TODO: use no-bias way
        # TODO: double check this formula
        # TODO: put in based on variable
        variance = self._rss / (num_obs - 2)
        resid_stddev = np.sqrt(variance)

        self._params = [est_params[0],
                        est_params[1],
                        est_params[2],
                        resid_stddev]
        self._params = np.array(self._params)
        self._is_estimated = True

        return self._params

    def rss(self):
        return self._rss

    def get_func(self):
        if not self._is_estimated:
            raise Exception(self._not_fit_message)

        return self.get_func_for_params(self._params)

    def get_func_for_params(self, params):
        a = params[0]
        b = params[1]
        c = params[2]

        def func(x):
            exponent = -1.0 * np.exp((x - self._shift - b) / c)

            return a * np.exp(exponent)

        return func

    def get_shift(self):
        return self._shift
