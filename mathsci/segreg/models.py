"""
OLS, ONE BKPT, TWO BKPT
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


from collections import namedtuple
import numpy as np
import pandas as pd

from segreg.model import OLSRegressionEstimator
from segreg.model import OneBkptSegRegEstimator
from segreg.model import TwoBkptSegRegEstimator

from mathsci.segreg import predictions
from mathsci.segreg.model import Model
from mathsci.statistics import stats_util
from mathsci.statistics.stats_util import BIC_TYPE


FitData = namedtuple("FitData", ["num_params",
                                 "mle",
                                 "loglikelihood",
                                 "at_boundary",
                                 "prediction",
                                 "level_prediction"])


def check_boundary_one_bkpt(model_params, num_end_to_skip, indep):
    bkpt = model_params[0]
    at_left = (bkpt - indep[0] == num_end_to_skip + 1)
    at_right = (indep[-1] - bkpt == num_end_to_skip + 1)

    at_boundary = (at_left or at_right)

    return at_boundary


def check_boundary_two_bkpt(model_params,
                            num_end_to_skip,
                            num_between_to_skip,
                            indep):
    u1 = model_params[0]
    u2 = model_params[2]
    at_left = (u1 - indep[0] == num_end_to_skip + 1)
    at_right = (indep[-1] - u2 == num_end_to_skip + 1)

    at_mid = (u2 - u1 == num_between_to_skip - 1)

    at_boundary = (at_left or at_right or at_mid)

    return at_boundary


def create_fit_map(indep,
                   dep,
                   num_end_to_skip,
                   num_between_to_skip,
                   predict_horizon=30,
                   predict_level=0.0):
    fit_map = {}
    for model in [Model.OLS, Model.ONE_BKPT, Model.TWO_BKPT]:
        estimator = model.estimator(num_end_to_skip=num_end_to_skip,
                                    num_between_to_skip=num_between_to_skip)

        num_params = estimator.num_params
        mle = estimator.fit(indep, dep)
        loglik = estimator.loglikelihood

        # todo: get func, then apply to 30yrs ahead to predict
        func = estimator.get_func_for_params(mle)
        predict_indep = indep[-1] + predict_horizon
        prediction = func(predict_indep)

        level_prediction = predictions.level_predict(estimator,
                                                     indep,
                                                     dep,
                                                     level=predict_level)

        if model == Model.OLS:
            at_boundary = False
        elif model == Model.ONE_BKPT:
            at_boundary = check_boundary_one_bkpt(model_params=mle,
                                                  num_end_to_skip=num_end_to_skip,
                                                  indep=indep)
        elif model == Model.TWO_BKPT:
            at_boundary = check_boundary_two_bkpt(model_params=mle,
                                                  num_end_to_skip=num_end_to_skip,
                                                  num_between_to_skip=num_between_to_skip,
                                                  indep=indep)

        fit_map[model] = FitData(num_params=num_params,
                                 mle=mle,
                                 loglikelihood=loglik,
                                 at_boundary=at_boundary,
                                 prediction=prediction,
                                 level_prediction=level_prediction)

    #at_boundary = one_bkpt_at_boundary or two_bkpt_at_boundary

    return fit_map


class MultiModelEstimator():

    """
    Deprecated: favor new version below -- swap back names after testing.
    classdocs
    """

    def __init__(self,
                 models=None,
                 num_end_to_skip=10,
                 num_between_to_skip=10,
                 priors=None,
                 bic_type=BIC_TYPE.STANDARD):

        if models is None:
            models = [Model.OLS, Model.ONE_BKPT, Model.TWO_BKPT]

        self._models = models

        self._priors = priors

        self._bic_calculator = stats_util.BicCalculator(bic_type)

        if priors is not None and (len(priors) != len(models)):
            raise Exception("num priors must equal num models")

        self._estimators = [x.estimator(num_end_to_skip=num_end_to_skip,
                                        num_between_to_skip=num_between_to_skip)
                            for x in self._models]

        self._indep = None
        self._dep = None
        self._fitted_models_params = None
        self._bma_weights = None
        self._bics = None
        self._is_estimated = False

        self._not_fit_message = "Need to call 'fit()' first"

    def _clear(self):
        self._indep = None
        self._dep = None
        self._fitted_models_params = None
        self._bma_weights = None
        self._bics = None
        self._is_estimated = False

    def _set_data(self, indep, dep):
        self._clear()

        # we sort the data
        argsort_for_indep = indep.argsort()
        self._indep = indep[argsort_for_indep]
        self._dep = dep[argsort_for_indep]

    @property
    def model_names(self):
        return [x.display for x in self._models]

    @property
    def models(self):
        return self._models

    @property
    def priors(self):
        return self._priors

    @property
    def estimators(self):
        return self._estimators

    @property
    def bics(self):
        return self._bics

    def fit(self, indep, dep):
        self._set_data(indep, dep)

        self._fitted_models_params = [estimator.fit(indep, dep) for estimator
                                      in self._estimators]

        num_data = len(self._indep)

        # TODO: use num_params property
        num_params_arr = [len(estimator.param_names) for estimator
                          in self._estimators]

        loglik_arr = [estimator.loglikelihood for estimator
                      in self._estimators]

        bic_arr = [self._bic_calculator.bic(num_params=num_params,
                                            loglikelihood=loglik,
                                            num_data=num_data)
                   for num_params, loglik in zip(num_params_arr, loglik_arr)]

        self._bics = bic_arr
        self._bma_weights = stats_util.bma_weights(bic_arr,
                                                   priors=self._priors)

        self._is_estimated = True

        return self._fitted_models_params

    def bma_weights(self):
        if not self._is_estimated:
            raise Exception(self._not_fit_message)
        return self._bma_weights

    def bma_weights_df(self):
        if not self._is_estimated:
            raise Exception(self._not_fit_message)

        bic_name = "bma_wt"
        cols = [x.display + "_" + bic_name for x in self._models]
        df = pd.DataFrame([self._bma_weights], columns=cols)

        return df

    def random_bma_model(self):
        if not self._is_estimated:
            raise Exception(self._not_fit_message)

        rand_ind = np.random.choice(len(self._models),
                                    size=1,
                                    p=self._bma_weights)[0]

        rand_model = self._models[rand_ind]
        return rand_model

    def bic_model(self):
        """
        Selected using BMA weights, so incorporates any given model priors.
        """
        if not self._is_estimated:
            raise Exception(self._not_fit_message)

        model_map = {x: y for x, y in zip(self._models, self._bma_weights)}
        return stats_util.model_select_by_weight(model_map)


class MultiModelEstimatorNEW():

    """
    classdocs
    """

    def __init__(self,
                 models=None,
                 num_end_to_skip=10,
                 num_between_to_skip=10,
                 priors=None,
                 bic_types=None):

        if models is None:
            models = [Model.OLS, Model.ONE_BKPT, Model.TWO_BKPT]

        if bic_types is None:
            bic_types = [BIC_TYPE.STANDARD for dummy in models]

        self._models = models

        self._priors = priors

        self._bic_types = bic_types

        self._bic_calculators = [stats_util.BicCalculator(bic_type) for
                                 bic_type in self._bic_types]

        if priors is not None and (len(priors) != len(models)):
            raise Exception("num priors must equal num models")

        self._estimators = [x.estimator(num_end_to_skip=num_end_to_skip,
                                        num_between_to_skip=num_between_to_skip)
                            for x in self._models]

        self._indep = None
        self._dep = None
        self._fitted_models_params = None
        self._bma_weights = None
        self._bics = None
        self._is_estimated = False

        self._not_fit_message = "Need to call 'fit()' first"

    def _clear(self):
        self._indep = None
        self._dep = None
        self._fitted_models_params = None
        self._bma_weights = None
        self._bics = None
        self._is_estimated = False

    def _set_data(self, indep, dep):
        self._clear()

        # we sort the data
        argsort_for_indep = indep.argsort()
        self._indep = indep[argsort_for_indep]
        self._dep = dep[argsort_for_indep]

    @property
    def model_names(self):
        return [x.display for x in self._models]

    @property
    def models(self):
        return self._models

    @property
    def priors(self):
        return self._priors

    @property
    def estimators(self):
        return self._estimators

    @property
    def bics(self):
        return self._bics

    @property
    def bic_types(self):
        return self._bic_types

    def fit(self, indep, dep):
        self._set_data(indep, dep)

        self._fitted_models_params = [estimator.fit(indep, dep) for estimator
                                      in self._estimators]

        num_data = len(self._indep)

        # TODO: use num_params property
        num_params_arr = [len(estimator.param_names) for estimator
                          in self._estimators]

        loglik_arr = [estimator.loglikelihood for estimator
                      in self._estimators]

        bic_arr = [bic_calculator.bic(num_params=num_params,
                                      loglikelihood=loglik,
                                      num_data=num_data)
                   for (bic_calculator,
                        num_params,
                        loglik) in zip(self._bic_calculators,
                                       num_params_arr,
                                       loglik_arr)]

        self._bics = bic_arr
        self._bma_weights = stats_util.bma_weights(bic_arr,
                                                   priors=self._priors)

        self._is_estimated = True

        return self._fitted_models_params

    def bma_weights(self):
        if not self._is_estimated:
            raise Exception(self._not_fit_message)
        return self._bma_weights

    def bma_weights_df(self):
        if not self._is_estimated:
            raise Exception(self._not_fit_message)

        bic_name = "bma_wt"
        cols = [x.display + "_" + bic_name for x in self._models]
        df = pd.DataFrame([self._bma_weights], columns=cols)

        return df

    def random_bma_model(self):
        if not self._is_estimated:
            raise Exception(self._not_fit_message)

        rand_ind = np.random.choice(len(self._models),
                                    size=1,
                                    p=self._bma_weights)[0]

        rand_model = self._models[rand_ind]
        return rand_model

    def bic_model(self):
        """
        Selected using BMA weights, so incorporates any given model priors.
        """
        if not self._is_estimated:
            raise Exception(self._not_fit_message)

        model_map = {x: y for x, y in zip(self._models, self._bma_weights)}
        return stats_util.model_select_by_weight(model_map)


def bic_wts_and_choice(bic_type,
                       indep,
                       fit_map,
                       dep,
                       num_end_to_skip=10,
                       num_between_to_skip=10,
                       seed=None,
                       num_iter=1000,
                       verbose=False,
                       priors=None):
    """
    Most ``bic_type`` use the fit_map, so there is only one fit occurring.
    However, boot will compute many fits for sims, so is extremely slow.
    """

    # these are namedtuples FitData
    ols_fit_data = fit_map[Model.OLS]
    one_bkpt_fit_data = fit_map[Model.ONE_BKPT]
    two_bkpt_fit_data = fit_map[Model.TWO_BKPT]

    num_data = len(indep)

    # annoying code here
    if bic_type == BIC_TYPE.STANDARD:
        ols_bic = stats_util.bic(num_params=ols_fit_data.num_params,
                                 loglikelihood=ols_fit_data.loglikelihood,
                                 num_data=num_data)
        one_bkpt_bic = stats_util.bic(num_params=one_bkpt_fit_data.num_params,
                                      loglikelihood=one_bkpt_fit_data.loglikelihood,
                                      num_data=num_data)
        two_bkpt_bic = stats_util.bic(num_params=two_bkpt_fit_data.num_params,
                                      loglikelihood=two_bkpt_fit_data.loglikelihood,
                                      num_data=num_data)

    elif bic_type in [BIC_TYPE.HOS]:

        ols_bic = stats_util.bic_hos(num_params=ols_fit_data.num_params,
                                     loglikelihood=ols_fit_data.loglikelihood,
                                     num_data=num_data)
        one_bkpt_bic = stats_util.bic_hos(num_params=one_bkpt_fit_data.num_params,
                                          loglikelihood=one_bkpt_fit_data.loglikelihood,
                                          num_data=num_data)
        two_bkpt_bic = stats_util.bic_hos(num_params=two_bkpt_fit_data.num_params,
                                          loglikelihood=two_bkpt_fit_data.loglikelihood,
                                          num_data=num_data)
    elif bic_type in [BIC_TYPE.HOS_2]:

        ols_bic = stats_util.bic_hos_two(num_params=ols_fit_data.num_params,
                                         loglikelihood=ols_fit_data.loglikelihood,
                                         num_data=num_data)
        one_bkpt_bic = stats_util.bic_hos_two(num_params=one_bkpt_fit_data.num_params,
                                              loglikelihood=one_bkpt_fit_data.loglikelihood,
                                              num_data=num_data)
        two_bkpt_bic = stats_util.bic_hos_two(num_params=two_bkpt_fit_data.num_params,
                                              loglikelihood=two_bkpt_fit_data.loglikelihood,
                                              num_data=num_data)
    elif bic_type in [BIC_TYPE.LWZ]:

        ols_bic = stats_util.bic_lwz(num_params=ols_fit_data.num_params,
                                     loglikelihood=ols_fit_data.loglikelihood,
                                     num_data=num_data)
        one_bkpt_bic = stats_util.bic_lwz(num_params=one_bkpt_fit_data.num_params,
                                          loglikelihood=one_bkpt_fit_data.loglikelihood,
                                          num_data=num_data)
        two_bkpt_bic = stats_util.bic_lwz(num_params=two_bkpt_fit_data.num_params,
                                          loglikelihood=two_bkpt_fit_data.loglikelihood,
                                          num_data=num_data)

    else:
        raise Exception("Unsupported BIC type: ", bic_type)

    bma_wts = stats_util.bma_weights([ols_bic,
                                      one_bkpt_bic,
                                      two_bkpt_bic],
                                     priors=priors)
    model_map = {Model.OLS: ols_bic,
                 Model.ONE_BKPT: one_bkpt_bic,
                 Model.TWO_BKPT: two_bkpt_bic}

    bic_model = stats_util.model_select_info_criterion(model_map)

    return bma_wts, bic_model


class BkptModelFitter():

    """
    Fits OLS, ONE BKPT, and TWO BKPT models.  This class is immutable.  A new
    instance should be created for each dataset.

    Difference to MultiModelEstimator is that this method knows things about
    the models.  But perhaps the two methods can still be combined?
    """

    def __init__(self,
                 indep,
                 dep,
                 num_end_to_skip=10,
                 num_between_to_skip=10,
                 index=None,
                 priors=None,
                 bic_type=None):

        self._indep = indep
        self._dep = dep
        self._num_end_to_skip = num_end_to_skip
        self._num_between_to_skip = num_between_to_skip
        self._index = index

        self._priors = priors

        if priors is not None and (len(priors) != 3):
            raise Exception("num priors must equal three")

        if bic_type is None:
            bic_type = stats_util.BIC_TYPE.STANDARD
        self._bic_calculator = stats_util.BicCalculator(bic_type)

        self._ols_estimator = OLSRegressionEstimator()
        self._one_bkpt_estimator = OneBkptSegRegEstimator(num_end_to_skip=num_end_to_skip)

        self._two_bkpt_estimator = TwoBkptSegRegEstimator(num_end_to_skip=num_end_to_skip,
                                                          num_between_to_skip=num_between_to_skip)

        self._initialize()

    def _initialize(self):

        (ols_intercept,
         ols_slope,
         ols_resid_stddev) = self._ols_estimator.fit(self._indep, self._dep)

        (obp_u,
         obp_v,
         obp_m1,
         obp_m2,
         obp_resid_stddev) = self._one_bkpt_estimator.fit(self._indep,
                                                          self._dep)

        (tbp_u1,
         tbp_v1,
         tbp_u2,
         tbp_v2,
         tbp_m1,
         tbp_m2,
         tbp_resid_stddev) = self._two_bkpt_estimator.fit(self._indep,
                                                          self._dep)

        self._ols_core_params = [ols_intercept, ols_slope]
        self._ols_resid_stddev = ols_resid_stddev

        self._one_bkpt_core_params = [obp_u, obp_v, obp_m1, obp_m2]
        self._one_bkpt_resid_stddev = obp_resid_stddev

        self._two_bkpt_core_params = [tbp_u1,
                                      tbp_v1,
                                      tbp_u2,
                                      tbp_v2,
                                      tbp_m1,
                                      tbp_m2]
        self._two_bkpt_resid_stddev = tbp_resid_stddev

        num_data = len(self._indep)
        num_ols_params = len(self._ols_estimator.param_names)
        num_one_bkpt_params = len(self._one_bkpt_estimator.param_names)
        num_two_bkpt_params = len(self._two_bkpt_estimator.param_names)

        ols_loglik = self._ols_estimator.loglikelihood
        one_bkpt_loglik = self._one_bkpt_estimator.loglikelihood
        two_bkpt_loglik = self._two_bkpt_estimator.loglikelihood

        # NOTE: the primary "bic" used here depends on calculator setting
        self._ols_bic = self._bic_calculator.bic(num_ols_params,
                                                 ols_loglik,
                                                 num_data)
        self._one_bkpt_bic = self._bic_calculator.bic(num_one_bkpt_params,
                                                      one_bkpt_loglik,
                                                      num_data)
        self._two_bkpt_bic = self._bic_calculator.bic(num_two_bkpt_params,
                                                      two_bkpt_loglik,
                                                      num_data)

        self._ols_bic_lwz = stats_util.bic_lwz(num_ols_params,
                                               ols_loglik,
                                               num_data)
        self._one_bkpt_bic_lwz = stats_util.bic_lwz(num_one_bkpt_params,
                                                    one_bkpt_loglik,
                                                    num_data)
        self._two_bkpt_bic_lwz = stats_util.bic_lwz(num_two_bkpt_params,
                                                    two_bkpt_loglik,
                                                    num_data)

        self._ols_bic_hos = stats_util.bic_hos(num_ols_params,
                                               ols_loglik,
                                               num_data)
        self._one_bkpt_bic_hos = stats_util.bic_hos(num_one_bkpt_params,
                                                    one_bkpt_loglik,
                                                    num_data)
        self._two_bkpt_bic_hos = stats_util.bic_hos(num_two_bkpt_params,
                                                    two_bkpt_loglik,
                                                    num_data)

        self._ols_aicc = stats_util.aicc(num_params=num_ols_params,
                                         loglikelihood=ols_loglik,
                                         num_data=num_data)
        self._one_bkpt_aicc = stats_util.aicc(num_params=num_one_bkpt_params,
                                              loglikelihood=one_bkpt_loglik,
                                              num_data=num_data)
        self._two_bkpt_aicc = stats_util.aicc(num_params=num_two_bkpt_params,
                                              loglikelihood=two_bkpt_loglik,
                                              num_data=num_data)

        bma_wts = stats_util.bma_weights([self._ols_bic,
                                          self._one_bkpt_bic,
                                          self._two_bkpt_bic],
                                         priors=self._priors)

        self._ols_bma_wt = bma_wts[0]
        self._one_bkpt_bma_wt = bma_wts[1]
        self._two_bkpt_bma_wt = bma_wts[2]

        self._ols_loglik = ols_loglik
        self._one_bkpt_loglik = one_bkpt_loglik
        self._two_bkpt_loglik = two_bkpt_loglik

        ind_of_min = np.where(bma_wts == min(bma_wts))[0][0]
        num = len(bma_wts)
        throw_out_worst_priors = np.ones(num) / (num - 1)
        throw_out_worst_priors[ind_of_min] = 0.0

        self._throw_out_worst_bma_wts = stats_util.bma_weights([self._ols_bic,
                                                                self._one_bkpt_bic,
                                                                self._two_bkpt_bic],
                                                               priors=throw_out_worst_priors)

    @property
    def throw_out_worst_bma_wts(self):
        return self._throw_out_worst_bma_wts

    @property
    def ols_estimator(self):
        return self._ols_estimator

    @property
    def one_bkpt_estimator(self):
        return self._one_bkpt_estimator

    @property
    def two_bkpt_estimator(self):
        return self._two_bkpt_estimator

    @property
    def ols_core_params(self):
        return self._ols_core_params

    @property
    def one_bkpt_core_params(self):
        return self._one_bkpt_core_params

    @property
    def two_bkpt_core_params(self):
        return self._two_bkpt_core_params

    @property
    def ols_resid_stddev(self):
        return self._ols_resid_stddev

    @property
    def one_bkpt_resid_stddev(self):
        return self._one_bkpt_resid_stddev

    @property
    def two_bkpt_resid_stddev(self):
        return self._two_bkpt_resid_stddev

    @property
    def ols_loglikelihood(self):
        return self._ols_loglik

    @property
    def one_bkpt_loglikelihood(self):
        return self._one_bkpt_loglik

    @property
    def two_bkpt_loglikelihood(self):
        return self._two_bkpt_loglik

    @property
    def ols_aicc(self):
        return self._ols_aicc

    @property
    def one_bkpt_aicc(self):
        return self._one_bkpt_aicc

    @property
    def two_bkpt_aicc(self):
        return self._two_bkpt_aicc

    def aicc_model(self):

        model_map = {Model.OLS: self.ols_aicc,
                     Model.ONE_BKPT: self.one_bkpt_aicc,
                     Model.TWO_BKPT: self.two_bkpt_aicc}
        return stats_util.model_select_info_criterion(model_map)

    @property
    def ols_bic(self):
        return self._ols_bic

    @property
    def one_bkpt_bic(self):
        return self._one_bkpt_bic

    @property
    def two_bkpt_bic(self):
        return self._two_bkpt_bic

    @property
    def ols_bma_wt(self):
        return self._ols_bma_wt

    @property
    def one_bkpt_bma_wt(self):
        return self._one_bkpt_bma_wt

    @property
    def two_bkpt_bma_wt(self):
        return self._two_bkpt_bma_wt

    def bic_model(self):
        """
        Selected using BMA weights, so incorporates any given model priors.
        """
        model_map = {Model.OLS: self.ols_bma_wt,
                     Model.ONE_BKPT: self.one_bkpt_bma_wt,
                     Model.TWO_BKPT: self.two_bkpt_bma_wt}
        return stats_util.model_select_by_weight(model_map)

    def bic_estimator(self):
        bic_chosen_model = self.bic_model()
        if bic_chosen_model == Model.OLS:
            result = self._ols_estimator
        elif bic_chosen_model == Model.ONE_BKPT:
            result = self._one_bkpt_estimator
        elif bic_chosen_model == Model.TWO_BKPT:
            result = self._two_bkpt_estimator
        return result

    def bic_weight(self):
        bic_chosen_model = self.bic_model()
        if bic_chosen_model == Model.OLS:
            result = self.ols_bma_wt
        elif bic_chosen_model == Model.ONE_BKPT:
            result = self.one_bkpt_bma_wt
        elif bic_chosen_model == Model.TWO_BKPT:
            result = self.two_bkpt_bma_wt
        return result

    @property
    def ols_bic_lwz(self):
        return self._ols_bic_lwz

    @property
    def one_bkpt_bic_lwz(self):
        return self._one_bkpt_bic_lwz

    @property
    def two_bkpt_bic_lwz(self):
        return self._two_bkpt_bic_lwz

    def bic_lwz_model(self):

        model_map = {Model.OLS: self.ols_bic_lwz,
                     Model.ONE_BKPT: self.one_bkpt_bic_lwz,
                     Model.TWO_BKPT: self.two_bkpt_bic_lwz}
        return stats_util.model_select_info_criterion(model_map)

    @property
    def ols_bic_hos(self):
        return self._ols_bic_hos

    @property
    def one_bkpt_bic_hos(self):
        return self._one_bkpt_bic_hos

    @property
    def two_bkpt_bic_hos(self):
        return self._two_bkpt_bic_hos

    def bic_hos_model(self):

        model_map = {Model.OLS: self.ols_bic_hos,
                     Model.ONE_BKPT: self.one_bkpt_bic_hos,
                     Model.TWO_BKPT: self.two_bkpt_bic_hos}
        return stats_util.model_select_info_criterion(model_map)

    def to_dataframe(self):
        ols_cols = [Model.OLS.display + "_" + x
                    for x in self._ols_estimator.param_names]
        one_bkpt_cols = [Model.ONE_BKPT.display + "_" + x
                         for x in self._one_bkpt_estimator.param_names]
        two_bkpt_cols = [Model.TWO_BKPT.display + "_" + x
                         for x in self._two_bkpt_estimator.param_names]

        est_cols = ols_cols
        est_cols.extend(one_bkpt_cols)
        est_cols.extend(two_bkpt_cols)

        row = list(self.ols_core_params)
        row.append(self._ols_resid_stddev)
        row.extend(self._one_bkpt_core_params)
        row.append(self._one_bkpt_resid_stddev)
        row.extend(self._two_bkpt_core_params)
        row.append(self._two_bkpt_resid_stddev)

        est_df = pd.DataFrame([row], columns=est_cols)

        bic_name = "bic"
        est_df[Model.OLS.display + "_" + bic_name] = self._ols_bic
        est_df[Model.ONE_BKPT.display + "_" + bic_name] = self._one_bkpt_bic
        est_df[Model.TWO_BKPT.display + "_" + bic_name] = self._two_bkpt_bic

        bma_name = "bma_wt"
        est_df[Model.OLS.display + "_" + bma_name] = self._ols_bma_wt
        est_df[Model.ONE_BKPT.display + "_" + bma_name] = self._one_bkpt_bma_wt
        est_df[Model.TWO_BKPT.display + "_" + bma_name] = self._two_bkpt_bma_wt

        est_df["bic_model"] = self.bic_model()

        aicc_name = "aicc"
        est_df[Model.OLS.display + "_" + aicc_name] = self._ols_aicc
        est_df[Model.ONE_BKPT.display + "_" + aicc_name] = self._one_bkpt_aicc
        est_df[Model.TWO_BKPT.display + "_" + aicc_name] = self._two_bkpt_aicc
        est_df["aicc_model"] = self.aicc_model()

        est_df["ols_loglik"] = self.ols_loglikelihood
        est_df["one_bkpt_loglik"] = self.one_bkpt_loglikelihood
        est_df["two_bkpt_loglik"] = self.two_bkpt_loglikelihood

        if self._index is not None:
            est_df.index = [self._index]

        return est_df
