"""
Unit tests for model fitting code.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import unittest

import numpy as np
from numpy.testing import assert_allclose

from segreg.model import OLSRegressionEstimator
from segreg.model import OneBkptSegRegEstimator
from segreg.model import TwoBkptSegRegEstimator
from segreg import data

from mathsci.segreg.model import Model
from mathsci.segreg.models import BkptModelFitter, MultiModelEstimator
from mathsci.statistics import stats_util
from mathsci.segreg import models

from mathsci.statistics.stats_util import BIC_TYPE


class TestModels(unittest.TestCase):

    def setUp(self):
        indep, dep = data.test1()
        self._indep = indep
        self._dep = dep

        indep2, dep2 = data.test2()
        self._indep2 = indep2
        self._dep2 = dep2

    def test_bic_wts_and_choice(self):
        # TODO: create larger dataset for this

        num_end_to_skip = 5
        num_between_to_skip = 5

        print("num data: ", len(self._indep2))

        fit_map = models.create_fit_map(indep=self._indep2,
                                        dep=self._dep2,
                                        num_end_to_skip=num_end_to_skip,
                                        num_between_to_skip=num_between_to_skip)

        bic_type = BIC_TYPE.STANDARD

        (bma_wts, bic_model) = models.bic_wts_and_choice(bic_type=bic_type,
                                                         indep=self._indep2,
                                                         dep=self._dep2,
                                                         fit_map=fit_map,
                                                         num_end_to_skip=num_end_to_skip,
                                                         num_between_to_skip=num_between_to_skip)

        expected_wts = [2.9683202359216508e-21,
                        0.9368464146759923,
                        0.0631535853240076]

        expected_bic_model = Model.ONE_BKPT

        assert_allclose(bma_wts, expected_wts, rtol=0.0, atol=1.0e-12)
        self.assertEqual(bic_model, expected_bic_model)

        ###########################
        bic_type = BIC_TYPE.HOS

        (bma_wts, bic_model) = models.bic_wts_and_choice(bic_type=bic_type,
                                                         indep=self._indep2,
                                                         dep=self._dep2,
                                                         fit_map=fit_map,
                                                         num_end_to_skip=num_end_to_skip,
                                                         num_between_to_skip=num_between_to_skip)

        expected_wts = [3.147201593873088e-20,
                        0.9933040558769494,
                        0.0066959441230506084]

        expected_bic_model = Model.ONE_BKPT

        assert_allclose(bma_wts, expected_wts, rtol=0.0, atol=1.0e-12)
        self.assertEqual(bic_model, expected_bic_model)

    def test_bkpt_model_fitter_priors(self):

        num_end_to_skip = 10
        num_between_to_skip = 10

        ols_estimator = OLSRegressionEstimator()
        (ols_intercept,
         ols_slope,
         ols_resid_stddev) = ols_estimator.fit(self._indep, self._dep)

        one_bkpt_estimator = OneBkptSegRegEstimator(num_end_to_skip=num_end_to_skip)
        (obp_u,
         obp_v,
         obp_m1,
         obp_m2,
         obp_resid_stddev) = one_bkpt_estimator.fit(self._indep, self._dep)

        two_bkpt_estimator = TwoBkptSegRegEstimator(num_end_to_skip=num_end_to_skip,
                                                    num_between_to_skip=num_between_to_skip)

        (tbp_u1,
         tbp_v1,
         tbp_u2,
         tbp_v2,
         tbp_m1,
         tbp_m2,
         tbp_resid_stddev) = two_bkpt_estimator.fit(self._indep, self._dep)

        fitter = BkptModelFitter(indep=self._indep,
                                 dep=self._dep,
                                 num_end_to_skip=num_end_to_skip,
                                 num_between_to_skip=num_between_to_skip)

        expected = Model.ONE_BKPT
        computed = fitter.bic_model()
        self.assertEqual(expected, computed)

        fitter = BkptModelFitter(indep=self._indep,
                                 dep=self._dep,
                                 num_end_to_skip=num_end_to_skip,
                                 num_between_to_skip=num_between_to_skip,
                                 priors=[1.0, 0.0, 0.0])

        expected = Model.OLS
        computed = fitter.bic_model()
        self.assertEqual(expected, computed)

        fitter = BkptModelFitter(indep=self._indep,
                                 dep=self._dep,
                                 num_end_to_skip=num_end_to_skip,
                                 num_between_to_skip=num_between_to_skip,
                                 priors=[0.0, 1.0, 0.0])

        expected = Model.ONE_BKPT
        computed = fitter.bic_model()
        self.assertEqual(expected, computed)

        fitter = BkptModelFitter(indep=self._indep,
                                 dep=self._dep,
                                 num_end_to_skip=num_end_to_skip,
                                 num_between_to_skip=num_between_to_skip,
                                 priors=[0.0, 0.0, 1.0])

        expected = Model.TWO_BKPT
        computed = fitter.bic_model()
        self.assertEqual(expected, computed)

    def test_bkpt_model_fitter(self):

        num_end_to_skip = 10
        num_between_to_skip = 10

        ols_estimator = OLSRegressionEstimator()
        (ols_intercept,
         ols_slope,
         ols_resid_stddev) = ols_estimator.fit(self._indep, self._dep)

        one_bkpt_estimator = OneBkptSegRegEstimator(num_end_to_skip=num_end_to_skip)
        (obp_u,
         obp_v,
         obp_m1,
         obp_m2,
         obp_resid_stddev) = one_bkpt_estimator.fit(self._indep, self._dep)

        two_bkpt_estimator = TwoBkptSegRegEstimator(num_end_to_skip=num_end_to_skip,
                                                    num_between_to_skip=num_between_to_skip)

        (tbp_u1,
         tbp_v1,
         tbp_u2,
         tbp_v2,
         tbp_m1,
         tbp_m2,
         tbp_resid_stddev) = two_bkpt_estimator.fit(self._indep, self._dep)

        fitter = BkptModelFitter(indep=self._indep,
                                 dep=self._dep,
                                 num_end_to_skip=num_end_to_skip,
                                 num_between_to_skip=num_between_to_skip)

        rtol = 0.0
        atol = 1.0e-14

        assert_allclose([ols_intercept, ols_slope],
                        fitter.ols_core_params, rtol=rtol, atol=atol)

        self.assertAlmostEqual(ols_resid_stddev,
                               fitter.ols_resid_stddev, delta=atol)

        assert_allclose([obp_u, obp_v, obp_m1, obp_m2],
                        fitter.one_bkpt_core_params, rtol=rtol, atol=atol)

        self.assertAlmostEqual(obp_resid_stddev,
                               fitter.one_bkpt_resid_stddev, delta=atol)

        assert_allclose([tbp_u1, tbp_v1, tbp_u2, tbp_v2, tbp_m1, tbp_m2],
                        fitter.two_bkpt_core_params, rtol=rtol, atol=atol)

        self.assertAlmostEqual(tbp_resid_stddev,
                               fitter.two_bkpt_resid_stddev, delta=atol)

        ########################################################################
        # BIC -- include resid stddev as param
        ########################################################################

        num_data = len(self._indep)
        num_ols_params = 2 + 1
        num_one_bkpt_params = 4 + 1
        num_two_bkpt_params = 6 + 1

        ols_loglik = ols_estimator.loglikelihood
        one_bkpt_loglik = one_bkpt_estimator.loglikelihood
        two_bkpt_loglik = two_bkpt_estimator.loglikelihood

        ols_bic = stats_util.bic(num_ols_params, ols_loglik, num_data)
        one_bkpt_bic = stats_util.bic(num_one_bkpt_params,
                                      one_bkpt_loglik,
                                      num_data)
        two_bkpt_bic = stats_util.bic(num_two_bkpt_params,
                                      two_bkpt_loglik,
                                      num_data)

        self.assertAlmostEqual(ols_bic, fitter.ols_bic, delta=atol)
        self.assertAlmostEqual(one_bkpt_bic, fitter.one_bkpt_bic, delta=atol)
        self.assertAlmostEqual(two_bkpt_bic, fitter.two_bkpt_bic, delta=atol)

        bma_wts = stats_util.bma_weights(bic_arr=[ols_bic,
                                                  one_bkpt_bic,
                                                  two_bkpt_bic])

        expected = Model.ONE_BKPT
        computed = fitter.bic_model()
        self.assertEqual(expected, computed)

        ########################################################################
        # to_dataframe
        ########################################################################

        df = fitter.to_dataframe()

        self.assertAlmostEqual(ols_intercept, df.ols_intercept.values, delta=atol)
        self.assertAlmostEqual(ols_slope, df.ols_slope.values, delta=atol)
        self.assertAlmostEqual(ols_resid_stddev, df.ols_sigma.values, delta=atol)

        self.assertAlmostEqual(obp_u, df.one_bkpt_u.values, delta=atol)
        self.assertAlmostEqual(obp_v, df.one_bkpt_v.values, delta=atol)
        self.assertAlmostEqual(obp_m1, df.one_bkpt_m1.values, delta=atol)
        self.assertAlmostEqual(obp_m2, df.one_bkpt_m2.values, delta=atol)
        self.assertAlmostEqual(obp_resid_stddev, df.one_bkpt_sigma.values, delta=atol)

        self.assertAlmostEqual(tbp_u1, df.two_bkpt_u1.values, delta=atol)
        self.assertAlmostEqual(tbp_v1, df.two_bkpt_v1.values, delta=atol)
        self.assertAlmostEqual(tbp_u2, df.two_bkpt_u2.values, delta=atol)
        self.assertAlmostEqual(tbp_v2, df.two_bkpt_v2.values, delta=atol)
        self.assertAlmostEqual(tbp_m1, df.two_bkpt_m1.values, delta=atol)
        self.assertAlmostEqual(tbp_m2, df.two_bkpt_m2.values, delta=atol)
        self.assertAlmostEqual(tbp_resid_stddev, df.two_bkpt_sigma.values, delta=atol)

        self.assertAlmostEqual(ols_bic, df.ols_bic.values, delta=atol)
        self.assertAlmostEqual(one_bkpt_bic, df.one_bkpt_bic.values, delta=atol)
        self.assertAlmostEqual(two_bkpt_bic, df.two_bkpt_bic.values, delta=atol)

        self.assertAlmostEqual(bma_wts[0], df.ols_bma_wt.values, delta=atol)
        self.assertAlmostEqual(bma_wts[1], df.one_bkpt_bma_wt.values, delta=atol)
        self.assertAlmostEqual(bma_wts[2], df.two_bkpt_bma_wt.values, delta=atol)

    def _test_multi_model_estimator(self, bic_type):

        num_end_to_skip = 10
        num_between_to_skip = 10

        ols_estimator = OLSRegressionEstimator()
        (ols_intercept,
         ols_slope,
         ols_resid_stddev) = ols_estimator.fit(self._indep, self._dep)

        one_bkpt_estimator = OneBkptSegRegEstimator(num_end_to_skip=num_end_to_skip)
        (obp_u,
         obp_v,
         obp_m1,
         obp_m2,
         obp_resid_stddev) = one_bkpt_estimator.fit(self._indep, self._dep)

        two_bkpt_estimator = TwoBkptSegRegEstimator(num_end_to_skip=num_end_to_skip,
                                                    num_between_to_skip=num_between_to_skip)

        (tbp_u1,
         tbp_v1,
         tbp_u2,
         tbp_v2,
         tbp_m1,
         tbp_m2,
         tbp_resid_stddev) = two_bkpt_estimator.fit(self._indep, self._dep)

        mme = MultiModelEstimator(models=[Model.OLS,
                                          Model.ONE_BKPT,
                                          Model.TWO_BKPT],
                                  bic_type=bic_type)

        mme.fit(indep=self._indep, dep=self._dep)

        rtol = 0.0
        atol = 1.0e-14

        ########################################################################
        # BIC -- include resid stddev as param
        ########################################################################

        num_data = len(self._indep)
        num_ols_params = 2 + 1
        num_one_bkpt_params = 4 + 1
        num_two_bkpt_params = 6 + 1

        ols_loglik = ols_estimator.loglikelihood
        one_bkpt_loglik = one_bkpt_estimator.loglikelihood
        two_bkpt_loglik = two_bkpt_estimator.loglikelihood

        ols_bic = stats_util.bic(num_ols_params, ols_loglik, num_data)
        one_bkpt_bic = stats_util.bic(num_one_bkpt_params,
                                      one_bkpt_loglik,
                                      num_data)
        two_bkpt_bic = stats_util.bic(num_two_bkpt_params,
                                      two_bkpt_loglik,
                                      num_data)

        assert_allclose([ols_bic, one_bkpt_bic, two_bkpt_bic],
                        mme.bics,
                        rtol=rtol, atol=atol)

        bma_wts = stats_util.bma_weights(bic_arr=[ols_bic,
                                                  one_bkpt_bic,
                                                  two_bkpt_bic])

        assert_allclose(bma_wts, mme.bma_weights(), rtol=rtol, atol=atol)

        expected = Model.ONE_BKPT
        computed = mme.bic_model()
        self.assertEqual(expected, computed)

        df = mme.bma_weights_df()

        assert_allclose(np.ravel(df), bma_wts, rtol=rtol, atol=atol)

    def test_multi_model_estimator(self):
        self._test_multi_model_estimator(bic_type=BIC_TYPE.STANDARD)

    def test_multi_model_estimator2(self):
        self._test_multi_model_estimator(bic_type=BIC_TYPE.HYBRID)


if __name__ == "__main__":
    unittest.main()
