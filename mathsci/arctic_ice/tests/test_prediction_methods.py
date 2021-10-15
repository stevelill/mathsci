"""
Unit tests for prediction methods.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import unittest

from numpy.testing import assert_allclose
import numpy as np

from segreg.model.segreg_estimator import OneBkptSegRegEstimator
from mathsci.arctic_ice import prediction_methods


class TestPredictionMethods(unittest.TestCase):

    def test_zero_ice(self):

        level = 1.0

        ########################################################################
        # OLS
        ########################################################################
        v = 5.0
        m = -0.01

        ols_params = [v, m]

        expected = (level - v) / m
        computed = prediction_methods.zero_ice_date_ols(ols_params, level)
        self.assertAlmostEqual(computed, expected, delta=1.0e-14)

        ols_params2 = [ols_params, ols_params]
        expected2 = [expected, expected]
        computed2 = prediction_methods.zero_ice_date_ols(ols_params2, level)
        assert_allclose(computed2, expected2, rtol=0.0, atol=1.0e-14)

        ########################################################################
        # ONE BKPT
        ########################################################################

        u = 3.0
        v = 5.0
        m1 = -0.02
        m2 = -0.04
        one_bkpt_params = [u, v, m1, m2]

        expected = (level - v) / m2 + u
        computed = prediction_methods.zero_ice_date_one_bkpt(one_bkpt_params,
                                                             level)
        self.assertAlmostEqual(computed, expected, delta=1.0e-14)

        one_bkpt_params2 = [one_bkpt_params, one_bkpt_params]
        expected2 = [expected, expected]
        computed2 = prediction_methods.zero_ice_date_one_bkpt(one_bkpt_params2,
                                                              level)
        assert_allclose(computed2, expected2, rtol=0.0, atol=1.0e-14)

        ########################################################################
        # TWO BKPT
        ########################################################################

        u1 = 1.0
        v1 = 10.0
        u2 = 3.0
        v2 = 5.0
        m1 = -0.02
        m2 = -0.04
        two_bkpt_params = [u1, v1, u2, v2, m1, m2]

        expected = (level - v2) / m2 + u2
        computed = prediction_methods.zero_ice_date_two_bkpt(two_bkpt_params,
                                                             level)
        self.assertAlmostEqual(computed, expected, delta=1.0e-14)

        two_bkpt_params2 = [two_bkpt_params, two_bkpt_params]
        expected2 = [expected, expected]
        computed2 = prediction_methods.zero_ice_date_two_bkpt(two_bkpt_params2,
                                                              level)
        assert_allclose(computed2, expected2, rtol=0.0, atol=1.0e-14)

    #@unittest.skip("skipping")
    def test_predict_first_zero(self):
        indep = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
                          13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.,
                          24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34.,
                          35., 36., 37., 38., 39., 40.])

        dep = np.array([7.3, 7.9, 7.1, 6.8, 7.5, 6.7, 6.8, 7.5, 7.1, 7.4, 7.2,
                        6., 6.4, 7.5, 6.4, 7.1, 5.9, 7.4, 6.7, 6.8, 6.1, 6.,
                        6.6, 5.8, 6., 5.7, 5.5, 6., 4.1, 4.4, 5., 4.8, 4.2,
                        3.8, 5., 5.4, 4.5, 4.3, 4.6, 4.7, 4.3])

        estimator = OneBkptSegRegEstimator(num_end_to_skip=10)

        seed = 23094234

        (first_zero_prediction,
         zero_time) = prediction_methods.predict_first_zero(indep=indep,
                                                            dep=dep,
                                                            estimator=estimator,
                                                            num_boot_sims=1000,
                                                            seed=seed)

        expected_first_zero = 72.118
        expected_zero_time = 75.79188675258828

        self.assertAlmostEqual(first_zero_prediction,
                               expected_first_zero,
                               delta=1.0e-12)

        self.assertAlmostEqual(zero_time,
                               expected_zero_time,
                               delta=1.0e-12)

    def test_predict_first_zero2(self):
        indep = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
                          13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.,
                          24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34.,
                          35., 36., 37., 38., 39., 40.])

        dep = np.array([7.3, 7.9, 7.1, 6.8, 7.5, 6.7, 6.8, 7.5, 7.1, 7.4, 7.2,
                        6., 6.4, 7.5, 6.4, 7.1, 5.9, 7.4, 6.7, 6.8, 6.1, 6.,
                        6.6, 5.8, 6., 5.7, 5.5, 6., 4.1, 4.4, 5., 4.8, 4.2,
                        3.8, 5.5, 5.4, 4.5, 4.5, 5.6, 5.7, 5.3])

        estimator = OneBkptSegRegEstimator(num_end_to_skip=10)

        seed = 23094234

        (first_zero_prediction,
         zero_time) = prediction_methods.predict_first_zero(indep=indep,
                                                            dep=dep,
                                                            estimator=estimator,
                                                            num_boot_sims=1000,
                                                            seed=seed)

        expected_first_zero = prediction_methods.ZERO_ICE_INFINITY_PROXY
        expected_zero_time = prediction_methods.ZERO_ICE_INFINITY_PROXY

        self.assertEqual(first_zero_prediction, expected_first_zero)
        self.assertEqual(zero_time, expected_zero_time)

        (first_zero_prediction,
         zero_time) = prediction_methods.predict_first_zero(indep=indep,
                                                            dep=dep,
                                                            estimator=estimator,
                                                            num_boot_sims=1000,
                                                            seed=seed,
                                                            convert_infinity=False)

        expected_first_zero = 441.592
        expected_zero_time = 549.4138929087824

        self.assertAlmostEqual(first_zero_prediction,
                               expected_first_zero,
                               delta=1.0e-12)

        self.assertAlmostEqual(zero_time,
                               expected_zero_time,
                               delta=1.0e-12)


if __name__ == "__main__":
    unittest.main()
