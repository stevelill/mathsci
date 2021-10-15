"""
Unit tests for stats util routines.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import unittest

import numpy as np
import numpy.testing

from mathsci.statistics import stats_util


class TestStatsUtil(unittest.TestCase):

    def test_rel_bic(self):
        bics = [[40.0, 44.0, 35.0],
                [2.0, 1.0, 3.0]]

        expected = [[5.0, 9.0, 0.0],
                    [1.0, 0.0, 2.0]]
        computed = stats_util.rel_bic(bics)

        numpy.testing.assert_allclose(computed,
                                      expected,
                                      rtol=0.0,
                                      atol=1.0e-14)

        bics = [[40.0, 44.0, 35.0]]
        computed = stats_util.rel_bic(bics)

        numpy.testing.assert_allclose(computed,
                                      [expected[0]],
                                      rtol=0.0,
                                      atol=1.0e-14)

    def test_bma_weights(self):
        bics = [[40.0, 44.0, 35.0],
                [2.0, 1.0, 3.0]]

        row1 = np.exp(-0.5 * np.array(bics[0]))
        expect1 = row1 / np.sum(row1)

        row2 = np.exp(-0.5 * np.array(bics[1]))
        expect2 = row2 / np.sum(row2)

        expected = [expect1, expect2]

        computed = stats_util.bma_weights(bics)

        numpy.testing.assert_allclose(computed,
                                      expected,
                                      rtol=0.0,
                                      atol=1.0e-14)

        # 1d array
        bics2 = bics[0]
        computed = stats_util.bma_weights(bics2)

        expected2 = expected[0]
        numpy.testing.assert_allclose(computed,
                                      expected2,
                                      rtol=0.0,
                                      atol=1.0e-14)

        # PRIORS
        weight = 1.0 / 3.0
        priors = np.ones(3) * weight
        #print(priors)

        computed = stats_util.bma_weights(bics, priors)
        numpy.testing.assert_allclose(computed,
                                      expected,
                                      rtol=0.0,
                                      atol=1.0e-14)

        computed = stats_util.bma_weights(bics2, priors)
        numpy.testing.assert_allclose(computed,
                                      expected2,
                                      rtol=0.0,
                                      atol=1.0e-14)

        priors = [0.1, 0.5, 0.4]
        #print(priors)
        row1 = np.exp(-0.5 * np.array(bics[0])) * np.array(priors)
        expect1 = row1 / np.sum(row1)

        row2 = np.exp(-0.5 * np.array(bics[1])) * np.array(priors)
        expect2 = row2 / np.sum(row2)

        expected = [expect1, expect2]

        computed = stats_util.bma_weights(bics, priors)

        numpy.testing.assert_allclose(computed,
                                      expected,
                                      rtol=0.0,
                                      atol=1.0e-14)

        ## single element
        computed = stats_util.bma_weights(45.0)

        numpy.testing.assert_allclose(computed,
                                      1.0,
                                      rtol=0.0,
                                      atol=1.0e-14)
        

    def test_bic(self):
        k = 4
        n = 40
        l = 20

        bic1 = stats_util.bic(k, l, n)
        bic_calculator = stats_util.BicCalculator(stats_util.BIC_TYPE.STANDARD)
        bic2 = bic_calculator.bic(k, l, n)
        self.assertEqual(bic1, bic2)

        bic1 = stats_util.bic_lwz(k, l, n)
        bic_calculator = stats_util.BicCalculator(stats_util.BIC_TYPE.LWZ)
        bic2 = bic_calculator.bic(k, l, n)
        self.assertEqual(bic1, bic2)

        bic1 = stats_util.bic_hos(k, l, n)
        bic_calculator = stats_util.BicCalculator(stats_util.BIC_TYPE.HOS)
        bic2 = bic_calculator.bic(k, l, n)
        self.assertEqual(bic1, bic2)


if __name__ == "__main__":
    unittest.main()
