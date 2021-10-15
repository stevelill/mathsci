"""
Testing Ornstein-Uhlenbeck code.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import unittest

from mathsci.statistics.ou import OU


class TestOU(unittest.TestCase):

    def test_basics(self):
        mean_rev_rate = 0.1
        mean_rev_level = 0.05
        vol = 0.02
        init_val = 0.1

        ou = OU(mean_rev_rate=mean_rev_rate,
                mean_rev_level=mean_rev_level,
                vol=vol)

        expected_half_life = 6.931471805599452
        expected_eq_vol = 0.044721359549995794

        computed_half_life = ou.half_life()
        computed_eq_vol = ou.equilibrium_vol()

        tol = 1.0e-14
        self.assertAlmostEqual(expected_half_life,
                               computed_half_life,
                               delta=tol)
        self.assertAlmostEqual(expected_eq_vol,
                               computed_eq_vol,
                               delta=tol)

        mean, variance = ou.forecast(init_val=init_val,
                                     time_to_horizon=1.0)

        expected_mean = 0.09524187090179799
        expected_variance = 0.0003625384938440364

        self.assertAlmostEqual(expected_mean,
                               mean,
                               delta=tol)
        self.assertAlmostEqual(expected_variance,
                               variance,
                               delta=tol)


if __name__ == "__main__":
    unittest.main()
