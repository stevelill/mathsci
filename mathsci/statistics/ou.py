"""
Ornstein-Uhlenbeck model.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numpy as np
import pandas as pd


class OU(object):
    """
    TODO: make it nicer

    Parameters
    ----------
    mean_rev_rate: float
    mean_rev_level: float
    vol: float
    """

    def __init__(self, mean_rev_rate, mean_rev_level, vol):

        self._mean_rev_rate = mean_rev_rate
        self._mean_rev_level = mean_rev_level
        self._vol = vol

        self._initialize()

    def _initialize(self):
        self._half_life = np.log(2.0) / self._mean_rev_rate
        self._eq_vol = self._vol / np.sqrt(2.0 * self._mean_rev_rate)

    def get_mean_rev_rate(self):
        return self._mean_rev_rate

    def get_mean_rev_level(self):
        return self._mean_rev_level

    def get_vol(self):
        return self._vol

    mean_rev_rate = property(get_mean_rev_rate, None,
                             None, "mean_rev_rate's docstring")
    mean_rev_level = property(get_mean_rev_level, None,
                              None, "mean_rev_level's docstring")
    vol = property(get_vol, None, None, "vol's docstring")

    def half_life(self):
        return self._half_life

    def equilibrium_vol(self):
        return self._eq_vol

    def simulate(self, num_sims, num_data, dt, init_val=None):
        k = self._mean_rev_rate
        theta = self._mean_rev_level
        s = self._vol

        equilibrium_mean = theta
        equilibrium_stddev = self._eq_vol

        period_variance = s * s * (1.0 - np.exp(-2.0 * k * dt)) / (2.0 * k)

        term = np.exp(-k * dt)
        term2 = theta * (1.0 - term)

        # first sim data will be init_val or draw from equilibrium dist
        # so we only need num_data -1 disturbances
        normal_draws = np.random.randn(num_sims, num_data - 1)

        draws = np.sqrt(period_variance) * normal_draws

        sims = []
        for row in draws:
            curr_row = []
            if init_val is None:
                curr_init = (equilibrium_mean +
                             equilibrium_stddev * np.random.randn())
            else:
                curr_init = init_val

            curr_row.append(curr_init)
            prev = curr_init
            for curr_draw in row:
                curr_val = prev * term + term2 + curr_draw
                curr_row.append(curr_val)
                prev = curr_val
            sims.append(curr_row)
        return np.array(sims)

    def forecast(self, init_val, time_to_horizon):
        exp_term = np.exp(-self._mean_rev_rate * time_to_horizon)

        mean = init_val * exp_term + self._mean_rev_level * (1.0 - exp_term)

        exponent = -2.0 * self._mean_rev_rate * time_to_horizon
        numer = self._vol * self._vol * (1.0 - np.exp(exponent))
        denom = (2.0 * self._mean_rev_rate)

        variance = numer / denom

        return mean, variance

    def to_dataframe(self):
        df = pd.DataFrame(index=["OU"])
        df["mean_rev_rate"] = self._mean_rev_rate
        df["mean_rev_level"] = self._mean_rev_level
        df["vol"] = self._vol
        df["half_life"] = self._half_life
        df["eq_vol"] = self._eq_vol
        return df

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        df = self.to_dataframe().T
        df.columns = [""]
        df.columns.name = "OU"

        return df.to_string()
