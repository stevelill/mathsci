"""
Constants for arctic sea ice.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

from collections import namedtuple
from enum import Enum


class Month(Enum):
    JAN = 1
    FEB = 2
    MAR = 3
    APR = 4
    MAY = 5
    JUN = 6
    JUL = 7
    AUG = 8
    SEP = 9
    OCT = 10
    NOV = 11
    DEC = 12


class Dataset(Enum):
    NSIDC = 1
    SIBT1850 = 2
    BLEND = 3

    @property
    def name(self):
        if self == Dataset.NSIDC:
            return "SII"
        else:
            return self._name_


class Stat(Enum):
    AIC = 1
    AICC = 2
    BIC = 3
    LRT = 4
    RSS = 5
    RMSE = 6
    R_SQUARED = 7


SegregMeta = namedtuple("SegregMeta", ["num_end_to_skip",
                                       "num_between_to_skip",
                                       "num_iter",
                                       "month",
                                       "dataset",
                                       "start_year",
                                       "end_year",
                                       "seed",
                                       "forwards"])
SegregMeta.__new__.__defaults__ = (None,) * len(SegregMeta._fields)
