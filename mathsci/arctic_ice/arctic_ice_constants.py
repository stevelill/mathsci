"""
Constants specific to arctic ice routines.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause 

import os

from mathsci.statistics.stats_util import BIC_TYPE

NUM_END_TO_SKIP = 10
NUM_BETWEEN_TO_SKIP = 10

ZERO_ICE_LEVEL = 1.0
SEED = 2357238
# 100000 does not seem to change results
NUM_BOOT_SIMS = 10000
TYPE_OF_BIC = BIC_TYPE.HYBRID

NA_REP = "--"

CONF_INTERVAL_SIGNIFICANCE = 0.05

fyzi = "first year of zero sea ice"
FYZI = "First Year of Zero Sea Ice"

# USER NEEDS TO FILL THESE IN
# TODO: move these to a config file
ROLLING_BLEND_DIR = ""
ROLLING_SIBT_DIR = ""
DATA_ROOT_DIR = ""
PLOTS_TABLES_ROOTDIR = ""