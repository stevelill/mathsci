"""
Manages production of all plots and tables.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import argparse

from mathsci.constants import Month
from mathsci.arctic_ice import plots_tables_manager as ptm


def main():
    desc = "Create all plots and tables for the Arctic Sea Ice Project"
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)

    # TODO: add outdir arg, and eliminate global variable for this
#    parser.add_argument("outdir", help="output directory")

    args = parser.parse_args()

    print()
    print("Creating plots and tables for Arctic Sea Ice Paper")
    print()

    months = [Month.AUG, Month.SEP, Month.OCT]

    ptm.create_all_plots_tables(save=True, months=months)


if __name__ == '__main__':
    main()
