"""
Runs monte carlo simulations that test BIC model selection for segmented
regression models. 
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import argparse
import os
import json

from matplotlib import pyplot as plt
import numpy as np

from mathsci.segreg.bic_sim_test_suite import BICTestSuite
from mathsci.arctic_ice.arctic_ice_constants import (
    NUM_END_TO_SKIP,
    NUM_BETWEEN_TO_SKIP,
    SEED
)


NUM_SIMS = 10000


def run_sims(bic_test_suite,
             truth_model_str,
             index,
             outdir_root):
    if truth_model_str == "ols_truth":
        (frac_chosen_df,
         frac_significant_df,
         level_pred_rmse_df,
         meta) = bic_test_suite.ols_truth_bic_monte_carlo(index=index)
    elif truth_model_str == "one_bkpt_truth":
        (frac_chosen_df,
         frac_significant_df,
         level_pred_rmse_df,
         meta) = bic_test_suite.one_bkpt_truth_bic_monte_carlo(index=index)
    elif truth_model_str == "two_bkpt_truth":
        (frac_chosen_df,
         frac_significant_df,
         level_pred_rmse_df,
         meta) = bic_test_suite.two_bkpt_truth_bic_monte_carlo(index=index)

    # SAVE RESULTS
    outdir = os.path.join(outdir_root, truth_model_str, str(index))

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    suffix = ".csv"

    frac_chosen_df.to_csv(os.path.join(outdir, "frac_chosen" + suffix))
    frac_significant_df.to_csv(os.path.join(outdir, "frac_significant" + suffix))
    level_pred_rmse_df.to_csv(os.path.join(outdir, "level_pred_rmse" + suffix))

    plt.savefig(os.path.join(outdir, "example"))
    plt.close()

    json_out = os.path.join(outdir, "meta.json")
    with open(json_out, 'w') as fp:
        json.dump(meta, fp, indent=4)


def main():
    desc = "Run monte carlo BIC test for the Arctic Sea Ice Project"
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)

    parser.add_argument("outdir", help="output directory")

    args = parser.parse_args()

    outdir = args.outdir

    print()
    print("running segreg BIC monte carlo test")
    print("outdir: ", outdir)
    print()

    bic_test_suite = BICTestSuite(num_sims=NUM_SIMS,
                                  seed=SEED,
                                  num_end_to_skip=NUM_END_TO_SKIP,
                                  num_between_to_skip=NUM_BETWEEN_TO_SKIP)

    truth_model_strs = ["ols_truth",
                        "one_bkpt_truth",
                        "two_bkpt_truth"]

    index_list = [[1, 2],
                  np.arange(1, 16),
                  np.arange(1, 10)]

    for truth_model_str, indices in zip(truth_model_strs, index_list):

        print("-" * 75)
        print()
        print(truth_model_str)
        print()

        for index in indices:

            print("-" * 15)
            print("index: ", index)
            run_sims(bic_test_suite=bic_test_suite,
                     truth_model_str=truth_model_str,
                     index=index,
                     outdir_root=outdir)


if __name__ == '__main__':
    main()
