"""
Rolling predictions
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import os

import pandas as pd

from mathsci.arctic_ice import final_predictions, data_util
from mathsci.arctic_ice.arctic_ice_constants import (
    NUM_END_TO_SKIP,
    NUM_BETWEEN_TO_SKIP,
    ZERO_ICE_LEVEL,
    SEED,
    NUM_BOOT_SIMS,
    TYPE_OF_BIC,
    ROLLING_BLEND_DIR
)
from mathsci.constants import Dataset, SegregMeta
from mathsci.utilities import general_utilities as gu
from mathsci.segreg.model import Model


def _run_sims(month,
              dataset_name,
              ice_extent_df,
              indep,
              dep,
              outdir_root,
              model_priors=None,
              model_name="bic",
              skip_interval=False,
              forwards=False):

    outdir = os.path.join(outdir_root, month.name)

    print(outdir)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    predict_filename = os.path.join(outdir, model_name + "_predict.csv")
    meta_filename = os.path.join(outdir, "meta.csv")

    end_year = ice_extent_df.index[-1]

    rows = []
    index = []
    cols = None

#    preserve = 2 * (num_end_to_skip + num_between_to_skip + 2) - 1

    preserve = 40

    num = len(indep) - preserve

    index = []

    for i in range(num):

        if forwards:
            backwards_ind = preserve + i + 1
            indep_to_use = indep[0:backwards_ind]
            dep_to_use = dep[0:backwards_ind]
            index.append(ice_extent_df.index[backwards_ind])
        else:
            indep_to_use = indep[i:]
            dep_to_use = dep[i:]
            index.append(ice_extent_df.index[i])

        if i % 10 == 0:
            print("iter: ", i)

        curr_year = ice_extent_df.index[i].year

        # NOTE: BAD HACK HERE
        if model_name == "bic" and (curr_year == 1850 or curr_year == 1957):
            skip_interval_to_use = False
        else:
            skip_interval_to_use = skip_interval

        (predict_df,
         est_df,
         zero_ice_df) = final_predictions.predict_omnibus(
            indep=indep_to_use,
            dep=dep_to_use,
            num_end_to_skip=NUM_END_TO_SKIP,
            num_between_to_skip=NUM_BETWEEN_TO_SKIP,
            zero_ice_level=ZERO_ICE_LEVEL,
            seed=SEED,
            num_boot_sims=NUM_BOOT_SIMS,
            model_priors=model_priors,
            skip_interval=skip_interval_to_use,
            bic_type=TYPE_OF_BIC)

        rows.append(predict_df.values[0])

        if cols is None:
            cols = predict_df.columns

    df = pd.DataFrame(rows, columns=cols, index=index)
    df.index.name = "Date"
    df.bic_model = [x.name for x in df.bic_model]

    df.to_csv(predict_filename)

    if model_priors is None:
        meta = SegregMeta(num_end_to_skip=NUM_END_TO_SKIP,
                          num_between_to_skip=NUM_BETWEEN_TO_SKIP,
                          num_iter=NUM_BOOT_SIMS,
                          month=month.name,
                          dataset=dataset_name,
                          start_year=None,
                          end_year=end_year,
                          seed=SEED,
                          forwards=forwards)

        meta_df = gu.namedtuple_asdf(meta)

        meta_df.to_csv(meta_filename, index=None, na_rep="NA")


def rolling_predict():
    skip_interval = True

    # only need to run sibt1850 once -- there are no data updates
    ##dataset = Dataset.SIBT1850

    dataset = Dataset.BLEND
    outdir_root = ROLLING_BLEND_DIR

    if not os.path.exists(outdir_root):
        os.makedirs(outdir_root)

    data_map = data_util.fetch_all_months(dataset=dataset,
                                          months=None,
                                          start_year=None,
                                          end_year=None)

    for month, fetched_data in data_map.items():

        #name = fetched_data.name
        ice_extent_df = fetched_data.data_df
        indep = fetched_data.indep
        dep = fetched_data.dep

        model_priors_list = [None,
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]]

        model_names = ["bic",
                       Model.OLS.display,
                       Model.ONE_BKPT.display,
                       Model.TWO_BKPT.display]

        for model_priors, model_name in zip(model_priors_list, model_names):

            _run_sims(month=month,
                      dataset_name=dataset.name,
                      ice_extent_df=ice_extent_df,
                      indep=indep,
                      dep=dep,
                      outdir_root=outdir_root,
                      model_priors=model_priors,
                      model_name=model_name,
                      skip_interval=skip_interval,
                      forwards=False)
