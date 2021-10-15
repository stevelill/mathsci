"""

"""

# Author: Steven Lillywhite
# License: BSD 3 clause

from collections import defaultdict

import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt

from mathsci.segreg import bic_sim_test_routines as tester, fake_data
from mathsci.statistics.stats_util import BIC_TYPE
from mathsci.segreg.model import Model
from mathsci.segreg import segreg_func_generators as func_gen


_NUM_DATA = [45, 50, 60, 70, 80, 100, 150]

_BIC_TYPES = [BIC_TYPE.STANDARD,
              BIC_TYPE.HOS,
              BIC_TYPE.HOS_2,
              BIC_TYPE.LWZ]

_MODELS = [Model.OLS, Model.ONE_BKPT, Model.TWO_BKPT]
_NUM_SIMS = 1000
_RESID_STDDEV = 0.5
_PREDICT_HORIZON = 30

# this is important for ``segreg_func_generators``
_INDEP_START = 1.0

_SKIP_FIT_AT_BOUNDARY = False
_ONLY_BOUNDARY = False


def _set_val_nested_dict(mydict, key1, key2, key3, val):
    """
    Sets values on a three-level dict
    """
    val1 = mydict.get(key1)
    if val1 is None:
        mydict[key1] = {}

    val2 = mydict[key1].get(key2)
    if val2 is None:
        (mydict[key1])[key2] = {}

    (mydict[key1][key2])[key3] = val


class BICTestSuite():

    """
    classdocs
    """

    def __init__(self,
                 num_data_arr=None,
                 bic_types=None,
                 models=None,
                 num_sims=_NUM_SIMS,
                 num_end_to_skip=10,
                 num_between_to_skip=10,
                 resid_stddev=_RESID_STDDEV,
                 seed=None,
                 predict_horizon=_PREDICT_HORIZON,
                 skip_fit_at_boundary=_SKIP_FIT_AT_BOUNDARY,
                 only_boundary=_ONLY_BOUNDARY):

        if num_data_arr is not None:
            self._num_data_arr = num_data_arr
        else:
            self._num_data_arr = _NUM_DATA

        if bic_types is not None:
            self._bic_types = bic_types
        else:
            self._bic_types = _BIC_TYPES

        if models is not None:
            self._models = models
        else:
            self._models = _MODELS

        self._num_sims = num_sims
        self._num_end_to_skip = num_end_to_skip
        self._num_between_to_skip = num_between_to_skip
        self._resid_stddev = resid_stddev
        self._seed = seed
        self._predict_horizon = predict_horizon
        self._skip_fit_at_boundary = skip_fit_at_boundary
        self._only_boundary = only_boundary

        # helpers
        self._reset()
        #self._have_plotted = False
        #self._meta = self._create_meta()
#        self._num_sims_processed = defaultdict(list)

    def _reset(self):
        self._frac_chosen = {}
        self._frac_significant = {}
        self._level_pred_rmse = {}
        self._num_sims_processed = defaultdict(list)

        self._have_plotted = False
        self._meta = self._create_meta()

    def _create_meta(self):
        meta = {}
        meta["num_data"] = self._num_data_arr
        meta["bic_types"] = [x.name for x in self._bic_types]
        meta["models"] = [y.name for y in self._models]
        meta["num_sims"] = self._num_sims
        meta["num_end_to_skip"] = self._num_end_to_skip
        meta["num_between_to_skip"] = self._num_between_to_skip
        meta["resid_stddev"] = self._resid_stddev
        meta["seed"] = self._seed
        meta["predict_horizon"] = self._predict_horizon
        meta["indep_start"] = _INDEP_START
        return meta

    def _dict_to_dataframe(self, mydict):
        outer_keys = list(mydict.keys())
        ####inner_keys = [x.name for x in list(mydict[outer_keys[0]].keys())]
        inner_keys = list(mydict[outer_keys[0]].keys())

        # lower case them ?
        #outer_keys = [x.lower() for x in outer_keys]
        inner_keys = [y.lower() for y in inner_keys]

        index = pd.MultiIndex.from_product([outer_keys, inner_keys],
                                           names=("Resid Type", "BIC Type"))

        data = defaultdict(list)
        for resid_type, dict_val in mydict.items():
            for bic_type, dict_val2 in dict_val.items():
                for num_data, val in dict_val2.items():
                    data[num_data].append(val)

        df = pd.DataFrame(data, index=index)
        return df

    def ols_truth_bic_monte_carlo(self, index):

        model = Model.OLS

        (model_func_gen,
         description) = func_gen.get_ols_func_generator(index=index)

        print()
        print(description)
        print()

        (frac_chosen_df,
         frac_significant_df,
         level_pred_rmse_df) = self._run_tests(model, model_func_gen)

        self._meta["truth_model_desc"] = description

        return (frac_chosen_df,
                frac_significant_df,
                level_pred_rmse_df,
                self._meta)

    def one_bkpt_truth_bic_monte_carlo(self, index):

        model = Model.ONE_BKPT

        (model_func_gen,
         description) = func_gen.get_one_bkpt_func_generator(index=index)

        print()
        print(description)
        print()

        (frac_chosen_df,
         frac_significant_df,
         level_pred_rmse_df) = self._run_tests(model, model_func_gen)

        self._meta["truth_model_desc"] = description

        return (frac_chosen_df,
                frac_significant_df,
                level_pred_rmse_df,
                self._meta)

    def two_bkpt_truth_bic_monte_carlo(self, index):

        model = Model.TWO_BKPT

        (model_func_gen,
         description) = func_gen.get_two_bkpt_func_generator(index=index)

        print()
        print(description)
        print()

        (frac_chosen_df,
         frac_significant_df,
         level_pred_rmse_df) = self._run_tests(model, model_func_gen)

        self._meta["truth_model_desc"] = description

        return (frac_chosen_df,
                frac_significant_df,
                level_pred_rmse_df,
                self._meta)

    def _run_tests(self, model, model_func_generator):
        """
        Main internal method.
        """

        # reset data structures for new test
        self._reset()

        for num_data in self._num_data_arr:

            model_func = model_func_generator(num_data)

            self._run_tests_num_data(num_data=num_data,
                                     model=model,
                                     model_func=model_func)

        # add post-run data to meta
        if self._skip_fit_at_boundary or self._only_boundary:
            self._meta["num_sims_processed"] = dict(self._num_sims_processed)

        return (self._dict_to_dataframe(self._frac_chosen),
                self._dict_to_dataframe(self._frac_significant),
                self._dict_to_dataframe(self._level_pred_rmse))

    def _run_tests_num_data(self, num_data, model, model_func):

        (normal_data,
         student_data,
         ou_data1,
         ou_data2) = fake_data.fake_data_suite(model_func=model_func,
                                               num_data=num_data,
                                               num_sims=self._num_sims,
                                               indep_start=_INDEP_START,
                                               resid_stddev=self._resid_stddev,
                                               seed=self._seed)

        for data in [normal_data, student_data, ou_data1, ou_data2]:
            self._run_tests_fake_data(fake_data=data,
                                      model=model,
                                      model_func=model_func)

    def _run_tests_fake_data(self, fake_data, model, model_func):

        indep = fake_data.indep
        deps = fake_data.deps
        fake_data_name = fake_data.name

        ##############################################
        # begin erase
        if not self._have_plotted:
            plt.plot(indep, model_func(indep))
            plt.scatter(indep, deps[0], s=0.5)
            plt.grid()
            # plt.show()
            plt.title("Example Truth Function and Simulated Data")
            self._have_plotted = True
        # end erase
        ##############################################

        truth_horizon_indep_val = indep[-1] + self._predict_horizon
        truth_prediction = model_func(truth_horizon_indep_val)

        (bic_type_to_wts_df,
         model_pred_df,
         bic_pred_df,
         model_level_pred_df,
         bic_level_pred_df) = tester.bic_sim_summary(self._bic_types,
                                                     indep,
                                                     deps,
                                                     num_end_to_skip=self._num_end_to_skip,
                                                     num_between_to_skip=self._num_between_to_skip,
                                                     truth_model=model,
                                                     skip_fit_at_boundary=self._skip_fit_at_boundary,
                                                     predict_horizon=self._predict_horizon,
                                                     predict_level=truth_prediction,
                                                     only_boundary=self._only_boundary)

        num_data = len(indep)

        # keep track of num processed in case of skip_fit_at_boundary or
        # only-boundary options
        num_processed = len(model_pred_df)
        assert(num_processed == len(bic_pred_df))
        assert(num_processed == len(model_level_pred_df))
        assert(num_processed == len(bic_level_pred_df))
        self._num_sims_processed[fake_data_name].append(num_processed)
        # end erase

        self._process_results(model=model,
                              fake_data_name=fake_data_name,
                              num_data=num_data,
                              bic_type_to_wts_df=bic_type_to_wts_df,
                              model_pred_df=model_pred_df,
                              bic_pred_df=bic_pred_df,
                              model_level_pred_df=model_level_pred_df,
                              bic_level_pred_df=bic_level_pred_df,
                              truth_horizon_indep_val=truth_horizon_indep_val)

    def _process_results(self,
                         model,
                         fake_data_name,
                         num_data,
                         bic_type_to_wts_df,
                         model_pred_df,
                         bic_pred_df,
                         model_level_pred_df,
                         bic_level_pred_df,
                         truth_horizon_indep_val):

        for bic_type in self._bic_types:
            wts_df = bic_type_to_wts_df[bic_type]
#             print("-------------------")
#             print("num data: ", len(indep))
#             print(fake_data.name)
#             print(bic_type)

            bic_type_name = bic_type.name

            frac_chosen = wts_df.loc["frac_chosen", model.display]
            _set_val_nested_dict(self._frac_chosen,
                                 key1=fake_data_name,
                                 key2=bic_type_name,
                                 key3=num_data,
                                 val=frac_chosen)

            frac_significant = wts_df.loc["frac_significant", model.display]
            _set_val_nested_dict(self._frac_significant,
                                 key1=fake_data_name,
                                 key2=bic_type_name,
                                 key3=num_data,
                                 val=frac_significant)

        #####cutoff = truth_horizon_indep_val + 500.0
        # cutoff at 99 percentile
#         cutoff = model_level_pred_df.quantile([0.99]).to_numpy().ravel().max()
#         print(fake_data_name, " ; cutoff: ", cutoff)
#         trim_model_level_pred_df = model_level_pred_df[model_level_pred_df < cutoff]
#         trim_bic_level_pred_df = bic_level_pred_df[bic_level_pred_df < cutoff]

        remove_outliers = True

        df1 = tester.compute_rmse(bic_level_pred_df,
                                  truth_horizon_indep_val,
                                  remove_outliers=remove_outliers)

        df2 = tester.compute_rmse(model_level_pred_df,
                                  truth_horizon_indep_val,
                                  remove_outliers=remove_outliers)

        for i, colname in enumerate(df1.columns):
            _set_val_nested_dict(self._level_pred_rmse,
                                 key1=fake_data_name,
                                 key2=colname,
                                 key3=num_data,
                                 val=df1.iloc[0, i])

        for i, colname in enumerate(df2.columns):
            _set_val_nested_dict(self._level_pred_rmse,
                                 key1=fake_data_name,
                                 key2=colname,
                                 key3=num_data,
                                 val=df2.iloc[0, i])
