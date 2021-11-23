"""
Bootstrap prediction intervals.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

from multiprocessing import Pool
import multiprocessing

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from mathsci.arctic_ice import prediction_methods
from mathsci.segreg.models import MultiModelEstimator
from mathsci.statistics import stats_util
from mathsci.statistics.stats_util import BIC_TYPE
from segreg.bootstrap import resampling
from IPython.display import display


def _boot_zero_estim_data(estimator,
                          indep,
                          dep,
                          level=0.0,
                          verbose=False):

    estimator.fit(indep, dep)

    fit_func = estimator.get_func()
    fit_fitted_indep = fit_func(indep)
    fit_resid = estimator.residuals

    fit_zero_time = prediction_methods.zero_ice(estimator=estimator,
                                                indep=indep,
                                                dep=dep,
                                                level=level)

    result_is_good = True

    if prediction_methods.value_is_far(fit_zero_time, indep[-1]):
        if verbose:
            print()
            print("WARNING: cannot compute for infinity zero ice")
            print("orig zero time: ", fit_zero_time)
            print()
        result_is_good = False

    return fit_fitted_indep, fit_resid, result_is_good


def prediction_interval_bic(indep,
                            dep,
                            width=90,
                            num_boot_sims=10000,
                            show_plot=False,
                            verbose=False,
                            seed=None,
                            expanded_percentiles=False,
                            return_dist=False,
                            zero_ice_level=0.0,
                            sub_bic=True,
                            num_end_to_skip=10,
                            num_between_to_skip=10,
                            models=None,
                            model_priors=None,
                            boot_model_priors=None,
                            bic_type=BIC_TYPE.STANDARD):
    """
    Computes prediction intervals associated with prediction the time of
    reaching a certain level.
    
    Covers a number of models, but primarily segmented regression models.
    
    Given a list of models, BIC is used to choose a model.  From this model,
    bootstrap simulations are produced.  For each bootstrap dataset, either
    the same model is used, or else BIC selection is employed.  The chosen model
    is then used to predict the time of reaching a certain level.  The 
    motivating example is a zero threshold for arctic sea ice extent.
    
    More precisely, the predictions are for the first year of zero ice, which
    computes the mean time to cross the threshold for the estimated regression 
    model, including the residual noise.

    We recommend using sub_bic=True.


    Parameters
    ----------
    indep: 
    dep:
    width: int
    num_boot_sims: int
    show_plot: bool
    verbose: bool
    seed: int
    expanded_percentiles: bool
    return_dist: bool
    zero_ice_level: float
    sub_bic: bool
    num_end_to_skip: int
    num_between_to_skip: int
    models: list
    model_priors: list
    boot_model_priors: list
    bic_type: mathsci.statistics.stats_util.BIC_TYPE
    
    Returns
    -------
    interval: array-like
        contains the left and right values of the prediction interval
    
    if ``return_dist`` is True, additionally returns the empirical distribution
    corresponding to the first zero ice time and the time when the model
    extrapolation reaches the threshold level (ie: the conditional mean in the
    regression model).        
    """
    if width >= 100.0:
        raise Exception("width must be less than 100.0")

    if seed is not None:
        np.random.seed(seed)

    if models is not None:
        mme = MultiModelEstimator(models=models,
                                  num_end_to_skip=num_end_to_skip,
                                  num_between_to_skip=num_between_to_skip,
                                  priors=model_priors,
                                  bic_type=bic_type)
        curr_mme = MultiModelEstimator(models=models,
                                       num_end_to_skip=num_end_to_skip,
                                       num_between_to_skip=num_between_to_skip,
                                       priors=boot_model_priors,
                                       bic_type=bic_type)

    else:
        mme = MultiModelEstimator(num_end_to_skip=num_end_to_skip,
                                  num_between_to_skip=num_between_to_skip,
                                  priors=model_priors,
                                  bic_type=bic_type)

        curr_mme = MultiModelEstimator(num_end_to_skip=num_end_to_skip,
                                       num_between_to_skip=num_between_to_skip,
                                       priors=boot_model_priors,
                                       bic_type=bic_type)

    mme.fit(indep, dep)

    # original fit to data (analogue of "truth")
    bic_model = mme.bic_model()
    model_index = mme.models.index(bic_model)
    bic_estimator = mme.estimators[model_index]

    if verbose:
        bma_weights_df = mme.bma_weights_df()
        bma_weights_df.index = ["bma_wts"]
        display(bma_weights_df)
        print()
        print("orig fit bic model: ", bic_model)
        #####print("BIC estimator: ", type(bic_estimator))
        print()

    (fitted_indep,
     fitted_resid,
     result_is_good) = _boot_zero_estim_data(estimator=bic_estimator,
                                             indep=indep,
                                             dep=dep,
                                             level=zero_ice_level,
                                             verbose=verbose)

    ########################################################################
    # MODIFY ORIG FIT RESIDUALS
    # Try to overcome narrowness bias.  See Hesterberg.
    ########################################################################
    fitted_resid = fitted_resid / np.sqrt((1.0 - 2.0 / len(fitted_resid)))

    ########################################################################
    # NOTE THE CONDITIONAL EARLY RETURN !!!
    # These are guaranteed to generate dist way far out -- they are not
    # realistic sims.  They guarantee misses on the left for CI.  To keep,
    # or not to keep?
    ########################################################################
    if not result_is_good:
        print("WARNING: cannot compute prediction interval!")
        return None

    (boot_first_zeros,
     zero_times) = _run_sims(num_boot_sims=num_boot_sims,
                             indep=indep,
                             dep=dep,
                             fitted_indep=fitted_indep,
                             fitted_resid=fitted_resid,
                             sub_bic=sub_bic,
                             curr_mme=curr_mme,
                             zero_ice_level=zero_ice_level,
                             bic_estimator=bic_estimator)

    finite_boot_first_zeros = prediction_methods.infinity_to_max(boot_first_zeros)
    finite_zero_times = prediction_methods.infinity_to_max(zero_times)

    side = (100.0 - width) / 2.0
    percentiles = [side, 100.0 - side]

    # TODO: this should be done ahead of time for mc
    if expanded_percentiles:
        left_quantile = 0.01 * side
        expanded_left_quantile = stats_util.expanded_quantiles(left_quantile,
                                                               num=len(indep))

        percentiles = 100.0 * np.array([expanded_left_quantile,
                                        1.0 - expanded_left_quantile])

    # here, the distribution is discrete years, so any fraction of year needs
    # to round up to following year
    interval = np.percentile(finite_boot_first_zeros,
                             percentiles,
                             interpolation="higher")

    if show_plot:
        plt.figure()
        sns.distplot(finite_boot_first_zeros, bins=200)
        plt.title("boot first zeros")
        plt.grid()

        plt.figure()
        sns.distplot(finite_zero_times, bins=200)
        plt.title("zero times")
        plt.grid()

        plt.show()

    if return_dist:
        return interval, finite_boot_first_zeros, finite_zero_times
    else:
        return interval


def _run_sims(num_boot_sims,
              indep,
              dep,
              fitted_indep,
              fitted_resid,
              sub_bic,
              curr_mme,
              zero_ice_level,
              bic_estimator):

    pool_size = multiprocessing.cpu_count()
    num_per_pool = int(np.ceil(num_boot_sims / pool_size))

    all_params = []
    for dummy in range(pool_size):

        new_seed = np.random.randint(100, 1000000)

        curr_params = [num_per_pool,
                       indep,
                       dep,
                       fitted_indep,
                       fitted_resid,
                       sub_bic,
                       curr_mme,
                       zero_ice_level,
                       bic_estimator,
                       new_seed]
        all_params.append(curr_params)

    pool = Pool(processes=pool_size)
    result = pool.map(_run_sims_impl_wrapper, all_params)
    pool.close()
    pool.join()

    all_boot_first_zeros = []
    all_zero_times = []

    for result_arr in result:
        boot_first_zeros, zero_times = result_arr
        all_boot_first_zeros.extend(boot_first_zeros)
        all_zero_times.extend(zero_times)

    return all_boot_first_zeros, all_zero_times


def _run_sims_impl_wrapper(params):
    return _run_sims_impl(*params)


def _run_sims_impl(num_boot_sims,
                   indep,
                   dep,
                   fitted_indep,
                   fitted_resid,
                   sub_bic,
                   curr_mme,
                   zero_ice_level,
                   bic_estimator,
                   seed=None):

    if seed is not None:
        np.random.seed(seed)

    boot_first_zeros = []
    zero_times = []

    for dummy in range(num_boot_sims):

        indep_resample, dep_resample = resampling.boot_resample(indep,
                                                                dep,
                                                                fitted_indep,
                                                                fitted_resid)
        # TODO: for some reason, we need to set the seed here
        # if we just ask for 1 boot sim here, it will return first zero for
        # the given resampled data

        new_seed = np.random.randint(100, 1000000)

        if sub_bic:
            # here we select bic model for each boot sample and use it to
            # predict zero ice
            curr_mme.fit(indep_resample, dep_resample)
            curr_bic_model = curr_mme.bic_model()
            curr_model_index = curr_mme.models.index(curr_bic_model)
            curr_estimator = curr_mme.estimators[curr_model_index]

            (boot_first_zero,
             zero_time) = prediction_methods.predict_first_zero(indep_resample,
                                                                dep_resample,
                                                                curr_estimator,
                                                                num_boot_sims=1,
                                                                seed=new_seed,
                                                                level=zero_ice_level)
        else:
            # here we use the bic model for the original dataset to
            # predict zero ice
            (boot_first_zero,
             zero_time) = prediction_methods.predict_first_zero(indep_resample,
                                                                dep_resample,
                                                                bic_estimator,
                                                                num_boot_sims=1,
                                                                seed=new_seed,
                                                                level=zero_ice_level)

        boot_first_zeros.append(boot_first_zero)
        zero_times.append(zero_time)

    return boot_first_zeros, zero_times
