"""

"""

# Author: Steven Lillywhite
# License: BSD 3 clause

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from mathsci.arctic_ice import prediction_methods, dataaccess
from mathsci.arctic_ice.arctic_ice_constants import FYZI
from mathsci.segreg.model import Model
from segreg.analysis import plot_models


def plot_fit(indep,
             dep,
             name,
             xlabel,
             ylabel,
             num_end_to_skip=10,
             num_between_to_skip=10,
             models=None,
             show_zero_ice=False,
             xaxis_num_ticks=None,
             yaxis_num_ticks=None,
             num_boot_sims=10000,
             seed=None,
             zero_ice_level=1.0,
             scatter_size=3,
             scatter_color="gray",
             marker="o",
             ax=None):
    """
    NOTES
    -----
    We have removed "monthly-avg" from SII title.

    PARAMETERS
    ----------
    models: list
        list of ``mathsci.segreg.model.Model``
    """

    if models is None:
        models = [Model.OLS, Model.ONE_BKPT, Model.TWO_BKPT]

    models_func = {}
    models_bkpt = {}
    models_first_zero_ice = {}
    models_zero_ice = {}

    for model in models:
        estimator = model.estimator(num_end_to_skip=num_end_to_skip,
                                    num_between_to_skip=num_between_to_skip)

        params = estimator.fit(indep, dep)

        (first_zero,
         zero_ice_time) = prediction_methods.predict_first_zero(indep=indep,
                                                                dep=dep,
                                                                estimator=estimator,
                                                                num_boot_sims=num_boot_sims,
                                                                seed=seed,
                                                                level=zero_ice_level)

        models_func[model] = estimator.model_function
        models_bkpt[model] = None
        models_first_zero_ice[model] = first_zero
        models_zero_ice[model] = zero_ice_time

        if model == Model.ONE_BKPT:
            models_bkpt[Model.ONE_BKPT] = [params[0]]
        if model == Model.TWO_BKPT:
            models_bkpt[Model.TWO_BKPT] = [params[0], params[2]]

    # this will use indep[0] in the called plot method
    domain_begin_arr = None

    if show_zero_ice:
        #domain_end_arr = [models_first_zero_ice[x] for x in models]
        domain_end_arr = [models_zero_ice[x] for x in models]
    else:
        domain_end_arr = None

    title_post = "Model Fit"
    if show_zero_ice:
        title_post += (" and " + FYZI + " Prediction    Extent: " +
                       str(zero_ice_level))
    title = name + "\n" + title_post

    # hack
    title = title.replace(dataaccess._MONTHLY_AVG_STR, "")
    # end hack

    title_bottom = ("\nDashed Vertical Line Indicates Data End" +
                    "    Solid Vertical Line Indicates " + FYZI + " prediction")

    #title += title_bottom

    func_arr = [models_func[x] for x in models]
    extra_pts_arr = [models_bkpt[x] for x in models]

    mark_ends = False

    if show_zero_ice and mark_ends:
        for i in range(len(extra_pts_arr)):

            extra_pts = extra_pts_arr[i]

            if extra_pts is None:
                extra_pts = []
                extra_pts_arr[i] = extra_pts

            domain_end = domain_end_arr[i]
            extra_pts.append(domain_end)

    ax = plot_models(func_arr=func_arr,
                     indep=indep,
                     dep=dep,
                     domain_begin_arr=domain_begin_arr,
                     domain_end_arr=domain_end_arr,
                     title=title,
                     xlabel=xlabel,
                     ylabel=ylabel,
                     extra_pts_arr=extra_pts_arr,
                     mark_extra_pts=True,
                     scatter_size=scatter_size,
                     scatter_color=scatter_color,
                     marker=marker,
                     ax=ax)

    if yaxis_num_ticks is not None:
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=yaxis_num_ticks,
                                                   integer=True))
    else:
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

#    ax.xaxis.set_major_locator(MaxNLocator(min_n_ticks=20, integer=True))
    if xaxis_num_ticks is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=xaxis_num_ticks,
                                               integer=True))
        # plt.xticks(rotation=67.5)

        props = {"rotation": 67.5}
        plt.setp(ax.get_xticklabels(), **props)

    if show_zero_ice:
        ax.axvline(indep[-1], color="gray", linestyle='--')

        for first_zero in models_first_zero_ice.values():
            ax.axvline(first_zero, color="green")  # , linewidth=0.75)

    if show_zero_ice:
        ax.legend(["Breakpoint", "Model Fit", "Data End", FYZI + " Prediction"])
