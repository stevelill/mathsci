"""
Manages production of all plots and tables.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import matplotlib

from mathsci.arctic_ice import plots_tables_predict as ptp
from mathsci.arctic_ice import plots_tables_data as ptd
from mathsci.arctic_ice import plots_tables_estimation as pte
from mathsci.constants import Month, Dataset
from mathsci.segreg.model import Model
from mathsci.arctic_ice.arctic_ice_constants import CONF_INTERVAL_SIGNIFICANCE

from matplotlib import pyplot as plt

figsize = (12, 6)

matplotlib.rcParams['figure.figsize'] = figsize
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['legend.numpoints'] = 1


def create_all_plots_tables(save=False):
    create_data_plots_tables(save=save)
    create_estimation_plots_tables(save=save)
    create_prediction_plots_tables(save=save)


def create_data_plots_tables(save=False):
    ptd.plot_sii_hist_all_months(save=save)
    plt.close("all")

    ptd.plot_sibt_hist_all_months(save=save)
    plt.close("all")

    ptd.plot_sibt_sii_hist_sep(save=save)
    plt.close("all")

    ptd.plot_seasonality(dataset=Dataset.NSIDC, save=save)
    plt.close("all")

    ptd.plot_seasonality(dataset=Dataset.SIBT1850, save=save)
    plt.close("all")

    ptd.plot_sibt_sii_blend_hist_sep_intersect(save=save)
    plt.close("all")


def create_estimation_plots_tables(save=False):
    pte.table_estim_sii_all_months(save=save)
    pte.table_estim_blend_all_months(start_year=1850, save=save)
    pte.table_estim_blend_all_months(start_year=1957, save=save)

    pte.table_bca_conf_int(months=[Month.AUG, Month.SEP, Month.OCT],
                           dataset=Dataset.NSIDC,
                           month_to_priors={Month.SEP: [0, 1, 0]},
                           save=save,
                           verbose=False,
                           significance=CONF_INTERVAL_SIGNIFICANCE)
    # this one is very slow, so we only do 100000 iterations
    pte.table_bca_conf_int(months=[Month.AUG, Month.SEP, Month.OCT],
                           dataset=Dataset.BLEND,
                           save=save,
                           num_iter=100000,
                           significance=CONF_INTERVAL_SIGNIFICANCE)
    pte.table_bca_conf_int(months=[Month.AUG, Month.SEP, Month.OCT],
                           dataset=Dataset.BLEND,
                           save=save,
                           start_year=1957,
                           significance=CONF_INTERVAL_SIGNIFICANCE)

    # info criteria for alternative models
    models = [Model.OLS, Model.ONE_BKPT, Model.QUAD, Model.EXP]
    pte.table_info_criteria(months=[Month.AUG, Month.SEP, Month.OCT],
                            dataset=Dataset.NSIDC,
                            models=models,
                            save=save)

    pte.table_info_criteria(months=Month,
                            dataset=Dataset.BLEND,
                            models=[Model.QUAD, Model.EXP],
                            bic_bkpt=True,
                            save=save)


def create_prediction_plots_tables(save=False):

    ptp.plot_sii_end_years(save=save)
    plt.close("all")

    ptp.plot_and_table_sii_predict_mod_sep(save=save)
    plt.close("all")

    ptp.plot_rolling_end_date(include_sibt=True, save=save)
    plt.close("all")

    ptp.plot_rolling_end_date(include_sibt=False, save=save)
    plt.close("all")

    ptp.plot_fyzi_and_ranges_table_sii_blend(save=save)
    plt.close("all")

    ptp.plot_mean_compare_halves_blend(save=save)
    plt.close("all")

    ptp.rolling_predictions_backwards_blend(save=save)
    plt.close("all")

    ptp.prediction_plot_sep_sii(save=save)
    plt.close("all")

    ptp.prediction_plot_sep_blend(save=save)
    plt.close("all")

    ptp.table_sii_predict(save=save)
    ptp.table_sibt_blend_ranges(save=save)
    ptp.tables_predict_from_precompute(save=save)
    ptp.table_sii_predict_poly(save=save)

    # alternative models
    models = [Model.OLS, Model.ONE_BKPT, Model.TWO_BKPT, Model.QUAD, Model.EXP]
    ptp.table_predictions(months=[Month.AUG, Month.SEP, Month.OCT],
                          dataset=Dataset.NSIDC,
                          models=models,
                          bic_bkpt=False,
                          save=save)

    ptp.table_predictions(months=Month,
                          dataset=Dataset.BLEND,
                          models=[Model.QUAD, Model.EXP],
                          bic_bkpt=True,
                          save=save)
