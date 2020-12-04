__doc__ = """Data decomposition analyses performed on VTE data.

PCA originals found in :mod:`mtml.modeling._vte_models.py`.

For example, principal components analysis (eigendecomposition/SVD).
"""

# pylint: disable=import-error
from functools import partial
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler

# pylint: disable=relative-beyond-top-level
from .. import BASE_RESULTS_DIR
from ... import VTE_CONT_INPUT_COLS, VTE_OUTPUT_COLS
from ...data.vte import vte_slp_factory
from .data_transforms import replace_hdl_tot_chol_with_ratio
from ...feature_selection.univariate import roc_auc_score_func
from ...utils.persist import persist_csv, persist_json, persist_pickle


@persist_json(
    target = BASE_RESULTS_DIR + "/vte_whitened_pca_params.json", enabled = True,
    out_transform = lambda x: (x["pcas"][0].get_params(), 
                               x["pcas"][1].get_params())
)
@persist_pickle(
    target = BASE_RESULTS_DIR + "/vte_whitened_pcas.pickle", enabled = True,
    out_transform = lambda x: x["pcas"]
)
def whitened_pca(*, report = False, plotting_dir = BASE_RESULTS_DIR + "",
                 random_seed = None, figsize = (12, 4), 
                 fig_fname = "vte_whitened_pcas_pct_trace.png",
                 dpi = 150, tight_layout = True, plot_kwargs = None):
    """Analysis method that performs whitened PCA on the VTE data set.

    PCA is performed twice, first on all the columns specified by
    ``VTE_CONT_INPUT_COLS`` and then on the seven columns chosen by a
    :class:`sklearn.feature_selection.SelectKBest` instance using our homegrown
    :func:`mtml.feature_selection.univariate.roc_auc_score_func` score function.

    A report may optionally be printed which shows the explained variance ratios
    and if ``plotting_dir`` is not ``None``, then plots of the explained
    variance ratios will be written to the directory path provided to
    ``plotting_dir``.

    Parameter descriptions in progress.
    
    :rtype: dict
    """
    # if plot_kwargs are None, set to empty dict
    if plot_kwargs is None:
        plot_kwargs = {}
    # get data set of continuous features from vte_slp_factory
    X_train, X_test, y_train, y_test = vte_slp_factory(
        data_transform = replace_hdl_tot_chol_with_ratio,
        inputs = VTE_CONT_INPUT_COLS, targets = VTE_OUTPUT_COLS, dropna = True,
        random_state = random_seed
    )
    # use univariate ROC AUC scores to select top 7 features and get a reduced
    # matrix. note that we have to wrap in lambda in order to pass seed.
    skbest = SelectKBest(
        partial(roc_auc_score_func, random_state = random_seed), k = 7
    )
    # first fit so we can get feature mask using get_support
    skbest.fit(X_train, y_train)
    col_mask = skbest.get_support()
    # get reduced X_train and X_test by selecting only the 7 selected columns
    X_train_red, X_test_red = X_train[:, col_mask], X_test[:, col_mask]
    # names of the 7 selected columns; order by descending AUC score
    kbest_cols = np.array(VTE_CONT_INPUT_COLS)[col_mask][
        np.flip(np.argsort(skbest.scores_[col_mask]))
    ]
    # fit StandardScalers to training data and use them to normalize the data
    # before feeding training data to PCA
    scaler_full, scaler_red = StandardScaler(), StandardScaler()
    scaler_full.fit(X_train), scaler_red.fit(X_train_red)
    X_train, X_test = (
        scaler_full.transform(X_train), scaler_full.transform(X_test)
    )
    X_train_red, X_test_red = (
        scaler_red.transform(X_train_red), scaler_red.transform(X_test_red)
    )
    # PCA for full continuous columns and PCA for 7 highest AUC columns. note
    # that we apply whitening transform since we'll need that for the models
    # that are scaling-sensitive (like regularized models)
    pca_full = PCA(whiten = True, random_state = random_seed)
    pca_red = PCA(whiten = True, random_state = random_seed)
    # fit on the (standardized) training data
    pca_full.fit(X_train)
    pca_red.fit(X_train_red)
    # n_components needed to explain 95% variance for full and reduced data.
    n_full_95 = n_red_95 = 0
    # compute for each PCA how many components are needed to explain 95% var
    for i in range(pca_full.n_components_):
        full_var_pct = pca_full.explained_variance_ratio_[:(i + 1)].sum()
        if full_var_pct >= 0.95:
            n_full_95 = i + 1
            break
    for i in range(pca_red.n_components_):
        red_var_pct = pca_red.explained_variance_ratio_[:(i + 1)].sum()
        if red_var_pct >= 0.95:
            n_red_95 = i + 1
            break
    # print explained variance ratios (eigenvalues standardized by trace) if
    # the option to report is True
    if report is True:
        print(f"---- VTE PCA on VTE_CONT_INPUT_COLS ", end = "")
        print("-" * 44, end = "\n\n")
        print(
            f"explained variance ratios:\n{pca_full.explained_variance_ratio_}"
            f"\n\nn_components needed to explain 95% variance: {n_full_95}\n"
        )
        print(f"---- VTE PCA on top 7 columns ", end = "")
        print("-" * 50, end = "\n\n")
        print(
            f"selected columns (by univariate AUC):\n{kbest_cols}\n\n"
            f"explained variance ratios:\n{pca_red.explained_variance_ratio_}"
            f"\n\nn_components needed to explain 95% variance: {n_red_95}"
        )
    # if plotting dir is None, don't plot
    if plotting_dir is None:
        pass
    # else make plots for full and reduced PCA and write plots to plotting_dir
    else:
        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = figsize)
        # plot
        for ax, pca, n_for_95, ax_title in zip(
            axs, (pca_full, pca_red), (n_full_95, n_red_95),
            ("VTE_CONT_INPUT_COLS", "top 7 columns by AUC")
        ):
            # plot the normalized eigenvalues; use square marker
            ax.plot(
                1 + np.arange(pca.n_components_), pca.explained_variance_ratio_,
                marker = "s", **plot_kwargs
            )
            # show all the eigenvalues (there aren't many so this is fine)
            ax.set_xticks(1 + np.arange(pca.n_components_))
            # plot vertical line indicating last eigenvalue needed to explain
            # 95% of variance (add 0.5 for aesthetic)
            ax.axvline(
                x = n_for_95, color = "red",
                label = (r"$ k $ s.t. $ \frac{1}{\mathrm{tr}(\mathbf{C})}"
                         r"\sum_{i = 1}^k\lambda_i \geq 0.95 $")
            )
            # show legend (vline label)
            ax.legend()
            # set axis labels and axis title
            ax.set_xlabel("eigenvalue number")
            ax.set_ylabel(r"percent of trace")
            ax.set_title(ax_title)
        # set overall title for the figure
        fig.suptitle(
            r"Percent trace of $ \mathbf{C} = \frac{1}{n}\mathbf{X}^\top"
            r"\mathbf{X} $ per eigenvalue $ \lambda_i $", size = "x-large"
        )
        # if tight_layout, call tight_layout
        if tight_layout is True:
            fig.tight_layout()
        # save figure at dpi at plotting_dir/fig_fname
        fig.savefig(plotting_dir + "/" + fig_fname, dpi = dpi)
    # return PCA objects and (standardized) data so that they can be
    # appropriately persisted/passed to another analysis/model fitting method
    return {
        "pcas": (pca_full, pca_red), 
        "data_train": (X_train, X_train_red, y_train),
        "data_test": (X_test, X_test_red, y_test)
    }


def plot_pca_components_2d(*, plotting_dir = BASE_RESULTS_DIR + "",
                           random_seed = None, figsize = (8, 6), 
                           fig_fname = "vte_whitened_pca_2d_eig.png", 
                           dpi = 150, cmap = "viridis",
                           tight_layout = True, plot_kwargs = None):
    """Calls :func:`whitened_pca` and plots scatter using top 2 eignvectors.

    Only uses the data from the (standardized) full training data matrix (uses
    all the columns from ``VTE_CONT_INPUT_COLS``) when retrieving eigenspace
    coordinates. Note that the data has already been centered and scaled to unit
    variance during the PCA fitting process.

    The ``random_seed`` parameter is also used by :func:`whitened_pca` but all
    the plotting parameters only apply to the plot drawn by this function.

    Parameter descriptions in progress.

    :rtype: tuple
    """
    # sanity checks
    if plotting_dir is None:
        raise ValueError("plotting_dir is None")
    if not hasattr(matplotlib.cm, cmap):
        raise AttributeError(f"unknown cmap value {cmap}")
    # if None, set to empty dict
    if plot_kwargs is None:
        plot_kwargs = {}
    # get results from whitened_pca (doesn't generate plots)
    pca_res = whitened_pca(random_seed = random_seed, plotting_dir = None)
    # get full training data and PCA object fitted on them. note that this data
    # has already been standardized using StandardScaler.
    X_train, _, y_train = pca_res["data_train"]
    pca = pca_res["pcas"][0]
    # transform X_train and retrieve eigenbasis data coordinates
    Z_train = pca.transform(X_train)
    # new figure
    fig, ax = plt.subplots(figsize = figsize)
    # color map
    cmap = getattr(matplotlib.cm, cmap)
    # plot scatter of first two components, using label as color and cmap to
    # get the correct color map from matplotlib.cm
    for data_label, plot_label in zip((0, 1), ("negative", "positive")):
        ax.scatter(
            Z_train[:, 0][y_train == data_label], 
            Z_train[:, 1][y_train == data_label],
            c = [cmap(abs(data_label - 0.2))], label = plot_label
        )
    # label axes, add legend, add title
    ax.set_xlabel(r"$ z_1 $")
    ax.set_ylabel(r"$ z_2 $")
    ax.legend()
    fig.suptitle(
        r"VTE train data in eigenbasis coordinates $ [ \ z_1 \ \ z_2 "
        r"\ \ \ldots \ ] $ ", size = "x-large"
    )
    # if tight_layout is True, call tight_layout
    if tight_layout:
        fig.tight_layout()
    # save to disk
    fig.savefig(plotting_dir + "/" + fig_fname, dpi = dpi)
    # return figure and axis
    return fig, ax


if __name__ == "__main__":
    # _ = whitened_pca(report = True, random_seed = 7)
    # _ = plot_pca_components_2d(random_seed = 7)
    pass