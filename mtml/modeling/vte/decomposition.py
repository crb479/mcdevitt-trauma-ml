__doc__ = """Data decomposition analyses performed on VTE data.

PCA originals found in :mod:`mtml.modeling._vte_models.py`.

For example, principal components analysis (eigendecomposition/SVD).
"""

# pylint: disable=import-error
from functools import partial
import matplotlib.cm
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA, PCA
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# pylint: disable=relative-beyond-top-level
from .. import BASE_RESULTS_DIR
from ... import VTE_CONT_INPUT_COLS, VTE_OUTPUT_COLS
from ...data.vte import vte_slp_factory
from .data_transforms import replace_hdl_tot_chol_with_ratio
from ...feature_selection.univariate import roc_auc_score_func
from ...utils.extmath.eigenvalues import n_eigs_pct_trace
from ...utils.models import json_safe_get_params
from ...utils.persist import persist_csv, persist_json, persist_pickle
from ...utils.plotting import normalized_scree_plot


@persist_json(
    target = BASE_RESULTS_DIR + "/vte_whitened_pca_params.json", enabled = True,
    out_transform = lambda x: (x["pcas"][0].get_params(), 
                               x["pcas"][1].get_params())
)
@persist_pickle(
    target = BASE_RESULTS_DIR + "/vte_whitened_pcas.pickle", enabled = True,
    out_transform = lambda x: x["pcas"]
)
def whitened_pca(*, report = False, plotting_dir = BASE_RESULTS_DIR,
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
    n_full_95 = n_eigs_pct_trace(pca_full.explained_variance_, presort = False)
    n_red_95 = n_eigs_pct_trace(pca_red.explained_variance_, presort = False)
    # print explained variance ratios (eigenvalues standardized by trace) if
    # the option to report is True
    if report is True:
        # for each of the PCA objects and n_components to explain 95% variance
        for pca, n_comp_95, title in zip(
            (pca_full, pca_red), (n_full_95, n_red_95),
            ("---- VTE PCA on VTE_CONT_INPUT_COLS " + "-" * 44,
             "---- VTE PCA on top 7 columns " + "-" * 50)
        ):
            print(title, end = "\n\n")
            # print the selected columns if pca is pca_red
            if pca == pca_red:
                print(
                    f"selected columns (by univariate AUC):\n{kbest_cols}\n"
                )
            print(
                f"explained variance ratios:\n{pca.explained_variance_ratio_}"
                f"\n\nn_components needed to explain 95% variance: {n_comp_95}",
                end = "\n\n"
            )
    # if plotting dir is None, don't plot
    if plotting_dir is None:
        pass
    # else make plots for full and reduced PCA and write plots to plotting_dir
    else:
        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = figsize)
        # plot
        for ax, pca, ax_title in zip(
            axs, (pca_full, pca_red),
            ("VTE_CONT_INPUT_COLS", "top 7 columns by AUC")
        ):
            # make scree plot. don't need to presort eigenvalues (already in
            # descending order), don't need to normalize (already normalized),
            # add vertical line at 95% mark, add x ticks
            normalized_scree_plot(
                ax, pca.explained_variance_ratio_, normalize = False,
                presort = False, pct_trace = 0.95, plot_title = ax_title,
                plot_xticks = 1 + np.arange(pca.n_components_)
            )
        # set overall title for the figure
        fig.suptitle(
            r"Percent trace of $ \mathbf{C} \triangleq \frac{1}{n}\mathbf{X}^\top"
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


def plot_pca_components_2d(*, plotting_dir = BASE_RESULTS_DIR,
                           random_seed = None, figsize = (8, 6), 
                           fig_fname = "vte_whitened_pca_2d_eig.png", 
                           dpi = 150, cmap = "viridis",
                           tight_layout = True, plot_kwargs = None):
    """Calls :func:`whitened_pca` and plots scatter using top 2 eigenvectors.

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


# decorator ensures that call to get_params doesn't return estimators if
# safe = True. then, estimators will be replaced by their full class names.
@json_safe_get_params
class ScoringKernelPCA(KernelPCA):
    """Extension of scikit-learn kernel PCA class that provides scoring method.

    Scoring method fits the specified estimator to an exogenously provided
    response vector. The score returned by the :meth:`score` method is also
    specified in the constructor and must correspond to the name of a function
    in :mod:`sklearn.metrics`, for example ``f1_score`` or ``accuracy_score``.

    We describe new parameters below. Others are the same as the ones used in
    the scikit-learn kernel PCA. See `kernel PCA documentation`__ for details.

    .. __: https://scikit-learn.org/stable/modules/generated/sklearn.
       decomposition.KernelPCA.html

    :param estimator: Scikit-learn compatible estimator that implements the
        ``fit`` and ``predict`` methods. If not provided, then the default
        logistic regression classifier provided in scikit-learn is used.
    :type estimator: scikit-learn compatible estimator, optional
    :param metric: Metric to use when computing score, default the ``score``
        method of ``estimator``. If provided, then it must be a valid metric
        defined in ``sklearn.metrics``, like ``r2_score`` or ``accuracy_score``.
    :type metric: str, optional
    :param whiten: Whether to add whitening to the output of :meth:`transform`.
        Whitening ensures that the transformed data has an identity covariance
        matrix and is accomplished by multiplying the data by the square root of
        the number of examples and dividing each column by the corresponding
        square root of the eigenvalue of the (centered) kernel matrix.
    :type whiten: bool, optional
    """
    def __init__(self, estimator = None, *, metric = None, whiten = False,
                 n_components = None, kernel = "linear", gamma = None,
                 degree = 3, coef0 = 1, kernel_params = None, alpha = 1.0,
                 fit_inverse_transform = False, eigen_solver = "auto",
                 tol = 0, max_iter = None, remove_zero_eig = False,
                 random_state = None, copy_X = True, n_jobs = None):
        KernelPCA.__init__(
            self, n_components = n_components, kernel = kernel, gamma = gamma,
            degree = degree, coef0 = coef0, kernel_params = kernel_params,
            alpha = alpha, fit_inverse_transform = fit_inverse_transform,
            eigen_solver = eigen_solver, tol = tol, max_iter = max_iter,
            remove_zero_eig = remove_zero_eig, random_state = random_state,
            copy_X = copy_X, n_jobs = n_jobs
        )
        # if estimator is None, use scikit-learn LogisticRegression
        if estimator is None:
            estimator = LogisticRegression()
        self.estimator = estimator
        self.whiten = whiten
        # if None, use default estimator metric
        if metric is None:
            self._scorer = None
        # check that metric is in sklearn.metrics and set metric
        elif hasattr(sklearn.metrics, metric):
            self._scorer = getattr(sklearn.metrics, metric)
        # else error
        else:
            raise AttributeError(
                f"sklearn.metrics does not have member {metric}"
            )
        # set metric
        self.metric = metric

    def score(self, X, y):
        """Compute score on supervised data set.

        ``X`` will first be transformed and then the class's estimator will be
        fitted on ``Z``, the transformed ``X``, and ``y``, the response vector.
        Predictions will be made and the class's chosen scoring method will
        compute the score that will be returned.

        :param X: Input matrix, shape ``(n_samples, n_features)``.
        :type X: :class:`numpy.ndarray`
        :param y: Response vector, shape ``(n_samples,)``
        :type y: :class:`numpy.ndarray`

        :rtype: float
        """
        # if not fitted, raise RuntimeError
        if not hasattr(self, "lambdas_"):
            raise RuntimeError("cannot score with unfitted kernel PCA")
        # transform X, fit on X and y, make predictions, and return score. if
        # whiten = True then the transformation includes whitening.
        X = self.transform(X)
        self.estimator.fit(X, y)
        # if no scorer, use default score method
        if self._scorer is None:
            return self.estimator.score(X, y)
        # else make predictions and use class's chosen scoring method
        y_pred = self.estimator.predict(X)
        return self._scorer(y, y_pred)

    def transform(self, X):
        """Transform ``X``.

        Equivalent to right-multiplying the centered kernel matrix with the
        matrix of unit eigenvectors of the centered kernel matrix. If
        ``whiten = True`` was passed to the constructor, then the data is also
        whitened, i.e. adjusted to have an identity covariance matrix.

        :param X: Input matrix, shape ``(n_samples, n_features)``.
        :type X: :class:`numpy.ndarray`
        :returns: Coordinates of ``X`` in new eigenbasis, shape
            ``(n_samples, n_components)``.
        :rtype: :class:`numpy.ndarray`
        """
        # transform.
        Z = KernelPCA.transform(self, X)
        # apply whitening so covariance is identity matrix
        if self.whiten:
            Z = math.sqrt(X.shape[0]) / np.sqrt(self.lambdas_) * Z
        return Z


# save cv results for the full and reduced kernel PCA
@persist_csv(
    target = BASE_RESULTS_DIR + "/vte_whitened_kernel_pca_cv_full.csv",
    enabled = True,
    out_transform = lambda x: x["cv_results"][0]
)
@persist_csv(
    target = BASE_RESULTS_DIR + "/vte_whitened_kernel_pca_cv_red.csv",
    enabled = True,
    out_transform = lambda x: x["cv_results"][1]
)
# persist ScoringKernelPCA hyperparameters and actual models. note that passing
# safe = True replaces estimators with their full class names. deep = True by
# default so the nested estimator hyperparameters are included.
@persist_json(
    target = BASE_RESULTS_DIR + "/vte_whitened_kernel_pca_params.json",
    enabled = True,
    out_transform = lambda x: (x["pcas"][0].get_params(safe = True), 
                               x["pcas"][1].get_params(safe = True))
)
@persist_pickle(
    target = BASE_RESULTS_DIR + "/vte_whitened_kernel_pcas.pickle",
    enabled = True,
    out_transform = lambda x: x["pcas"]
)
def whitened_kernel_pca(
    *, report = False, plotting_dir = BASE_RESULTS_DIR, random_seed = None,
    figsize = (12, 4), metric = "f1_score", cv = 3, n_jobs = -1,
    fig_fname = "vte_whitened_kernel_pcas_pct_trace.png", dpi = 150,
    tight_layout = True, plot_kwargs = None
):
    """Analysis method that performs whitened kernel PCA on the VTE data set.

    Kernel PCA is performed twice for each kernel type, first on all the columns
    specified by ``VTE_CONT_INPUT_COLS`` and then on the seven columns chosen by
    a :class:`sklearn.feature_selection.SelectKBest` instance using our
    :func:`mtml.feature_selection.univariate.roc_auc_score_func` score function.

    The effectiveness of the kernel PCA is chosen by running a logistic
    regression classifier on the transformed data, computing a score given the
    metric specified in the list of arguments, and then using scikit-learn
    GridSearchCV to perform grid search and pick the kernel parameters that give
    the highest score. The number of CV splits does not need to be high. Note
    that the runtime for this function is quite long since each fit takes about
    a minute or two and becuase there are ``10 * cv`` fits being done.

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
    # get reduced X_train and X_test by selecting only the 7 selected columns
    best_cols = list(
        pd.read_csv(
            BASE_RESULTS_DIR + "/vte_selected_cols.csv", index_col = 0
        ).index
    )
    X_train_red, X_test_red, _, _ = vte_slp_factory(
        data_transform = replace_hdl_tot_chol_with_ratio,
        inputs = best_cols, targets = VTE_OUTPUT_COLS, dropna = True,
        random_state = random_seed
    )
    # fit StandardScalers to training data and use them to normalize the data
    # before feeding training data to kernel PCA. we don't really need to do
    # this, but this is for consistency with all our other work.
    scaler_full, scaler_red = StandardScaler(), StandardScaler()
    scaler_full.fit(X_train), scaler_red.fit(X_train_red)
    X_train, X_test = (
        scaler_full.transform(X_train), scaler_full.transform(X_test)
    )
    X_train_red, X_test_red = (
        scaler_red.transform(X_train_red), scaler_red.transform(X_test_red)
    )
    # list of kernels to cycle through. note that "laplacian" is not shown as a
    # choice in the documentation, but internally, kernel computation is done
    # using sklearn.metrics.pairwise.pairwise_kernels, which does accept it.
    kernels = dict(
        kernel = ["linear", "poly", "rbf", "laplacian"]
    )
    # for both X_train and X_train_red, grid search across the kernels to find
    # the best kernel cross-validated on the training data. again use F1-score,
    # which is done internally in the ScoringKernelPCA. whitening is again
    # applied to be friendly for scaling-sensitive models.
    pca_full_gscv = GridSearchCV(
        ScoringKernelPCA( # kernel PCA for full continuous columns
            estimator = LogisticRegression(random_state = random_seed,
                                           class_weight = "balanced"),
            metric = metric, whiten = True, random_state = random_seed
        ), kernels, n_jobs = n_jobs, cv = cv
    )
    pca_red_gscv = GridSearchCV(
        ScoringKernelPCA( # kernel PCA for 7 highest AUC columns
            estimator = LogisticRegression(random_state = random_seed,
                                           class_weight = "balanced"),
            metric = metric, whiten = True, random_state = random_seed
        ), kernels, n_jobs = n_jobs, cv = cv
    )
    # fit on the (pre-standardized) training data. use y_train in order for the
    # score method of the ScoringKernelPCA to work correctly.
    pca_full_gscv.fit(X_train, y_train)
    pca_red_gscv.fit(X_train_red, y_train)
    # best ScoringKernelPCA for each grid search across kernels
    pca_full = pca_full_gscv.best_estimator_
    pca_red = pca_red_gscv.best_estimator_
    # compute traces for full and reduced best-fittig kernel PCAs
    trace_full = pca_full.lambdas_.sum()
    trace_red = pca_red.lambdas_.sum()
    # n_components needed to explain 95% variance for full and reduced data
    n_full_95 = n_eigs_pct_trace(pca_full.lambdas_, presort = False)
    n_red_95 = n_eigs_pct_trace(pca_red.lambdas_, presort = False)
    # get truncated GridSearchCV results dicts. contains info on kernel used,
    # mean test score, rank of the test scores, and mean fit time.
    cv_cols = [
        "param_kernel", "mean_test_score", "rank_test_score", "mean_fit_time"
    ]
    cv_results_full = pd.DataFrame(pca_full_gscv.cv_results_).loc[:, cv_cols]
    cv_results_red = pd.DataFrame(pca_red_gscv.cv_results_).loc[:, cv_cols]
    # print explained variance ratios (eigenvalues standardized by trace) if
    # the option to report is True
    if report is True:
        # for each kernel PCA, kernel PCA CV results, trace, n_components needed
        # to explain 95% of the data variance in the higher dimensional space
        for pca, cv_results, trace, n_comp_95, title in zip(
            (pca_full, pca_red), (cv_results_full, cv_results_red),
            (trace_full, trace_red), (n_full_95, n_red_95),
            ("---- VTE kernel PCA on VTE_CONT_INPUT_COLS " + "-" * 37,
             "---- VTE kernel PCA on top 7 columns " + "-" * 43)
        ):
            print(title, end = "\n\n")
            # if pca == pca_red, then also print the selected columns. using
            # np.array on the list best_cols allows it to wrap at 80 columns.
            if pca == pca_red:
                print("selected columns (by univariate AUC):\n"
                      f"{np.array(best_cols)}\n")
            # print out estimator, cv folds, metric info
            print(f"estimator: {pca.estimator.__class__.__name__}\n"
                  f"cv folds: {cv}\nmetric: {metric}\n")
            # best kernel, CV results, normalized eigenvalues (clipped since too
            # many) + the number of components needs to explain 95% variance
            print(f"best kernel: {pca.kernel}\n\ncv results:\n{cv_results}\n")
            print(
                f"best explained variance ratios:\n{pca.lambdas_ / trace}\n\n"
                f"n_components needed to explain 95% variance: {n_comp_95}\n"
            )
    # if plotting dir is None, don't plot
    if plotting_dir is None:
        pass
    # else make plots for full and reduced PCA and write plots to plotting_dir
    else:
        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = figsize)
        # plot
        for ax, pca, ax_title in zip(
            axs, (pca_full, pca_red),
            ("VTE_CONT_INPUT_COLS", "top 7 columns by AUC")
        ):
            # make scree plot. don't need to presort eigenvalues (already in
            # descending order), add vertical line at 95% mark. no manual
            # x-ticks since there are thousands of eigenvalues. replace C with
            # K since this is a kernel matrix.
            normalized_scree_plot(
                ax, pca.lambdas_, presort = False, pct_trace = 0.95,
                matrix_letter = "K", plot_title = ax_title,
            )
        # set overall title for the figure
        fig.suptitle(
            r"Percent trace of $ \mathbf{K} \triangleq "
            r"\phi\left(\mathbf{XX}^\top\right) $, "
            r"$ \phi : \mathbb{R}^{n \times n} \rightarrow "
            r"\mathbb{R}^{n \times n} \text{" + str(pca.kernel) + r"} $, "
            r"per eigenvalue $ \lambda_i $", size = "x-large"
        )
        # if tight_layout, call tight_layout
        if tight_layout is True:
            fig.tight_layout()
        # save figure at dpi at plotting_dir/fig_fname
        fig.savefig(plotting_dir + "/" + fig_fname, dpi = dpi)
    # return PCA objects and (standardized) data so that they can be
    # appropriately persisted/passed to another analysis/model fitting method.
    # CV results are also included as an extra bonus.
    return {
        "pcas": (pca_full, pca_red), 
        "data_train": (X_train, X_train_red, y_train),
        "data_test": (X_test, X_test_red, y_test),
        "cv_results": (cv_results_full, cv_results_red)
    }


if __name__ == "__main__":
    # _ = whitened_pca(report = True, random_seed = 7)
    # _ = whitened_kernel_pca(report = True, random_seed = 7)
    # _ = plot_pca_components_2d(random_seed = 7)
    pass