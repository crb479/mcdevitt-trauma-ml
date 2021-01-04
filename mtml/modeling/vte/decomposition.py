__doc__ = """Data decomposition analyses performed on VTE data.

PCA originals found in :mod:`mtml.modeling._vte_models.py`.

For example, principal components analysis (eigendecomposition/SVD).
"""

# pylint: disable=import-error
from dask.distributed import Client, LocalCluster
import dask_ml.model_selection
from functools import partial
import gc
import joblib
import matplotlib.cm
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import KernelPCA, PCA
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import sys

# pylint: disable=relative-beyond-top-level
from .. import BASE_RESULTS_DIR
from ... import VTE_CONT_INPUT_COLS, VTE_OUTPUT_COLS, VTE_QIDA_INPUT_COLS
from ...data.vte import vte_slp_factory
from .data_transforms import replace_hdl_tot_chol_with_ratio
from ...feature_selection.univariate import roc_auc_score_func
from ...utils.extmath.eigenvalues import n_eigs_pct_trace
from ...utils.models import json_safe_get_params
from ...utils.path import find_results_home_ascending
from ...utils.persist import persist_csv, persist_json, persist_pickle
from ...utils.plotting import normalized_scree_plot


def whitened_pca(*, report = False, stream = sys.stdout, random_seed = None):
    """Analysis method that performs whitened PCA on the VTE data set.

    PCA is performed twice, first on all the columns specified by
    ``VTE_CONT_INPUT_COLS`` and then on the seven columns chosen by a
    :class:`sklearn.feature_selection.SelectKBest` instance using our homegrown
    :func:`mtml.feature_selection.univariate.roc_auc_score_func` score function.

    A report may optionally be printed which shows the explained variance ratios
    (normalized eigenvalues). Plots of the explained variance ratios can be
    plotted with :func:`whitened_pca_scree_from_pickle` if the PCA estimators
    of this object have been appropriately pickled.

    Parameter descriptions in progress.
    
    :rtype: dict
    """
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
            print(title, end = "\n\n", file = stream)
            # print the selected columns if pca is pca_red
            if pca == pca_red:
                print(
                    f"selected columns (by univariate AUC):\n{kbest_cols}\n",
                    file = stream
                )
            print(
                f"explained variance ratios:\n{pca.explained_variance_ratio_}"
                f"\n\nn_components needed to explain 95% variance: {n_comp_95}",
                end = "\n\n", file = stream
            )
    # return PCA objects and (standardized) data so that they can be
    # appropriately persisted/passed to another analysis/model fitting method
    return {
        "pcas": (pca_full, pca_red), 
        "data_train": (X_train, X_train_red, y_train),
        "data_test": (X_test, X_test_red, y_test)
    }


def plot_pca_components_2d(
    *, plotting_dir = BASE_RESULTS_DIR, random_seed = None, figsize = (8, 6),
    fig_fname = "vte_whitened_pca_2d_eig.png", dpi = 150, cmap = "viridis",
    tight_layout = True, plot_kwargs = None
):
    """Calls :func:`whitened_pca` and plots scatter using top 2 eigenvectors.

    .. note:: Inefficient due to embedded calling of :func:`whitened_pca`.

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
    # get results from whitened_pca
    pca_res = whitened_pca(random_seed = random_seed)
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


def whitened_kernel_pca(
    *, report = False, stream = sys.stdout, random_seed = None,
    metric = "f1_score", copy_X = False, cv = 3, backend = "loky",
    cluster = None, n_jobs = 1, mmap_dir = None, verbosity = 0
):
    """Analysis method that performs whitened kernel PCA on the VTE data set.

    .. note::
    
       Although this method can use multiprocessing to parallelize the grid
       search, it is still limited to use on a single machine (node), i.e. it
       cannot be run in a distributed fashion.

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
    a minute or two and because there are ``10 * cv`` fits being done.

    A report may optionally be printed which shows the explained variance ratios
    and if ``plotting_dir`` is not ``None``, then plots of the explained
    variance ratios will be written to the directory path provided to
    ``plotting_dir``.

    :param report: Whether or not to produce a report after fitting
    :type report: bool, optional
    :param stream: Text file stream to print report to if ``report = True``.
        Defaults to :class:`sys.stdout`.
    :type stream: :class:`io.TextIOBase` subclass, optional
    :param random_seed: Global random seed for reproducible output
    :type random_seed: int, optional
    :param metric: Metric to pass to the :class:`ScoringKernelPCA` object. Must
        be a valid member of :mod:`sklearn.metrics`.
    :type metric: str, optional
    :param cv: Number of cross-validation folds to fit kernel PCAs on.
    :type cv: int, optional
    :param backend: Name of backend to use for multiprocessing. Defaults to
        ``"loky"``. Set to ``"dask"`` in order to use the ``cluster`` argument.
    :type backend: str, optional
    :param cluster: :class:`distributed.deploy.spec.SpecCluster` subclass
        instance that represents a [distributed] cluster. Unless
        ``backend = "dask"``, ``cluster`` will be ignored. The ``scale`` method
        of ``cluster`` will be invoked in this function.
    :type cluster: :class:`distributed.deploy.spec.SpecCluster` subclass
        instance, optional
    :param n_jobs: Number of processes for ``joblib`` to use for parallel
        multiprocessing execution of grid search, or when ``cluster`` is not
        ``None`` and when ``backend = "dask"``, the number of worker jobs that
        ``cluster.scale`` should start.
    :type n_jobs: int, optional
    :param mmap_dir: Directory where input data will be dumped to disk and then
        accessed as a :class:`numpy.memmap`.
    :type mmap_dir: str, optional
    :param verbosity: Level of verbosity of the GridSearchCV. Has no effect
        when ``backend = "dask"` since
        :class:`dask_ml.model_selection.GridSearchCV` does not accept the
        ``verbose`` kwarg in its constructor.
    :type verbosity: int, optional
    :rtype: dict
    """
    # if backend == "dask"
    if backend == "dask":
        # if cluster is None, start Client with LocalCluster, n_jobs workers
        # pylint: disable=unused-variable
        if cluster is None:
            client = Client(LocalCluster(n_workers = n_jobs))
        # else call cluster.scale and pass cluster straight into Client
        else:
            # start jobs n_jobs workers (may have multiple threads, processes)
            cluster.scale(jobs = n_jobs)
            client = Client(cluster)
        # pylint: enable=unused-variable
    # get data set of continuous features from vte_slp_factory
    X_train, X_test, y_train, y_test = vte_slp_factory(
        data_transform = replace_hdl_tot_chol_with_ratio,
        inputs = VTE_CONT_INPUT_COLS, targets = VTE_OUTPUT_COLS, dropna = True,
        random_state = random_seed
    )
    # get reduced X_train and X_test by selecting only the 7 selected columns
    X_train_red, X_test_red, _, _ = vte_slp_factory(
        data_transform = replace_hdl_tot_chol_with_ratio,
        inputs = VTE_QIDA_INPUT_COLS, targets = VTE_OUTPUT_COLS, dropna = True,
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
    # if mmap_dir is not None, then memory-map the inputs and outputs. dump and
    # reload everything so the forked processes have a smaller size
    if mmap_dir is not None:
        # joblib.dump returns the names of the files the data is pickled to in
        # a list so we have to index the names. since we are pickling one array
        # at a time, there will be only one filename in each list.
        X_train_loc = joblib.dump(
            X_train, mmap_dir + "/whitened_kernel_pca__X_train.pickle"
        )[0]
        X_train = joblib.load(X_train_loc, mmap_mode = "r")
        X_test_loc = joblib.dump(
            X_test, mmap_dir + "/whitened_kernel_pca__X_test.pickle"
        )[0]
        X_test = joblib.load(X_test_loc, mmap_mode = "r")
        X_train_red_loc = joblib.dump(
            X_train_red, mmap_dir + "/whitened_kernel_pca__X_train_red.pickle"
        )[0]
        X_train_red = joblib.load(X_train_red_loc, mmap_mode = "r")
        X_test_red_loc = joblib.dump(
            X_test_red, mmap_dir + "/whitened_kernel_pca__X_test_red.pickle"
        )[0]
        X_test_red = joblib.load(X_test_red_loc, mmap_mode = "r")
        y_train_loc = joblib.dump(
            y_train, mmap_dir + "/whitened_kernel_pca__y_train.pickle"
        )[0]
        y_train = joblib.load(y_train_loc, mmap_mode = "r")
        y_test_loc = joblib.dump(
            y_test, mmap_dir + "/whitened_kernel_pca__y_test.pickle"
        )[0]
        y_test = joblib.load(y_test_loc, mmap_mode = "r")
        # collect garbage
        gc.collect()
    # list of kernels to cycle through. note that "laplacian" is not shown as a
    # choice in the documentation, but internally, kernel computation is done
    # using sklearn.metrics.pairwise.pairwise_kernels, which does accept it.
    kernels = dict(kernel = ["linear", "poly", "rbf", "laplacian"])
    # use the dask_ml version of GridSearchCV with client as scheduler if
    # backend = "dask" (don't return training scores)
    if backend == "dask":
        # set dask_ml GridSearchCV scheduler to client and don't use train score
        grid_search_cls = partial(
            dask_ml.model_selection.GridSearchCV, scheduler = client,
            return_train_score = False
        )
    # else use normal GridSearchCV
    else:
        grid_search_cls = GridSearchCV
    # number of local jobs (processes). -1 if using dask, n_jobs otherwise
    n_local_jobs = -1 if backend == "dask" else n_jobs
    # dask_ml GridSearchCV does not accept verbose kwarg in constructor
    verbosity_ = {} if backend == "dask" else {"verbose": verbosity}
    # for both X_train and X_train_red, grid search across the kernels to find
    # the best kernel cross-validated on the training data. again use F1-score,
    # which is done internally in the ScoringKernelPCA. whitening is again
    # applied to be friendly for scaling-sensitive models. use n_jobs = -1 if
    # backend is dask, else just pass n_jobs in
    pca_full_gscv = grid_search_cls(
        ScoringKernelPCA( # kernel PCA for full continuous columns
            estimator = LogisticRegression(random_state = random_seed,
                                           class_weight = "balanced"),
            metric = metric, whiten = True, random_state = random_seed,
            copy_X = copy_X
        ), 
        kernels, cv = cv, n_jobs = n_local_jobs, **verbosity_
    )
    pca_red_gscv = grid_search_cls(
        ScoringKernelPCA( # kernel PCA for 7 highest AUC columns
            estimator = LogisticRegression(random_state = random_seed,
                                           class_weight = "balanced"),
            metric = metric, whiten = True, random_state = random_seed,
            copy_X = copy_X
        ),
        kernels, cv = cv, n_jobs = n_local_jobs, **verbosity_
    )
    # fit on the (pre-standardized) training data. use y_train in order for
    # the score method of the ScoringKernelPCA to work correctly.
    pca_full_gscv.fit(X_train, y_train)
    pca_red_gscv.fit(X_train_red, y_train)
    # best ScoringKernelPCA for each grid search across kernels
    pca_full = pca_full_gscv.best_estimator_
    pca_red = pca_red_gscv.best_estimator_
    # compute traces for full and reduced best-fitting kernel PCAs
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
            # if pca == pca_red, then also print VTE_QIDA_INPUT_COLS. using
            # np.array on the tuple allows wrapping at 80 columns.
            if pca == pca_red:
                print("selected columns (by univariate AUC):\n"
                      f"{np.array(VTE_QIDA_INPUT_COLS)}\n", file = stream)
            # print out estimator, cv folds, metric info
            print(f"estimator: {pca.estimator.__class__.__name__}\n"
                  f"cv folds: {cv}\nmetric: {metric}\n", file = stream)
            # best kernel, CV results, normalized eigenvalues (clipped since too
            # many) + the number of components needs to explain 95% variance
            print(
                f"best kernel: {pca.kernel}\n\ncv results:\n{cv_results}\n",
                file = stream
            )
            print(
                f"best explained variance ratios:\n{pca.lambdas_ / trace}\n\n"
                f"n_components needed to explain 95% variance: {n_comp_95}\n",
                file = stream
            )
    # if mmap_dir is not None, load dense arrays from pickles
    if mmap_dir is not None:
        X_train = joblib.load(X_train_loc)
        X_test = joblib.load(X_test_loc)
        X_train_red = joblib.load(X_train_red_loc)
        X_test_red = joblib.load(X_test_red_loc)
        y_train = joblib.load(y_train_loc)
        y_test = joblib.load(y_test_loc)
        # hopefully clean up garbage
        gc.collect()
    # return PCA objects and (standardized) data so that they can be
    # appropriately persisted/passed to another analysis/model fitting method.
    # CV results are also included as an extra bonus.
    return {
        "pcas": (pca_full, pca_red), 
        "data_train": (X_train, X_train_red, y_train),
        "data_test": (X_test, X_test_red, y_test),
        "cv_results": (cv_results_full, cv_results_red)
    }


# usually calling with default arguments is fine unless whitened_pca is pickling
# pickling its ScoringKernelPCA objects somewhere else
def whitened_pca_scree_from_pickle(
    pickle_path = BASE_RESULTS_DIR + "/vte_whitened_pcas.pickle", *,
    fig_path = BASE_RESULTS_DIR + "/vte_whitened_pcas_pct_trace.png",
    orientation = "vertical", pct_trace = 0.95, figsize = (6, 8),
    matrix_letter = "C", plot_marker = "s", dpi = 150, tight_layout = True,
    # note plot_xticks are hard-coded, but i didn't have another option here
    plot_xticks = (1 + np.arange(12), 1 + np.arange(7)), plot_kwargs = None
):
    """Using pickled PCA objects, create normalized scree plots.

    Uses pickled PCAs from :func:`whitened_pca` stored in ``pickle_path`` and
    then calls :func:`_pcas_scree_from_pickle` to create the figure with the two
    relevant scree plots and with appropriate formatting.

    :param pickle_path: Path to the pickle file holding the tuple of the two
        scikit-learn PCA estimators.
    :type pickle_path: str
    :param fig_path: Path to write the resulting figure to
    :type fig_path: str
    :param orientation: Whether the two scree plots should be stacked or be put
        side by side (``True`` to stack).
    :type orientation: bool, optional
    :param pct_trace: Percent of trace that controls where the vertical lines
        are drawn in each scree plot. ``None`` for no vertical line.
    :type pct_trace: float, optional
    :param figsize: ``(width, height)`` of figure in inches.
    :type figsize: tuple, optional
    :param matrix_letter: Letter for the matrix printed in the scree plots.
    :type matrix_letter: str, optional
    :param plot_marker: Valid marker value for :func:`matplotlib.axes.Axes.plot`
    :type plot_marker: str, optional
    :param dpi: DPI of the saved figure
    :type dpi: int, optional
    :param tight_layout: ``True`` to call the figure's
        :func:`matplotlib.figure.Figure.tight_layout` method.
    :type tight_layout: bool, optional
    :param plot_xticks:
    :type plot_xticks: :class:`numpy.ndarray`, optional
    :param plot_kwargs: Other keyword args for :func:`matplotlib.axes.Axes.plot`
    :type plot_kwargs: dict, optional
    """
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be \"vertical\" or \"horizontal\"")
    # call _pcas_scree_from_pickle to compute normalized scree plots
    fig, _ = _pcas_scree_from_pickle(
        pickle_path, "explained_variance_ratio_",
        ("VTE_CONT_INPUT_COLS", "top 7 columns by AUC"),
        max_rows = 2 if "vertical" else 1, max_cols = 1 if "vertical" else 2,
        figsize = figsize, fig_title = (
            r"Percent trace of $ \mathbf{C} \triangleq \frac{1}{n}"
            r"\mathbf{X}^\top\mathbf{X} $ per eigenvalue $ \lambda_i $"
        ), normalize = False, presort = False, pct_trace = pct_trace,
        matrix_letter = matrix_letter, plot_marker = plot_marker,
        plot_xticks = plot_xticks, plot_kwargs = plot_kwargs,
        tight_layout = tight_layout
    )
    # save figure at dpi at plotting_dir/fig_fname
    fig.savefig(fig_path, dpi = dpi)
    # close manually
    matplotlib.pyplot.close(fig)


# usually calling with default arguments is fine unless whitened_kernel_pca is
# pickling its ScoringKernelPCA objects somewhere else
def whitened_kernel_pca_scree_from_pickle(
    pickle_path = BASE_RESULTS_DIR + "/vte_whitened_kernel_pcas.pickle", *,
    fig_path = BASE_RESULTS_DIR + "/vte_whitened_kernel_pcas_pct_trace.png",
    orientation = "vertical", kernel = "laplacian", pct_trace = 0.95,
    figsize = (6, 8), matrix_letter = "K", plot_marker = "D", dpi = 150,
    tight_layout = True, plot_xticks = None, plot_kwargs = dict(markersize = 3)
):
    """Using pickled ScoringKernelPCA objects, create normalized scree plots.

    Uses pickled :class:`ScoringKernelPCA` objects from
    :func:`whitened_kernel_pca` stored in ``pickle_path`` and then calls
    :func:`_pcas_scree_from_pickle` to create the figure with the two relevant
    scree plots and with appropriate formatting.

    Omitted named parameters have the same function as they do in
    :func:`whitened_pca_scree_from_pickle` defined above.

    :param kernel: Name of the kernel used in the best ScoringKernelPCA objects
        that are pickled to ``pickle_path``. This can be checked by unpickling
        the tuple of ScoringKernelPCA objects and checking their ``kernel``
        attribute or by seeing the report printed by
        :func:`whitened_kernel_pca` after its execution.
    :type kernel: str, optional
    """
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be \"vertical\" or \"horizontal\"")
    # call _pcas_scree_from_pickle to compute normalized scree plots
    fig, _ = _pcas_scree_from_pickle(
        pickle_path, "lambdas_",
        ("VTE_CONT_INPUT_COLS", "top 7 columns by AUC"),
        max_rows = 2 if "vertical" else 1, max_cols = 1 if "vertical" else 2,
        figsize = figsize, fig_title = (
            r"Percent trace of $ \mathbf{K} \triangleq "
            r"\mathbf{X}_\phi\mathbf{X}_\phi^\top $, " + kernel + "\n"
            r"$ \mathbf{K}_{ij} \triangleq \phi(\mathbf{x}_i)^\top"
            r"\phi(\mathbf{x}_j) $, per eigenvalue $ \lambda_i $"
        ), presort = False, pct_trace = pct_trace,
        matrix_letter = matrix_letter, plot_marker = plot_marker,
        plot_xticks = plot_xticks, plot_kwargs = plot_kwargs,
        tight_layout = tight_layout
    )
    # save figure at dpi at plotting_dir/fig_fname
    fig.savefig(fig_path, dpi = dpi)
    # close manually
    matplotlib.pyplot.close(fig)


# horizontal figure size was (12, 4)
def _pcas_scree_from_pickle(
    pickle_path, eig_attr_name, ax_titles, *, max_rows = None, max_cols = 1,
    figsize = (6, 8), fig_title = None, normalize = True, presort = True,
    pct_trace = 0.95, vline_color = "red", vline_kwargs = None,
    matrix_letter = "C", plot_marker = "s", show_legend = True,
    plot_xticks = None, plot_kwargs = None, tight_layout = True
):
    """Generic method for creating normalized scree plots from sklearn PCA.

    The PCA objects are read from a pickle file. Since scikit-learn PCA and
    KernelPCA have different attribute names, ``eig_attr_name`` is required to
    specify which attribute holds the [normalized] eigenvalues. Most parameters
    are simply the default values that are passed through to the function
    :func:`mtml.utils.plotting.normalized_scree_plot`.

    If ``max_cols = None``, i.e. number of columns is chosen automatically, then
    the resulting plot will only have two columns.

    The parameters shown below are specific to this function. Any omitted
    named parameters are described in the docstring for
    :func:`mtml.utils.plotting.normalized_scree_plot`.

    :param pickle_path: Path to pickle file containing a list of objects that
        have an attribute containing relevant eigenvalues. These should all be
        same class of objects with the same attribute holding the eigenvalues.
    :type pickle_path: str
    :param eig_attr_name: Name of the object attribute holding the eigenvalues.
    :type eig_attr_name: str
    :param ax_titles: List/array of strings giving titles for each of the scree
        plots that will be made. Must equal the number of PCA-like objects that
        will be read from the pickle file.
    :type ax_titles: list
    :param max_rows: Maximum number of rows the plot can have. If there are not
        enough rows, then an error is raised. Leave as ``None`` for the function
        to automatically determine how many rows the plot will have.
    :type max_rows: int, optional
    :param max_cols: Maximum number of columns the plot can have. If there are
        not enough columns, then an error is raised. Lave as ``None`` for the
        function to automatically set number of columns to 2.
    :type max_cols: int, optional
    :param figsize: Tuple of ``(width, height)`` in inches.
    :type figsize: tuple, optional
    :param fig_title: Title for the overall figure holding the scree plots. If
        ``None``, then no title is created for the overall figure (suptitle).
    :type fig_title: str, optional
    :param plot_xticks: Array-like of x axis tick locations for each subplot. If
        not ``None``, then must have length equal to number of plots.
    :type plot_xticks: array-like, optional
    :returns: Figure and Axes tuple
    :rtype: tuple
    """
    # load PCA objects from disk (must be a list/tuple of them)
    with open(pickle_path, "rb") as f:
        pcas = pickle.load(f)
    # ax_titles must have same length as pcas
    if len(pcas) != len(ax_titles):
        raise ValueError("must have same number of axis titles as PCA objects")
    # number of scree plots that will have to be made
    n_plots = len(pcas)
    # if plot_xticks is None, set to list of None
    if plot_xticks is None:
        plot_xticks = [None] * n_plots
    # plot_xticks must have same length as pcas
    if n_plots != len(plot_xticks):
        raise ValueError(
            "number of x ticks specifications must equal number of PCA objects"
        )
    # figure out how many rows and columns. if max_cols is None, it is set to
    # 2 automatically. then max_rows is adjusted as needed.
    if max_cols is None:
        max_cols = 2
    if max_rows is None:
        max_rows = n_plots // max_cols
        # add one more row if not evenly divisible
        if max_rows % max_cols > 0:
            max_rows = max_rows + 1
    # check that max_rows * max_cols >= n_plots. if not, error
    if max_rows * max_cols < n_plots:
        raise ValueError(
            "max_rows and/or max_cols values are too small. pass None for "
            "automatic determination of number of rows and/or columns or "
            "increase the values of max_rows, max_cols"
        )
    # make plots for full and reduced PCA and write plots to plotting_dir
    fig, axs = plt.subplots(nrows = max_rows, 
                            ncols = max_cols, figsize = figsize)
    # flatten the axes
    axs = axs.ravel()
    # plot
    for ax, pca, ax_title, xticks in zip(axs, pcas, ax_titles, plot_xticks):
        # make scree plot
        normalized_scree_plot(
            ax, getattr(pca, eig_attr_name), presort = presort,
            pct_trace = pct_trace, vline_color = vline_color,
            vline_kwargs = vline_kwargs, matrix_letter = matrix_letter,
            plot_title = ax_title, plot_marker = plot_marker,
            plot_xticks = xticks, plot_kwargs = plot_kwargs
        )
    # set overall title for the figure
    fig.suptitle(fig_title, size = "x-large")
    # if tight_layout, call tight_layout
    if tight_layout is True:
        fig.tight_layout()
    # return figure and axes
    return fig, axs


if __name__ == "__main__":
    # _ = whitened_pca(report = True, random_seed = 7)
    # _ = whitened_kernel_pca(report = True, random_seed = 7)
    # _ = plot_pca_components_2d(random_seed = 7)
    pass