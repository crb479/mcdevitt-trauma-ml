__doc__ = """Use isolation forest in attempt to detect minority class examples.

Idea is to forget about the two-class classification problem and attempt to use
an siolation forest to give predictions as to whether a point is in the inlier
group (majority class) or in the outlier group (minority class).

Inspired by `this KDNuggets post`__.

.. __: https://www.kdnuggets.com/2016/08/learning-from-imbalanced-classes.html/2
"""

from functools import partial
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# pylint: disable=relative-beyond-top-level
from ... import VTE_CONT_INPUT_COLS, VTE_OUTPUT_COLS
from .. import BASE_RESULTS_DIR
from ...data.vte import vte_slp_factory
from .data_transforms import replace_hdl_tot_chol_with_ratio
from ...utils.persist import persist_csv, persist_json, persist_pickle


class IsolationForestClassifier(IsolationForest):
    """A binary classifier built from the scikit-learn isolation forest.

    Init parameters are identical to those of the scikit-learn isolation forest.
    See the `isolation forest documentation`__ for details.

    .. __: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.
       IsolationForest.html
    """
    # note removed deprecated behavior parameter
    def __init__(self, *, n_estimators = 100, max_samples = "auto",
                 contamination = "auto", max_features =1., bootstrap = False,
                 n_jobs = None, random_state = None, 
                 verbose = 0, warm_start = False):
        # note we need to name all the keyword arguments explicitly since the
        # sklearn get_params method inspects the __init__ signature to determine
        # which arguments can be passed to the estimator
        IsolationForest.__init__(
            self, n_estimators = n_estimators, max_samples = max_samples,
            contamination = contamination, max_features = max_features,
            bootstrap = bootstrap, n_jobs = n_jobs, random_state = random_state,
            verbose = verbose, warm_start = warm_start
        )

    def fit_predict(self, X, y = None, sample_weight = None):
        """Perform fit and then return inlier/outlier predictions.

        Minority class (outlier) points are assigned label 1 and majority class
        (inlier) points are assigned label 0. Wraps original fit_predict method.

        :rtype: :class:`numpy.ndarray`
        """
        # need functools.partial to freeze the self argument
        return isoforest_label_adjust(
            partial(IsolationForest.fit_predict, self)
        )(X, y = y, sample_weight = sample_weight)

    def predict(self, X):
        """Return inlier/outlier predictions.

        Minority class (outlier) points are assigned label 1 and majority class
        (inlier) points are assigned label 0. Wraps original predict method.

        :rtype: :class:`numpy.ndarray`
        """
        return isoforest_label_adjust(partial(IsolationForest.predict, self))(X)

    def score(self, X, y, sample_weight = None):
        """Returns accuracy of the classifier.

        :rtype: float
        """
        # get predictions and send to accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight = sample_weight)
        

def isoforest_label_adjust(pred_func):
    """Adjusts isolation forest predictions to be 1 for outliers, 0 for inliers.

    By default the scikit-learn isolation forest returns -1 for outliers and 1
    for inliers, so this method is used to wrap fit_predict or predict methods
    and return 0 for inliers, 1 for outliers.

    :param pred_func: Scikit-learn prediction function that returns a flat
        :class:`numpy.ndarray` of labels ``-1`` and ``1``.
    :type pred_func: function or method
    :rtype: function
    """
    def adjust_pred_func(*args, **kwargs):
        res = pred_func(*args, **kwargs)
        res[res == -1] = 1
        res[res == 1] = 0
        return res

    return adjust_pred_func


@persist_csv(
    target = BASE_RESULTS_DIR + "/vte_isoforest_scores.csv", enabled = True,
    out_transform = lambda x: x[2]
)
@persist_json(
    target = BASE_RESULTS_DIR + "/vte_isoforest_params.json", enabled = True,
    out_transform = lambda x: x[1]
)
@persist_pickle(
    target = BASE_RESULTS_DIR + "/vte_isoforest.pickle", enabled = True,
    out_transform = lambda x: x[0]
)
def fit_isolation_forest(*, cv = 5, n_jobs = -1, verbose = False,
                         report = False, random_seed = None):
    """Fit isolation forest to solve imbalanced VTE classification problem.

    Note that we have two variants: one which operates on the original
    continuous features and one that operates on data projected onto the
    eigenspace returned by PCA. Intuition would lead one to believe that the
    projected data would give a better result, as the new basis vectors should
    point in directions of maximal variance.

    We avoid using a Pipeline object by performing the PCA before fitting.

    :param cv: Number of CV splits to make when doing grid search.
    :type cv: int, optional
    :param n_jobs: Number of jobs to run in parallel when grid searching.
        Defaults to ``-1`` to distribute load to all threads.
    :type n_jobs: int, optional
    :param verbose: Verbosity of the
        :class:`~sklearn.model_selection.GridSearchCV` during searching/fitting.
    :type verbose: bool, optional
    :param report: If ``True``, print to stdout a report on model scores.
    :type report: bool, optional
    :param random_seed: A int seed to pass for multiple calls to this function
        to be reproducible. Leave ``None`` for stochastic behavior.
    :type random_state: int, optional
    :rtype: tuple
    """
    # get data set of continuous features from vte_slp_factory
    X_train, X_test, y_train, y_test = vte_slp_factory(
        data_transform = replace_hdl_tot_chol_with_ratio,
        inputs = VTE_CONT_INPUT_COLS, targets = VTE_OUTPUT_COLS, dropna = True,
        random_state = random_seed
    )
    # use StandardScaler to standardize the X_train, X_test data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    # fit whitened PCA and use it to transform new training and test data
    pca = PCA(whiten = True, random_state = random_seed)
    pca.fit(X_train)
    Z_train, Z_test = pca.transform(X_train), pca.transform(X_test)
    # list of base estimator names
    base_names = ("isoforest", "isoforest_pca")
    ## hyperparameter grid for isolation forest ##
    isoforest_grid = dict(
        n_estimators = [50, 100, 200],
        # twenty percent of the data (0.2) is around 2200 examples
        max_samples = ["auto", 500, 0.2, 0.4],
        # try to set contamination to be fraction of minority class
        contamination = ["auto", y_train.mean()],
        # try both sampling without and with replacement
        bootstrap = [False, True],
        random_state = [random_seed]
    )
    # two identical IsolationForest models
    base_models = (IsolationForestClassifier(), IsolationForestClassifier())
    # references to appropriate data sets each classifier is fit on
    train_refs = (X_train, Z_train)
    test_refs = (X_test, Z_test)
    # holds saved binary model results
    mdata = {}
    # dictionary to hold saved model hyperparameters for plaintext persistence
    mparams = {}
    # DataFrame indexed by name of the model where columns are accuracy,
    # precision, recall, and ROC AUC for each model
    mscores = pd.DataFrame(
        index = base_names, 
        columns = ["accuracy", "precision", "recall", "roc_auc"]
    )
    # for PCA'ed and not PCA'ed model, train + record results
    for base_name, base_model, train_ref, test_ref in zip(
        base_names, base_models, train_refs, test_refs):
        # instantiate and fit the GridSearchCV object (optimize for precision)
        model = GridSearchCV(
            base_model, isoforest_grid, scoring = "precision", cv = cv,
            n_jobs = n_jobs, verbose = int(verbose)
        )
        # note that train_ref is either X_train or Z_train (y_train needed
        # in order for our precision scoring to work)
        model.fit(train_ref, y_train)
        # save model to mdata using model name
        mdata[base_name] = model
        # get hyperparameters of the best estimated model
        params = model.best_estimator_.get_params()
        # save hyperparameters to mparams
        mparams[base_name] = params
        # compute test predictions using refit model on test_ref
        y_pred = model.predict(test_ref)
        # get decision function values for computing ROC AUC. note that we have
        # to negate them since isolation forest treats negative decision
        # function values as being for the outliers
        y_pred_scores = -model.decision_function(test_ref)
        # save accuracy, precision, and recall to in mscores
        mscores.loc[base_name, :] = (
            accuracy_score(y_test, y_pred), precision_score(y_test, y_pred),
            recall_score(y_test, y_pred), roc_auc_score(y_test, y_pred_scores)
        )
    # if report is True, print mscores to stdout for analysis
    if report:
        print("---- isolation forest quality metrics ", end = "")
        print("-" * 42, end = "\n\n")
        print(mscores)
    # return results that can get picked up by decorators
    return mdata, mparams, mscores


if __name__ == "__main__":
    # _ = fit_isolation_forest(report = True, random_seed = 7)
    pass