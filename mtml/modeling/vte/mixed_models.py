__doc__ = """Training routines containing mixed classifier types.

.. note::

   This file has the potential to get cluttered like the
   :mod:`mtml.modeling._vte_models` module that is now deprecated.

Mixed classifier training.
"""

# pylint: disable=import-error
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score
)
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import sys
from xgboost import XGBClassifier

# pylint: disable=relative-beyond-top-level
from .. import BASE_RESULTS_DIR
from ... import VTE_CONT_INPUT_COLS, VTE_OUTPUT_COLS
from ...data.vte import vte_slp_factory
from .data_transforms import replace_hdl_tot_chol_with_ratio
from ...utils.persist import persist_csv, persist_json, persist_pickle


@persist_csv(
    target = BASE_RESULTS_DIR + "/vte_whitened_scores.csv",
    enabled = True, out_transform = lambda x: x[2]
)
@persist_json(
    target = BASE_RESULTS_DIR + "/vte_whitened_params.json",
    enabled = True, out_transform = lambda x: x[1]
)
@persist_pickle(
    target = BASE_RESULTS_DIR + "/vte_whitened.pickle",
    enabled = True, out_transform = lambda x: x[0]
)
def fit_pca_whitened_classifiers(
    cv = 5, n_jobs = -1, verbose = False, report = False, random_seed = None
):
    """Fit classifiers to [non-]undersampled PCA-whitened input data.
    
    .. note:: Spits a lot of ``liblinear`` convergence warnings.

    We start with the top 7 columns by univariate ROC AUC for the VTE data.
    We perform a whitening PCA transform of the data and then fit classifiers
    with balanced class weights. Formerly oversampling of the minority class
    was done with the use of a :class:`sklearn.model_selection.PredefinedSplit`
    to prevent the oversampled data from leaking into the validation sets
    during the grid search (all oversampled data appended to end of training
    set and now allowed to be part of validation sets), but the improvement was
    not as much as one would have hoped (actually worse). So we ended up going
    back to just using balanced class weights.

    Use 5-fold (by default) cross-validation to choose the best parameters,
    refit on best, evaluate accuracy, precision, recall, ROC AUC.

    Note that we need a scaler before doing PCA. Use F1 score to pick model.

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
    if cv < 3:
        raise ValueError("cv folds must be 3 or more")
    # use only the top seven columns selected by univariate AUC
    best_cols = list(
        pd.read_csv(
            BASE_RESULTS_DIR + "/vte_selected_cols.csv", index_col = 0
        ).index
    )
    # get data set of continuous features from vte_slp_factory
    X_train, X_test, y_train, y_test = vte_slp_factory(
        data_transform = replace_hdl_tot_chol_with_ratio,
        inputs = best_cols, targets = VTE_OUTPUT_COLS, dropna = True,
        random_state = random_seed
    )
    # fit StandardScaler and use to transform data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    # fit PCA and transform data yet again (whiten)
    pca = PCA(whiten = True, random_state = random_seed)
    pca.fit(X_train)
    X_train, X_test = pca.transform(X_train), pca.transform(X_test)
    # list of base estimator names for use in Pipeline and parameter naming
    base_names = (
        "l2_logistic", "l2_linsvc", "bagged_l2_logistic", "bagged_l2_linsvc",
        "rbf_svc", "xgboost", "random_forest"
    )
    ## hyperparameter grids for each model ##
    # note that intercepts are not fitted since data are centered + scaled.
    # l2-regularized logistic regression (baseline model)
    lrc_l2_grid = dict(
        penalty = ["l2"],
        C = [1],
        fit_intercept = [True],
        max_iter = [100],
        class_weight = ["balanced"]
    )
    # linear SVM with l2 penalty (baseline model)
    lsvc_l2_grid = dict(
        penalty = ["l2"],
        loss = ["hinge", "squared_hinge"],
        dual = [True],
        random_state = [random_seed],
        C = [1, 5, 10],
        fit_intercept = [True],
        class_weight = ["balanced"]
    )
    # bagged logistic regression model with l2 penalty
    bag_lrc_l2_grid = dict(
        base_estimator = [
            LogisticRegression(fit_intercept = True, class_weight = "balanced")
        ],
        n_estimators = [100, 200, 400],
        random_state = [random_seed]
    )
    # bagged linear SVM with l2 penalty (use default parameters + hinge loss)
    bag_lsvc_l2_grid = dict(
        base_estimator = [
            LinearSVC(loss = "hinge", fit_intercept = True,
                      class_weight = "balanced", random_state = random_seed)
            ],
        n_estimators = [100, 200, 400],
        random_state = [random_seed]
    )
    # RBF support vector classifier
    rbf_svc_grid = dict(
        C = [0.1, 1, 5],
        kernel = ["rbf"],
        gamma = ["scale", "auto"],
        class_weight = ["balanced"]
    )
    # compute ratio of 0 instances to 1 instances to get XGBoost
    # scale_pos_weight parameter (use training data only! don't be biased)
    neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    # XGBoost classifier
    xgb_grid = dict(
        max_depth = [3],
        n_estimators = [400, 600, 800],
        learning_rate = [0.1],
        booster = ["gbtree"],
        subsample = [0.5],
        reg_lambda = [0.1, 1],
        random_state = [random_seed],
        scale_pos_weight = [neg_pos_ratio]
    )
    # random forest classifier. note that according to ESL II, full trees are
    # fine to grow and allow you to have one less tuning parameter, but it's
    # still better to limit the overall tree depth.
    rf_grid = dict(
        max_depth = [6, 12, 24],
        n_estimators = [100, 200, 400],
        criterion = ["entropy"],
        random_state = [random_seed],
        class_weight = ["balanced"]
    )
    # models to use in our grid searches
    base_models = (
        LogisticRegression(), LinearSVC(), BaggingClassifier(),
        BaggingClassifier(), SVC(), XGBClassifier(), RandomForestClassifier()
    )
    base_names = (
        "l2_logistic", "l2_linsvc", "bagged_l2_logistic", "bagged_l2_linsvc",
        "rbf_svc", "xgboost", "random_forest"
    )
    # grid search parameters for all the models
    param_grids = (
        lrc_l2_grid, lsvc_l2_grid, bag_lrc_l2_grid, bag_lsvc_l2_grid,
        rbf_svc_grid, xgb_grid, rf_grid
    )
    # dictionary to hold saved model results for VTE classification problem.
    mdata = {}
    # dictionary to hold saved model hyperparameters for plaintext persistence.
    mparams = {}
    # DataFrame indexed by name of the model where columns are accuracy,
    # precision, and recall for each model
    mscores = pd.DataFrame(
        index = base_names, 
        columns = ["accuracy", "precision", "recall", "roc_auc"]
    )
    # for each model, train + record results into mdata, mparams, and mscores
    for base_name, base_model, param_grid, in zip(
        base_names, base_models, param_grids):
        # instantiate and fit the GridSearchCV object. may spit mad warnings.
        model = GridSearchCV(
            base_model, param_grid, scoring = "f1", cv = cv,
            n_jobs = n_jobs, verbose = int(verbose)
        )
        # fit
        model.fit(X_train, y_train)
        # save model to mdata using model name
        mdata[base_name] = model
        # get hyperparameters of the best estimated model
        params = model.best_estimator_.get_params()
        # if there are any predictors as a parameter, replace them with their
        # parameters from get_params (for ensemble models)
        for name in params.keys():
            if hasattr(params[name], "get_params"):
                params[name] = params[name].get_params()
        # save hyperparameters to mparams
        mparams[base_name] = params
        # compute test predictions using refit model on X_test
        y_pred = model.predict(X_test)
        # get decision function values for computing ROC AUC. if it isn't
        # present, try the predict_proba method
        if hasattr(model, "decision_function"):    
            y_pred_scores = model.decision_function(X_test)
        elif hasattr(model, "predict_proba"):
            # we only want probabilities for the greater class
            y_pred_scores = model.predict_proba(X_test)[:, 1]
        else:
            print(
                f"warning: {model.__class__.__name__} can't compute ROC AUC "
                "score; does not have decision_function or predict_proba",
                file = sys.stderr
            )
            y_pred_scores = None
        # save accuracy, precision, and recall to in mscores
        mscores.loc[base_name, :] = (
            accuracy_score(y_test, y_pred), precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            np.nan if y_pred_scores is None else roc_auc_score(
                y_test, y_pred_scores
            )
        )
    # if report is True, print mscores to stdout
    if report:
        print("---- classifier quality metrics ", end = "")
        print("-" * 48, end = "\n\n")
        print(mscores)
    # return results that can get picked up by decorators
    return mdata, mparams, mscores


if __name__ == "__main__":
    # fit_pca_whitened_classifiers(report = True, random_seed = 7)
    pass