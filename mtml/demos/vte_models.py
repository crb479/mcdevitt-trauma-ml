__doc__ = """Fit a few models on the VTE dataset.

The use of model pipelines allows us to preserve the affine scaling usually done
in the typical standardization step. For convenience, we omit the ``bmi`` column
and drop the 15 missing values in the ``age`` column (no way to fill these in).
"""

import numpy as np
import pandas as pd
import os.path
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# pylint: disable=relative-beyond-top-level
from .. import VTE_INPUT_COLS, VTE_CONT_INPUT_COLS, VTE_OUTPUT_COLS
from ..data.vte import vte_slp_factory
from ..feature_selection.univariate import roc_auc_score_func
from ..utils.persist import persist_csv, persist_json, persist_pickle

# absolute path to this file
OUR_PATH = os.path.dirname(os.path.abspath(__file__))


def _replace_hdl_tot_chol_with_ratio(df):
    """Replace ``tot_cholesterol_result`` and ``hdl_result`` with their ratio.

    :param df: :class:`pandas.DataFrame` containing the VTE data.
    :type df: :class:`pandas.DataFrame`
    :rtype: :class:`pandas.DataFrame`
    """
    # get ratio of total cholesterol and HDL
    ratio = df["tot_cholesterol_result"].values / df["hdl_result"].values
    # drop both original columns and add new column as tot_chol_over_hdl
    df.drop(columns = ["tot_cholesterol_result", "hdl_result"], inplace = True)
    df = df.assign(tot_chol_over_hdl = ratio)
    # done, so return
    return df


@persist_csv(target = OUR_PATH + "/results/vte_selected_cols.csv",
             enabled = True,
             out_transform = lambda x: x[3])
@persist_csv(target = OUR_PATH + "/results/vte_linear_scores.csv",
             enabled = True,
             out_transform = lambda x: x[2])
@persist_json(target = OUR_PATH + "/results/vte_linear_params.json",
             enabled = True,
             out_transform = lambda x: x[1])
@persist_pickle(target = OUR_PATH + "/results/vte_linear.pickle",
                enabled = True,
                out_transform = lambda x: x[0])
def fit_linear_classifiers(cv = 5, n_jobs = -1, verbose = False,
                           report = False, random_seed = None):
    """Fit a few linear classifiers on the data.

    See module docstring for rationale behind dropping ``bmi`` column and our
    missing value removal policy. Positive class has a ~7.73% class frequency
    while the negative class has a ~92.27% class frequency. The simple approach
    is to use balanced class weights, which is done for all the models.

    Use 5-fold (by default) cross-validation to choose the best parameters,
    refit on best, evaluate accuracy, precision, and recall.

    Pipeline and scaler required since regularization is being applied.

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
        data_transform = _replace_hdl_tot_chol_with_ratio,
        inputs = VTE_CONT_INPUT_COLS, targets = VTE_OUTPUT_COLS, dropna = True,
        random_state = random_seed
    )
    # use univariate ROC AUC scores to select the top 7 and get a reduced
    # matrix. note that we have to wrap in lambda in order to pass seed.
    # note: might want to tune the classifier later/try linear SVM instead?
    skbest = SelectKBest(
        lambda x, y: roc_auc_score_func(x, y, random_state = random_seed), k = 7
    )
    # first fit so we can get feature mask using get_support
    skbest.fit(X_train, y_train)
    col_mask = skbest.get_support()
    # get reduced X_train, X_test by selecting only the 7 selected columns
    X_train_red = X_train[:, col_mask]
    X_test_red = X_test[:, col_mask]
    # create small DataFrame of selected columns and their AUC scores, sorted in
    # descending order of the ROC AUC scores
    kbest_aucs = pd.DataFrame(
        data = skbest.scores_[col_mask].reshape(-1, 1),
        index = np.array(VTE_CONT_INPUT_COLS)[col_mask],
        columns = ["roc_auc_score"]
    )
    kbest_aucs.sort_values(kbest_aucs.columns[0],
                           inplace = True, ascending = False)
    # list of base estimator names for use in Pipeline and parameter naming
    base_names = (
        "l2_logistic", "l1_logistic", "en_logistic", "l2_linsvc", "l1_linsvc"
    )
    ## hyperparameter grids for each model pipeline ##
    # note that intercepts are not fitted since data will be centered and
    # standardized by the StandardScaler in the Pipeline. the pipeline params
    # are formatted as [name]__[param_name].
    # l2-regularized logistic regression (baseline model, use on X_train_red)
    lrc_l2_grid = {
        f"{base_names[0]}__penalty": ["l2"],
        f"{base_names[0]}__C": [0.2, 1, 5],
        f"{base_names[0]}__fit_intercept": [False],
        f"{base_names[0]}__class_weight": ["balanced"],
        f"{base_names[0]}__max_iter": [100, 150],
    }
    # l1-regularized logistic regression (use on X_train)
    lrc_l1_grid = {
        f"{base_names[1]}__penalty": ["l1"],
        f"{base_names[1]}__C": [0.2, 1, 5],
        f"{base_names[1]}__fit_intercept": [False],
        f"{base_names[1]}__class_weight": ["balanced"],
        # supports l1 penalty and is preferred over saga is data isn't too large
        f"{base_names[1]}__solver": ["liblinear"],
        # liblinear shuffles data, so pass random_seed
        f"{base_names[1]}__random_state": [random_seed],
        f"{base_names[1]}__max_iter": [100, 150]
    }
    # elastic net regularized logistic regression (use on X_train)
    lrc_en_grid = {
        f"{base_names[2]}__penalty": ["elasticnet"],
        f"{base_names[2]}__C": [0.2, 1, 5],
        f"{base_names[2]}__fit_intercept": [False],
        f"{base_names[2]}__class_weight": ["balanced"],
        # only solver that supports elastic net
        f"{base_names[2]}__solver": ["saga"],
        # saga stochastic, so pass random_seed
        f"{base_names[2]}__random_state": [random_seed],
        f"{base_names[2]}__max_iter": [100, 150],
        f"{base_names[2]}__l1_ratio": [0.25, 0.5, 0.75]
    }
    # linear SVM with l2 penalty (use on X_train_red)
    lsvc_l2_grid = {
        f"{base_names[3]}__penalty": ["l2"],
        f"{base_names[3]}__loss": ["hinge", "squared_hinge"],
        # prefer to solve primal if n_samples > n_features
        f"{base_names[3]}__dual": [True], # for some reason, require dual = True
        # dual is stochastic method so need random_state
        f"{base_names[3]}__random_state": [random_seed],
        f"{base_names[3]}__C": [0.2, 1, 5],
        f"{base_names[3]}__fit_intercept": [False],
        f"{base_names[3]}__class_weight": ["balanced"]
    }
    # linear SVM with l1 penalty (use on X_train)
    lsvc_l1_grid = {
        f"{base_names[4]}__penalty": ["l1"],
        f"{base_names[4]}__loss": ["squared_hinge"], # can't use hinge with l1
        # prefer to solve primal if n_samples > n_features
        f"{base_names[4]}__dual": [False],
        # not stochastic is dual = False
        f"{base_names[4]}__C": [0.2, 1, 5],
        f"{base_names[4]}__fit_intercept": [False],
        f"{base_names[4]}__class_weight": ["balanced"]
    }
    # Pipelines with base estimators to use in grid search
    base_models = (
        Pipeline(
            steps = [("std_scaler", StandardScaler()), (base_name, base_model)]
        )
        for base_name, base_model in zip(
            base_names, (LogisticRegression(), LogisticRegression(),
                         LogisticRegression(), LinearSVC(), LinearSVC())
        )
    )
    # grid search parameters for all the models
    param_grids = (
        lrc_l2_grid, lrc_l1_grid, lrc_en_grid, lsvc_l2_grid, lsvc_l1_grid
    )
    # references to the appropriate data sets that each classifier is fit on
    train_refs = (X_train_red, X_train, X_train, X_train_red, X_train)
    # references to the appropriate data sets used for validation
    val_refs = (X_test_red, X_test, X_test, X_test_red, X_test)
    # dictionary to hold saved model results for VTE classification problem.
    mdata = {}
    # dictionary to hold saved model hyperparameters for plaintext persistence.
    mparams = {}
    # DataFrame indexed by name of the model where columns are accuracy,
    # precision, and recall for each model
    mscores = pd.DataFrame(
        index = base_names, columns = ["accuracy", "precision", "recall"]
    )
    # for each model, train + record results into mdata, mparams, and mscores
    for base_name, base_model, param_grid, train_ref, val_ref in zip(
        base_names, base_models, param_grids, train_refs, val_refs):
        # instantiate and fit the GridSearchCV object. may spit mad warnings.
        model = GridSearchCV(base_model, param_grid,
                             cv = cv, n_jobs = n_jobs, verbose = int(verbose))
        # note that train_ref is either X_train or X_train_red
        model.fit(train_ref, y_train)
        # save model to mdata using model name
        mdata[base_name] = model
        # get hyperparameters of the best estimated model (no StandardScaler)
        params = model.best_estimator_.named_steps[base_name].get_params()
        # save hyperparameters to mparams
        mparams[base_name] = params
        # compute test predictions using refit model on val_ref
        y_pred = model.predict(val_ref)
        # save accuracy, precision, and recall to in mscores
        mscores.loc[base_name, :] = (
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred)
        )
    # if report is True, print top selected features + mscores to stdout
    if report:
        print("---- top 7 features by univariate ROC AUC ", end = "")
        print("-" * 38)
        print(kbest_aucs)
        print("---- linear classifier accuracy, precision, recall ", end = "")
        print("-" * 29)
        print(mscores)
    # return results that can get picked up by decorators
    return mdata, mparams, mscores, kbest_aucs


@persist_csv(target = OUR_PATH + "/results/vte_boosting_scores.csv",
             enabled = True,
             out_transform = lambda x: x[2])
@persist_json(target = OUR_PATH + "/results/vte_boosting_params.json",
              enabled = True,
              out_transform = lambda x: x[1])
@persist_pickle(target = OUR_PATH + "/results/vte_boosting.pickle",
                enabled = True,
                out_transform = lambda x: x[0])
def fit_boosting_classifiers(cv = 5, n_jobs = -1, verbose = False, 
                             report = False, random_seed = None):
    """Fit a few [tree] boosting classifiers on the data.

    See module docstring for rationale behind dropping ``bmi`` column and our
    missing value removal policy. The dataset is very unbalanced: the positive
    class has a ~7.73% class frequency while the negative class has a ~92.27%
    class frequency. The AdaBoost model is able to balance this by forcing
    trees to use balanced class weighting, while XGBoost has a parameter to
    control the control the balance of positive/negative examples, the
    ``scale_pos_weight`` parameter.

    Use 5-fold (by default) cross-validation to choose the best parameters,
    refit on best, evaluate accuracy, precision, and recall.

    Note that tree models are scaling insensitive, so no scaler/pipeline needed.

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
    # get data set from vte_slp_factory
    X_train, X_test, y_train, y_test = vte_slp_factory(
        data_transform = _replace_hdl_tot_chol_with_ratio,
        inputs = VTE_INPUT_COLS, targets = VTE_OUTPUT_COLS, dropna = True,
        random_state = random_seed
    )
    # standard hyperparameters for a depth-3 decision tree using entropy metric.
    dtc_params = {
        "criterion": "entropy", 
        "max_depth": 3, 
    }
    # force the tree to weight class weights to compensate for imbalance.
    dtc_w_params = {
        "criterion": "entropy",
        "max_depth": 3,
        "class_weight": "balanced"
    }
    ## hyperparameter grids for each model ##
    # adaboost with depth-3 trees (unbalanced + balancec)
    ada_grid = {
        "base_estimator": [DecisionTreeClassifier(**dtc_params),
                           DecisionTreeClassifier(**dtc_w_params)],
        "n_estimators": [50, 100, 200, 400],
        "random_state": [random_seed]
    }
    # gbt model with depth-3 trees (unbalanced only, i hate this API). also
    # does stochastic gradient boosting through subsampling.
    gbt_grid = {
        "max_depth": [3],
        "n_estimators": [50, 100, 200, 400],
        "subsample": [0.5, 1],
        "random_state": [random_seed]
    }
    # compute ratio of 0 instances to 1 instances to get XGBoost
    # scale_pos_weight parameter (use training data only! don't be biased)
    neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    # XGBoost model with depth-3 trees (unbalanced + balanced). also does
    # stochastic gradient boosting through subsampling. use single thread to
    # preclude thread contention with GridSearchCV
    xgb_grid = {
        "max_depth": [3],
        "n_estimators": [50, 100, 200, 400],
        "learning_rate": [0.1],
        "booster": ["gbtree"],
        "subsample": [0.5, 1],
        "reg_lambda": [0.1, 1, 5],
        "scale_pos_weight": [neg_pos_ratio],
        "random_state": [random_seed]
    }
    # base estimators to use in grid search
    base_models = (
        AdaBoostClassifier(), GradientBoostingClassifier(), XGBClassifier()
    )
    # grid search parameters for all the models
    param_grids = (ada_grid, gbt_grid, xgb_grid)
    # dictionary to hold saved model results for VTE classification problem.
    mdata = {}
    # dictionary to hold saved model hyperparameters for plaintext persistence.
    mparams = {}
    # DataFrame indexed by name of the model where columns are accuracy,
    # precision, and recall for each model
    mscores = pd.DataFrame(
        index = tuple(map(lambda x: x.__class__.__name__, base_models)),
        columns = ["accuracy", "precision", "recall"]
    )
    # fortunately, we don't need to use Pipeline (no scaling here). for each
    # model, just train and record the results into mdata, mparams, and mscores
    for base_model, param_grid in zip(base_models, param_grids):
        # instantiate and fit the GridSearchCV object. use precision as score
        # since the data set is really unbalanced so accuracy will of course
        # be very high if we just predict the majority class (0)
        model = GridSearchCV(base_model, param_grid,
                             cv = cv, n_jobs = n_jobs, verbose = int(verbose))
        # get name of the base model
        model_name = base_model.__class__.__name__
        model.fit(X_train, y_train)
        # save model to mdata using model name
        mdata[model_name] = model
        # get hyperparameters of the best estimated model
        params = model.best_estimator_.get_params()
        # if there are any predictors as a parameter, replace them with their
        # parameters from get_params (for AdaBoost)
        for name in params.keys():
            if hasattr(params[name], "get_params"):
                params[name] = params[name].get_params()
        # save hyperparameters to mparams
        mparams[model_name] = params
        # compute test predictions using refit model
        y_pred = model.predict(X_test)
        # save accuracy, precision, and recall to in mscores
        mscores.loc[model_name, :] = (
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred)
        )
    # if report is True, print mscores to stdout
    if report:
        print(mscores)
    # return results that can get picked up by decorators
    return mdata, mparams, mscores


if __name__ == "__main__":
    # _ = fit_boosting_classifiers(report = True, random_seed = 7)
    pass