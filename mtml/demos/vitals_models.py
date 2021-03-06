__doc__ = """Create classification models from SF vitals data.

Intended for demonstration purposes only.
"""

import json
import numpy as np
import os.path
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# pylint: disable=relative-beyond-top-level
from ..data.sf import cls_vitals_mof, cls_vitals_mort, cls_vitals_trauma
from ..utils.persist import persist_json, persist_pickle

# get file's absolute path
our_path = os.path.dirname(os.path.abspath(__file__))


@persist_json(target = our_path + "/results/vitals_models_params.json",
              enabled = True,
              out_transform = lambda x: x[1])
@persist_pickle(target = our_path + "/results/vitals_models.pickle",
                enabled = True,
                out_transform = lambda x: x[0])
def fit_classifiers_dropna(scaler = "standard", dropna = False, cv = 3, 
                           n_jobs = -1, verbose = False, persist_report = True, 
                           report = False, random_state = None):
    """Makes a few linear classification models, dropping missing data.
    
    Pass ``None`` to ``random_state`` for stochastic behavior. Models used are
    :math:`l_2` penalized logistic regression, RBF kernel SVM, and CART tree.
    Hyperparameters are searched over using 3-fold cross validation with simple
    starting hyperparameter sets. Estimators are refit using best
    hyperparameters on training data, accuracy, precision, recall evaluated on
    test data.

    :param random_state: :class:`numpy.random.RandomState` instance. If not
        provided, this function's fitting behavior may be stochastic.
    :type random_state: :class:`numpy.random.RandomState`, optional
    :param scaler: What kind of input scaling to use. Default is ``"standard"``
        to center + make data unit variance, ``"minmax"`` uses min-max scaling.
    :type scaler: str, optional
    :param dropna: ``True`` to drop missing values from data sets.
    :type dropna: bool, optional
    :param cv: Number of CV splits to make when doing grid search.
    :type cv: int, optional
    :param n_jobs: Number of jobs to run in parallel when grid searching.
        Defaults to ``-1`` to distribute load to all threads.
    :type n_jobs: int, optional
    :param persist: Persist the models to disk using ``pickle`` and the model
        hyperparameters to disk using ``json``. The pickled object is ``mdata``,
        and the ``json`` persisted object is ``mparams``. If ``report`` is
        ``True``, then the scoring results will also be saved to disk as CSV.
    :type persist: bool, optional
    :param verbose: Verbose output when fitting, default ``False``.
    :type verbose: bool, optional
    :param report: If ``True``, print to stdout a report on model scores.
    :type report: bool, optional
    :returns: A tuple of 3 dicts. The first is ``mdata``, dict of 3 dicts, each
        corresponding to a classification problem, each dict with 3 key-value
        mappings containing one of the three model types mentioned in the
        beginning of the docstring. The second is ``mparams`` a dict of 3 dicts,
        each corresponding to a classification problem, each dict with 3 key-
        value mappings containing a list of ``(name, hyperparams)`` tuples for
        each of the three model types. The last is a dict of 3 DataFrames, each
        with a 3 x 3 matrix of model types x accuracy + precision + recall for
        each of the model types, each DataFrame for a particular classification
        problem.
    :rtype: tuple
    """
    # globally use a particular scaler
    if scaler == "standard":
        scaler_name = scaler
        scaler = StandardScaler
    elif scaler == "minmax":
        scaler_name = scaler
        scaler = MinMaxScaler
    else:
        raise ValueError("scaler must be \"standard\" or \"minmax\"")
    # if random_state is None, create a new one
    if random_state is None:
        random_state = np.random.RandomState()
    # dictionary to hold saved model results per classification task
    mdata = {}
    # dictionary to hold saved model hyperparameters for plaintext persistence.
    # note that Pipeline steps are saved as list of 2-tuples (name, param_dict).
    mparams = {}
    ## hyperparameter grids for each model (auto-balance class weight) ##
    # logistic regression. use L-BFGS with l2 regularization + balanced class
    lrc_grid = {"C": [1.2, 1., 0.8],
                "class_weight": ["balanced"],
                "max_iter": [150]}
    # svm. use RBF kernel + balanced class
    svc_grid = {"C": [1.2, 1.],
                "class_weight": ["balanced"]}
    # CART tree. use entropy, best split method, cap depth. balance classes.
    dtc_grid = {"criterion": ["entropy"],
                "max_depth": [6, 8, 14],
                "class_weight": ["balanced"]}
    # for each of the data set creation functions, make vitals data
    for data_func in (cls_vitals_mof, cls_vitals_mort, cls_vitals_trauma):
        # create key for this classification task that's another dict.
        mdata[data_func.__name__] = {}
        mparams[data_func.__name__] = {}
        # get data; use smaller test set (since we will cross-validate to pick
        # our hyperparameters) since our data set is small without NA data.
        X_train, X_test, y_train, y_test = data_func(
            dropna = True, test_size = 0.1, random_state = random_state
        )
        # make three Pipelines: one for logistic regression, one for rbf SVM,
        # and one for CART tree classifier. use GridSearchCV for tuning.
        lrc = Pipeline(
            [(scaler_name, scaler()),
             ("grid_search",
              GridSearchCV(LogisticRegression(), lrc_grid, cv = cv,
                           n_jobs = n_jobs, verbose = int(verbose)))]
        )
        svc = Pipeline(
            [(scaler_name, scaler()),
             ("grid_search",
              GridSearchCV(SVC(), svc_grid, cv = cv, n_jobs = n_jobs,
                           verbose = int(verbose)))]
        )
        dtc = Pipeline(
            [(scaler_name, scaler()),
             ("grid_search",
              GridSearchCV(DecisionTreeClassifier(random_state = 7), dtc_grid,
                           cv = cv, n_jobs = n_jobs, verbose = int(verbose)))]
        )
        # fit all pipelines
        lrc.fit(X_train, y_train)
        svc.fit(X_train, y_train)
        dtc.fit(X_train, y_train)
        # save Pipelines in entirety to mdata
        mdata[data_func.__name__]["lrc"] = lrc
        mdata[data_func.__name__]["svc"] = svc
        mdata[data_func.__name__]["dtc"] = dtc
        # save list of (name, hyperparams) for scaler and best fitted model
        mparams[data_func.__name__]["lrc"] = [
            (scaler_name, lrc.named_steps[scaler_name].get_params()), 
            # note all estimators are refitted by default on all training data
            # with the best determined hyperparameters
            ("grid_search", 
             lrc.named_steps["grid_search"].best_estimator_.get_params())
        ]
        mparams[data_func.__name__]["svc"] = [
            (scaler_name, svc.named_steps[scaler_name].get_params()), 
            ("grid_search", 
             svc.named_steps["grid_search"].best_estimator_.get_params())
        ]
        mparams[data_func.__name__]["dtc"] = [
            (scaler_name, dtc.named_steps[scaler_name].get_params()), 
            ("grid_search", 
             dtc.named_steps["grid_search"].best_estimator_.get_params())
        ]
    # dict of DataFrames for model scores
    scores_dict = {}
    # evaluate scores on test data; print to screen if True
    # for each classification problem
    for cp, cdata in mdata.items():
        if report:
            print(f"{cp} task results:")
        # make DataFrame of the scoring results
        scores_dict[cp] = pd.DataFrame(
            index = ["lrc", "svc", "dtc"],
            columns = ["accuracy", "precision", "recall"]
        )
        # for all model names
        for mn in ("lrc", "svc", "dtc"):
            # compute test predictions
            y_pred = cdata[mn].predict(X_test)
            # compute metrics and fill DataFrame row
            scores_dict[cp].loc[mn, :] = (
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred),
                recall_score(y_test, y_pred)
            )
        # print metrics for this problem
        if report:
            print(scores_dict[cp])
    # if persist is True, persist report results to disk
    if persist_report:
        # get file's absolute path
        our_path = os.path.dirname(os.path.abspath(__file__))
        # save score report as CSV
        for cp, cscores in scores_dict.items():
            # need to print index to identify model
            cscores.to_csv(our_path + "/results/vitals_" + cp + ".csv")
    # return mdata, mparams, and scores_dict to caller
    return mdata, mparams, scores_dict


if __name__ == "__main__":
    # fit models and show scores of best cross-validated model
    # doesn't work if run from command line due to package discovery failure
    # reproduce original results using random_state = np.random.RandomState(7)
    #_ = fit_classifiers_dropna(verbose = True, report = True)
    pass