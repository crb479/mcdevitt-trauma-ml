__doc__ = """Fit a few models on the VTE dataset.

The use of model pipelines allows us to preserve the affine scaling usually done
in the typical standardization step. For convenience, we omit the ``bmi`` column
and drop the 15 missing values in the ``age`` column (no way to fill these in).
"""

import pandas as pd
import os.path
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# pylint: disable=relative-beyond-top-level
from ..data.vte import vte_slp_factory
from ..utils.persist import persist_csv, persist_json, persist_pickle

# absolute path to this file
OUR_PATH = os.path.dirname(os.path.abspath(__file__))

# input columns to use (note we drop BMI, which has 6000+ missing values, and
# patient_identifier, which is useless to us). also drop pt_result which is the
# prothrombin column since it relatively correlated with ptt_result. we drop
# tot_cholesterol_result and hdl_result, replace with column of their ratios.
# this is done with _replace_hdl_tot_chol_with_ratio
VTE_INPUT_COLS = [
    "inr_result", "ldl_result", "ptt_result", "d_dimer_result",
    "glucose_result", "crp_result", "fib_result", "trig_result",
    "thrombin_result", "plt_result", "gender_male0_female1", "age",
    "anticoagulant_use_yes1_no0"
]
# output column
VTE_OUTPUT_COLS = ["thrombosis_present_yes1_no0"]


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


@persist_csv(target = OUR_PATH + "/vte_models_scores.csv",
             enabled = True,
             out_transform = lambda x: x[2])
@persist_json(target = OUR_PATH + "/vte_models_params.json",
              enabled = True,
              out_transform = lambda x: x[1])
@persist_pickle(target = OUR_PATH + "/vte_models.pickle",
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