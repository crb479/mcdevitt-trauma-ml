__doc__ = "Test functions for persisting models returned by functions."

import numpy as np
import os.path
import pickle
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ..modeling.utils import persist_pickle


def test_persist_pickle():
    """Persist a logistic regression model returned from a function to disk.
    
    This works for any Python object, not just scikit-learn models. Tests both
    use of decorator without arguments and with target file + persistence
    function that extracts what is to be pickled from the function value.
    """
    # toy function returning a logistic regression model the achieved test
    # accuracy as a tuple. decorate with persist_pickle.
    def train_lr(X_train, X_test, y_train, y_test, **kwargs):
        lrc = LogisticRegression(**kwargs)
        lrc.fit(X_train, y_train)
        return lrc, lrc.score(X_test, y_test)
    
    # persistence function: we just want the model, not the accuracy
    def persist_func(res):
        return res[0]
    
    # get iris data
    X_train, X_test, y_train, y_test = train_test_split(
        *load_iris(return_X_y = True), test_size = 0.2,
        random_state = np.random.RandomState(7)
    )
    # call without arguments first (pickles everything)
    tli_no_args = persist_pickle(train_lr)
    res = tli_no_args(X_train, X_test, y_train, y_test)
    # save accuracy
    acc = res[1]
    # directory this file is in
    abspath = os.path.dirname(os.path.abspath(__file__))
    # call with persist function and specific target
    tli_persist = persist_pickle(
        persist_func = persist_func, target = abspath + "/.lrc.pickle"
    )(train_lr)
    res = tli_persist(X_train, X_test, y_train, y_test)
    # load model and check that accuracy is the same
    with open(abspath + "/.lrc.pickle", "rb") as pf:
        lrc = pickle.load(pf)
    assert acc == lrc.score(X_test, y_test)