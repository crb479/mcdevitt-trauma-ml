__doc__ = "Test functions for persisting models returned by functions."

import json
import numpy as np
import os.path
import pandas as pd
import pickle
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# pylint: disable=relative-beyond-top-level
from ..utils.persist import persist_csv, persist_json, persist_pickle

# directory this file is in
our_path = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.skip(reason = "already tested + creates garbage in the directory")
def test_persist_pickle():
    """Persist a logistic regression model returned from a function to disk.
    
    This works for any Python object, not just scikit-learn models. Tests both
    use of decorator without arguments and with target file + persistence
    function that extracts what is to be pickled from the function value.
    """
    # toy function returning a logistic regression model + the achieved test
    # accuracy as a tuple. decorate with persist_pickle.
    def train_lr(X_train, X_test, y_train, y_test, **kwargs):
        lrc = LogisticRegression(**kwargs)
        lrc.fit(X_train, y_train)
        return lrc, lrc.score(X_test, y_test)
    
    # output transformation function: we just want the model, not the accuracy
    def out_transform(res):
        return res[0]
    
    # get iris data
    X_train, X_test, y_train, y_test = train_test_split(
        *load_iris(return_X_y = True), test_size = 0.2,
        random_state = np.random.RandomState(7)
    )
    # call without arguments first (pickles everything)
    tlr_no_args = persist_pickle(train_lr)
    res = tlr_no_args(X_train, X_test, y_train, y_test)
    # save accuracy
    acc = res[1]
    # call with persist function and specific target
    tlr_persist = persist_pickle(
        out_transform = out_transform, target = our_path + "/.lrc.pickle"
    )(train_lr)
    res = tlr_persist(X_train, X_test, y_train, y_test)
    # load model and check that accuracy is the same
    with open(our_path + "/.lrc.pickle", "rb") as pf:
        lrc = pickle.load(pf)
    assert acc == lrc.score(X_test, y_test)


@pytest.mark.skip(reason = "already tested + creates garbage in the directory")
def test_persist_json():
    """Test persisting Python list + hyperparameters of a decision tree to disk.
    
    Only a few Python objects are natively JSON serializable.
    """
    # returns a list of even numbers up to n, including 0 and n if n even
    def make_evens(n):
        return [i * 2 for i in range(n // 2 + 1)]
    # toy function returning a decision tree model + the achieved test accuracy
    # as a tuple. decorate with persist_json.
    def train_dtc(X_train, X_test, y_train, y_test, **kwargs):
        dtc = DecisionTreeClassifier(**kwargs)
        dtc.fit(X_train, y_train)
        return dtc, dtc.score(X_test, y_test)
    
    # output transformation function: just want model hyperparameters
    def out_transform(res):
        return res[0].get_params()
    
    # get iris data
    X_train, X_test, y_train, y_test = train_test_split(
        *load_iris(return_X_y = True), test_size = 0.2,
        random_state = np.random.RandomState(7)
    )
    # call without arguments first
    te_no_args = persist_json(make_evens)
    res = te_no_args(10)
    # call with persist function and specific target
    tdtc_persist = persist_json(
        out_transform = out_transform, target = our_path + "/.dtc_params.json"
    )(train_dtc)
    res = tdtc_persist(X_train, X_test, y_train, y_test, random_state = 7)
    # save accuracy
    acc = res[1]
    # retrain model and check that accuracy is the same
    with open(our_path + "/.dtc_params.json", "r") as jf:
        dtc_params = json.load(jf)
    res = train_dtc(X_train, X_test, y_train, y_test, **dtc_params)
    assert acc == res[1]


@pytest.mark.skip(reason = "already tested + creates garbage in the directory")
def test_persist_csv():
    "Test persisting an arbitary DataFrame to current directory."
    # returns a simple DataFrame
    def make_df_1():
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # return a simple DataFrame and its column names as tuple
    def make_df_2():
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        return df, df.columns
    # call lambda without arguments (check by hand)
    dec_no_args = persist_csv(make_df_1)
    dec_no_args()
    # call with arguments. specify file to save to and out_transform
    dec_with_args = persist_csv(
        out_transform = lambda x: x[0], target = our_path + "/.make_df.csv"
    )(make_df_2)
    dec_with_args()
    # load DataFrame from csv file and check that the two are equal
    df = pd.read_csv(our_path + "/.make_df.csv", index_col = 0)
    assert df.equals(make_df_2()[0])