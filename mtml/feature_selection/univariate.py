__doc__ = "Input feature selection using univariate (per-feature) tests."

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
        
    
def roc_auc_score_func(X, y, random_state = None, test_size = 0.25,
                       classifier = None, **classifier_kwargs):
    """Computes univariate ROC AUC scores for each column of ``X``.

    Intended for use with univariate feature selectors from
    :mod:`sklearn.feature_selection` after being appropriately wrapped. Requires
    that the classifier exposes a ``decision_function`` method.

    ROC AUC computation is unbiased since the univariate model is only fit on a
    subset of the incoming data; the ROC AUC is computed on a held out portion.
    
    Here is an example of how to use the function with the
    :class:`sklearn.feature_selection.SelectKBest` class. Note that in order to
    comply with the required signature for ``score_func``, we make use of
    :func:`functools.partial` to pass a fixed seed to ``random_state``.
    
    .. code:: python3

       from functools import partial
       from sklearn.datasets import load_breast_cancer
       from sklearn.feature_selection import SelectKBest

       from mtml.feature_selection.univariate import roc_auc_score_func

       X, y = load_breast_cancer(return_X_y = True)
       # default value of k parameter is 10
       skbest = SelectKBest(partial(roc_auc_score_func, random_state = 7))
       skbest.fit(X, y)
       # all asserts should pass. note there is no pvalues_ attribute.
       assert skbest.scores_.shape == (30,)
       assert skbest.pvalues_ is None

    :param X: Input data matrix, shape ``(n_samples, n_features)``
    :type X: ndarray-like
    :param y: Output data matrixvector, shape ``(n_samples,)``
    :type y: ndarray-like
    :param random_state: Integer seed or :class:`numpy.random.RandomState`
        passed to the :func:`~sklearn.model_selection.train_test_split` call.
        This makes scoring results from this function deterministic.
    :type random_state: int or :class:`numpy.random.RandomState`, optional
    :param test_size: Value to pass to the ``test_size`` parameter of
        :func:`~sklearn.model_selection.train_test_split`.
    :type test_size: int or float, optional
    :param classifier: A scikit-learn [compatible] classifier type.
    :type classifier: :class:`type` or :class:`abc.ABCMeta`
    :param classifier_kwargs: Keyword arguments to pass to the :func:`__init__`
        method of ``classifier``.
    :returns: An array of ROC AUC scores of shape ``(n_features,)``.
    :rtype: :class:`numpy.ndarray`
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    # if random_state is None, use new numpy.random.RandomState
    if random_state is None:
        random_state = np.random.RandomState()
    # use LogisticRegression as default classifier
    if classifier is None:
        classifier = LogisticRegression
    # instantiate classifier with keyword arguments
    model = classifier(**classifier_kwargs)
    # ROC AUC scores for each feature in X
    auc_scores = np.zeros(X.shape[1])
    # split X, y into training and evaluation sections
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size = test_size, random_state = random_state
    )
    # for each column in X, fit a model on X_train and get discriminant/decision
    # function scores for y using X_val to compute ROC AUC for each. save
    # resulting ROC AUC scores to auc_scores.
    for col_i in range(X.shape[1]):
        # reshape columns into column vectors
        model.fit(X_train[:, col_i].reshape(-1, 1), y_train)
        # use raw decision function values with roc_auc_score
        auc_scores[col_i] = roc_auc_score(
            y_val, model.decision_function(X_val[:, col_i].reshape(-1, 1))
        )
    # return ROC AUC scores
    return auc_scores