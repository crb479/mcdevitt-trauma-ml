__doc__ = """Make supervised learning problems from ``sf_num_data_prep_path``.

``sf_num_data_prep_path`` corresponds to the
``data/prep/sf_trauma_data_num.csv`` data file, which is a miminally
preprocessed version of the raw data located in
``data/raw/sf_trauma_data_num_raw.csv``.

Current format is ``[(cls)(reg)]_[input_data_description]_[problem_type]``,
where ``cls`` is for classification, ``reg`` is for regression. ``problem_type``
for example could be ``mort`` for predicting mortality at discharge, ``mof``
for predicting multiple organ failure, ``trauma`` for predicting whether
someone has suffered significant trauma or not (>15 ISS score), etc.

``input_data_description`` must consist of the following quantifiers, separated
by an underscore: ``demo`` for patient demographics, ``prem`` for the single
feature indicating preexisting disease/comorbidity, ``vitals`` for patient
vitals data, and ``lab`` for patient lab panel data. These correspond to
the headers in ``SF Hospital trauma data-v2.xlsx``
`located in the Google Drive`__.

.. __: https://drive.google.com/drive/folders/1VyFHmTdDq-yMMvj_CPfEcV60Jvb70-RL
   ?usp=sharing
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .. import sf_num_data_prep_path, vitals_cols


def vitals_factory(targets, transform = None, transform_kwargs = None,
                   test_size = 0.25, shuffle = True, random_state = None):
    """Factory for generating classification problems using vitals data.
    
    Uses :func:`~sklearn.model_selection.train_test_split` for splitting data
    from ``data/prep/sf_trauma_data_num.csv`` into train/validation sets.
    
    :param targets: List of string column names in
        ``data/prep/sf_trauma_data_num.csv``, ex. ``["mortality at disch"]``.
    :type targets: iterable
    :param transform: Function to transform the given columns, which will be
        passed as a :class:`~pandas.DataFrame` into ``transform`` as a
        positional argument with keyword arguments from ``transform_kwargs``.
        ``transform`` should return a :class:`numpy.ndarray`, either a flat
        vector or a 2D matrix.
    :type transform: function, optional
    :param transform_kwargs: Keyword arguments to pass to ``transform``.
    :type transform_kwargs: dict, optional
    :param test_size: Float in ``[0, 1]`` to represent fraction of points to
        use for validation data or an int to represent number of points to use
        for validation data. Passed directly to
        :func:`~sklearn.model_selection.train_test_split`.
    :type test_size: float or int, optional
    :param shuffle: Whether or not to shuffle the data before splitting,
        defaults to ``True``. ``False`` leaves split data arranged in ascending
        index order.
    :type shuffle: bool, optional
    :param random_state: Int seed or :class:`~numpy.random.RandomState` to use
        to make multiple calls to this method reproducible. Passed directly to
        :func:`~sklearn.model_selection.train_test_split`.
    :type random_state: int or :class:`~numpy.random.RandomState`, optional
    :returns: A 4-tuple ``(X_train, X_test, y_train, y_test)`` of
        :class:`numpy.ndarray` objects
    :rtype: tuple
    """
    if targets is None:
        raise ValueError("targets is None")
    # if no transform is provided, set as function that returns the underlying
    # numpy array backing the DataFrame/Series
    if transform is None:
        transform = lambda x: x.values
    if transform_kwargs is None:
        transform_kwargs = {}
    # load data/prep/sf_trauma_data_num.csv
    df = pd.read_csv(sf_num_data_prep_path)
    # extract vitals columns
    X = df.loc[:, vitals_cols]
    # choose targets and apply transform
    y = transform(df.loc[:, targets], **transform_kwargs)
    # flatten if shape is not 1D. more than 2D output is not supported.
    if len(y.shape) == 1:
        pass
    elif len(y.shape) == 2:
        if y.shape[1] == 1:
            y = y.ravel()
    else:
        raise ValueError("output shape must be either (n_rows,) or (n_rows, "
                         "n_outputs); other shapes not supported")
    # use train_test_split and return
    out = train_test_split(X, y, test_size = test_size, shuffle = shuffle,
                           random_state = random_state)
    return tuple(out)


def cls_vitals_trauma(test_size = 0.25, shuffle = True, random_state = None):
    """Return vitals data from ``sf_num_data_prep_path`` to classify trauma.
    
    Trauma is indicated when ISS score is greater than 15.

    :param test_size: Float in ``[0, 1]`` to represent fraction of points to
        use for validation data or an int to represent number of points to use
        for validation data. Passed directly to
        :func:`~sklearn.model_selection.train_test_split`.
    :type test_size: float or int, optional
    :param shuffle: Whether or not to shuffle the data before splitting,
        defaults to ``True``. ``False`` leaves split data arranged in ascending
        index order.
    :type shuffle: bool, optional
    :param random_state: Int seed or :class:`~numpy.random.RandomState` to use
        to make multiple calls to this method reproducible. Passed directly to
        :func:`~sklearn.model_selection.train_test_split`.
    :type random_state: int or :class:`~numpy.random.RandomState`, optional
    :returns: A 4-tuple ``(X_train, X_test, y_train, y_test)`` of
        :class:`numpy.ndarray` objects
    :rtype: tuple
    """
    # function for creating 0/1 trauma feature from pandas Series
    def make_trauma_feature(ar):
        return np.array(list(map(lambda x: 1 if x > 15 else 0, ar.values)))
    # use factory method
    return vitals_factory(["iss"], transform = make_trauma_feature,
                          test_size = test_size, shuffle = shuffle,
                          random_state = random_state)


def cls_vitals_mort(test_size = 0.25, shuffle = True, random_state = None):
    """Return vitals data from ``sf_num_data_prep_path`` to classify mortality.
    
    Mortality corresponds to mortality at discharge.

    :param test_size: Float in ``[0, 1]`` to represent fraction of points to
        use for validation data or an int to represent number of points to use
        for validation data. Passed directly to
        :func:`~sklearn.model_selection.train_test_split`.
    :type test_size: float or int, optional
    :param shuffle: Whether or not to shuffle the data before splitting,
        defaults to ``True``. ``False`` leaves split data arranged in ascending
        index order.
    :type shuffle: bool, optional
    :param random_state: Int seed or :class:`~numpy.random.RandomState` to use
        to make multiple calls to this method reproducible. Passed directly to
        :func:`~sklearn.model_selection.train_test_split`.
    :type random_state: int or :class:`~numpy.random.RandomState`, optional
    :returns: A 4-tuple ``(X_train, X_test, y_train, y_test)`` of
        :class:`numpy.ndarray` objects
    :rtype: tuple
    """
    return vitals_factory(["mortality at disch"], test_size = test_size,
                          shuffle = shuffle, random_state = random_state)


def cls_vitals_mof(test_size = 0.25, shuffle = True, random_state = None):
    """Return vitals data from ``sf_num_data_prep_path`` to classify MOF.
    
    MOF = multiple organ failure.

    :param test_size: Float in ``[0, 1]`` to represent fraction of points to
        use for validation data or an int to represent number of points to use
        for validation data. Passed directly to
        :func:`~sklearn.model_selection.train_test_split`.
    :type test_size: float or int, optional
    :param shuffle: Whether or not to shuffle the data before splitting,
        defaults to ``True``. ``False`` leaves split data arranged in ascending
        index order.
    :type shuffle: bool, optional
    :param random_state: Int seed or :class:`~numpy.random.RandomState` to use
        to make multiple calls to this method reproducible. Passed directly to
        :func:`~sklearn.model_selection.train_test_split`.
    :type random_state: int or :class:`~numpy.random.RandomState`, optional
    :returns: A 4-tuple ``(X_train, X_test, y_train, y_test)`` of
        :class:`numpy.ndarray` objects
    :rtype: tuple
    """
    return vitals_factory(["mof"], test_size = test_size, shuffle = shuffle,
                          random_state = random_state)