__doc__ = """Factory methods to load data into format for supervised learning.

This is the general method that all the dataset-specific factories call.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_slp_from_data(data_path, inputs = None, targets = None, 
                       data_transform = None, data_transform_kwargs = None, 
                       target_transform = None, target_transform_kwargs = None, 
                       dropna = False, na_axis = 0, na_how = "any", 
                       na_thresh = None, na_subset = None, test_size = 0.25, 
                       shuffle = True, random_state = None):
    """Generates supervised learning problems from a data set.
    
    Uses :func:`~sklearn.model_selection.train_test_split` for splitting data
    from the file path specified by ``data_path`` into train/validation sets.
    
    :param inputs: List of string column names to use as input variable data. If
        ``None``, then all the columns except for the last will be used.
    :type inputs: iterable, optional
    :param targets: List of string column names to use as response variable
        data. If ``None``, then the last column only is used.
    :type targets: iterable, optional
    :param data_transform: Function to transform a :class:`pandas.DataFrame`
        containing the ``inputs`` columns joined with the ``targets`` columns.
        Takes a single :class:~pandas.DataFrame` as a positional argument and
        keyword arguments. Returns the modified :class:`~pandas.DataFrame`.
    :type data_transform: function, optional
    :param data_transform_kwargs: Keyword arguments to pass to the function
        passed to ``data_transform_args``.
    :type data_transform_kwargs: dict, optional
    :param target_transform: Function to transform the given columns, which will
        be passed as a :class:`~pandas.DataFrame` into ``transform`` as a
        positional argument with keyword arguments from ``transform_kwargs``.
        ``transform`` should return a :class:`numpy.ndarray`, either a flat
        vector or a 2D matrix.
    :type target_transform: function, optional
    :param target_transform_kwargs: Keyword arguments to pass to ``transform``.
    :type target_transform_kwargs: dict, optional
    :param dropna: ``True`` to drop missing values. Default ``False``.
    :type dropna: bool, optional
    :param na_axis: ``axis`` parameter of :meth:`pandas.DataFrame.dropna`.
    :type na_axis: int, optional
    :param na_how: ``how`` parameter of :meth:`pandas.DataFrame.dropna`.
    :type na_how: str, optional
    :param na_thresh: ``thresh`` parameter of :meth:`pandas.DataFrame.dropna`.
    :type na_thresh: int, optional
    :param na_subset: ``subset`` parameter of :meth:`pandas.DataFrame.dropna`.
    :type na_subset: array-like, optional
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
    # if no data transform, make it identity function
    if data_transform is None:
        data_transform = lambda x: x
    # if no data transform kwargs, make it empty dict
    if data_transform_kwargs is None:
        data_transform_kwargs = {}
    # if no target transform is provided, set as function that returns the
    # underlying numpy array backing the DataFrame/Series
    if target_transform is None:
        target_transform = lambda x: x.values
    if target_transform_kwargs is None:
        target_transform_kwargs = {}
    # load data from file
    df = pd.read_csv(data_path)
    # transform data with data_transform
    df = data_transform(df, **data_transform_kwargs)
    # if inputs is None, use all columns except for last as input columns
    if inputs is None:
        inputs  = df.columns[:-1]
    # if targets is None, use last column only as output column
    if targets is None:
        targets = df.columns[-1]
    # extract input and target columns
    data = df.loc[:, inputs].join(df.loc[:, targets])
    # drop NA values according to NA dropping strategy
    if dropna == True:
        data = data.dropna(axis = na_axis, how = na_how, thresh = na_thresh,
                           subset = na_subset)
    # split into X and y and apply transform to y
    X = data.loc[:, inputs]
    y = data.loc[:, targets]
    # apply transform to targets
    y = target_transform(y, **target_transform_kwargs)
    # flatten if shape is not 1D. more than 2D output is not supported.
    if len(y.shape) == 1:
        pass
    elif len(y.shape) == 2:
        if y.shape[1] == 1:
            y = y.ravel()
    else:
        raise ValueError("output shape must be either (n_rows,) or (n_rows, "
                         "n_outputs); other shapes not supported")
    # use train_test_split and return on only the values of the X DataFrame.
    # note that y is already of type ndarray
    out = train_test_split(X.values, y, test_size = test_size,
                           shuffle = shuffle, random_state = random_state)
    return tuple(out)